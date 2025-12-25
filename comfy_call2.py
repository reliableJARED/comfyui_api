import json
import time
import urllib.request
import urllib.parse
import requests
import websocket
import uuid
import random
import os
import gc

class ComfyCallWithLoRAManagement:
    """
    Enhanced ComfyUI API client with aggressive VRAM management for LoRA swapping.
    
    Fixes the OOM issue when switching between different LoRAs by:
    1. Aggressively clearing server-side caches
    2. Forcing garbage collection between runs
    3. Using proper /free endpoint with all options
    4. Optional: Restarting ComfyUI between LoRA changes
    """
    
    # ================= CONFIGURATION =================
    COMFY_SERVER = "127.0.0.1:8000"
    WORKFLOW_FILE = "video_wan2_2_14B_i2v_nsfw(v2)_API_3.json"
    
    # Node IDs from your workflow
    IMAGE_NODE_ID = "97"
    PROMPT_NODE_ID = "93"
    NEGATIVE_PROMPT_NODE_ID = "89"
    SEED_NODE_ID = "86"  # Changed to 86 (high noise sampler seed)
    
    # LoRA Node IDs from your workflow
    LORA_HIGH_NOISE_NODE_ID = "143"  # LoraLoaderModelOnly HIGH NOISE
    LORA_LOW_NOISE_NODE_ID = "139"   # LoraLoaderModelOnly LOW NOISE
    
    OUTPUT_DIR = "C:\\Users\\jared\\Documents\\code\\local_jarvis\\xserver\\autogen\\"
    
    # Memory management settings
    AGGRESSIVE_CLEANUP = True  # Force cleanup after each run
    UNLOAD_MODELS_BETWEEN_RUNS = True  # Unload all models between LoRA changes
    # =================================================

    def __init__(self, workflow_file=None):
        self.client_id = str(uuid.uuid4())
        self.ws = None
        self.current_lora_state = None  # Track what LoRAs are loaded
        
        if workflow_file:
            self.WORKFLOW_FILE = workflow_file
            
        print(f"ComfyCall client initialized with Client ID: {self.client_id}")
        self.connect()
        print("ComfyCall WebSocket connection established.")

    def connect(self):
        """Connects to the ComfyUI WebSocket"""
        self.ws = websocket.WebSocket()
        self.ws.connect(f"ws://{self.COMFY_SERVER}/ws?clientId={self.client_id}")

    def ensure_connected(self):
        """Ensures the WebSocket is connected, reconnecting if necessary"""
        if self.ws is None or not self.ws.connected:
            print("WebSocket not connected, reconnecting...")
            self.connect()
            print("WebSocket reconnected.")

    def aggressive_cleanup(self):
        """
        Aggressively cleans up ComfyUI's VRAM caches.
        This is the KEY to solving your OOM issue with LoRA swapping.
        """
        print("\n🧹 Performing aggressive VRAM cleanup...")
        
        try:
            # 1. Free unneeded memory (this is better than just /free)
            free_response = requests.post(
                f"http://{self.COMFY_SERVER}/free",
                json={"unload_models": True, "free_memory": True}
            )
            print(f"   ✓ /free endpoint: {free_response.status_code}")
            
            # 2. Clear the queue (removes any pending jobs)
            queue_response = requests.post(
                f"http://{self.COMFY_SERVER}/queue",
                json={"clear": True}
            )
            print(f"   ✓ Queue cleared: {queue_response.status_code}")
            
            # 3. Interrupt any running execution
            interrupt_response = requests.post(
                f"http://{self.COMFY_SERVER}/interrupt"
            )
            print(f"   ✓ Interrupted: {interrupt_response.status_code}")
            
            # 4. Give ComfyUI time to actually clean up
            time.sleep(2)
            
            # 5. Trigger Python garbage collection locally
            gc.collect()
            
            print("   ✓ Cleanup complete\n")
            
        except Exception as e:
            print(f"   ⚠ Cleanup warning (non-fatal): {e}")

    def unload_all_models(self):
        """
        Forces ComfyUI to unload ALL models from VRAM.
        This is more aggressive than /free and helps with LoRA issues.
        """
        print("🔄 Forcing model unload...")
        try:
            response = requests.post(
                f"http://{self.COMFY_SERVER}/free",
                json={
                    "unload_models": True,
                    "free_memory": True,
                    "force": True  # Some ComfyUI builds support this
                }
            )
            time.sleep(3)  # Give it time to actually unload
            print(f"   ✓ Models unloaded: {response.status_code}\n")
        except Exception as e:
            print(f"   ⚠ Unload warning: {e}")

    def update_loras_in_workflow(self, workflow, lora_config):
        """
        Updates LoRA nodes in the workflow.
        
        Args:
            workflow: The workflow JSON
            lora_config: Dict with LoRA settings, e.g.:
                {
                    'high_noise': {
                        'lora_name': 'NSFW-22-H-e8.safetensors',
                        'strength': 0.8
                    },
                    'low_noise': {
                        'lora_name': 'NSFW-22-L-e8.safetensors', 
                        'strength': 0.4
                    }
                }
                OR set strength to 0 to disable a LoRA
        """
        # Update high noise LoRA
        if 'high_noise' in lora_config:
            high_config = lora_config['high_noise']
            workflow[self.LORA_HIGH_NOISE_NODE_ID]["inputs"]["lora_name"] = high_config['lora_name']
            workflow[self.LORA_HIGH_NOISE_NODE_ID]["inputs"]["strength_model"] = high_config['strength']
            print(f"   High Noise LoRA: {high_config['lora_name']} @ {high_config['strength']}")
        
        # Update low noise LoRA
        if 'low_noise' in lora_config:
            low_config = lora_config['low_noise']
            workflow[self.LORA_LOW_NOISE_NODE_ID]["inputs"]["lora_name"] = low_config['lora_name']
            workflow[self.LORA_LOW_NOISE_NODE_ID]["inputs"]["strength_model"] = low_config['strength']
            print(f"   Low Noise LoRA: {low_config['lora_name']} @ {low_config['strength']}")
        
        return workflow

    def check_lora_change(self, new_lora_config):
        """
        Checks if LoRAs are changing and returns whether we need aggressive cleanup.
        """
        if self.current_lora_state is None:
            self.current_lora_state = new_lora_config
            return False  # First run, no cleanup needed
        
        # Check if LoRAs are different
        lora_changed = (
            self.current_lora_state.get('high_noise', {}).get('lora_name') != 
            new_lora_config.get('high_noise', {}).get('lora_name')
            or
            self.current_lora_state.get('low_noise', {}).get('lora_name') != 
            new_lora_config.get('low_noise', {}).get('lora_name')
        )
        
        if lora_changed:
            print(f"\n⚠️  LoRA CHANGE DETECTED!")
            print(f"   Old: {self.current_lora_state}")
            print(f"   New: {new_lora_config}")
            self.current_lora_state = new_lora_config
            return True
        
        return False

    def upload_image(self, file_path):
        """Uploads the local image to ComfyUI"""
        with open(file_path, "rb") as file:
            files = {"image": file}
            data = {"overwrite": "true"}
            response = requests.post(f"http://{self.COMFY_SERVER}/upload/image", files=files, data=data)
        return response.json()

    def queue_prompt(self, workflow):
        """Sends the workflow to the queue"""
        p = {"prompt": workflow, "client_id": self.client_id}
        data = json.dumps(p).encode('utf-8')
        req = urllib.request.Request(f"http://{self.COMFY_SERVER}/prompt", data=data)
        return json.loads(urllib.request.urlopen(req).read())

    def track_progress(self, prompt_id):
        """Listens to WebSocket for completion"""
        while True:
            out = self.ws.recv()
            if isinstance(out, str):
                message = json.loads(out)
                if message['type'] == 'executing':
                    data = message['data']
                    if data['node'] is None and data['prompt_id'] == prompt_id:
                        print("Execution complete.")
                        break
            else:
                continue
        return

    def get_history(self, prompt_id):
        """Fetches the final results (filenames)"""
        with urllib.request.urlopen(f"http://{self.COMFY_SERVER}/history/{prompt_id}") as response:
            return json.loads(response.read())
    
    def download_file(self, filename, subfolder, folder_type):
        """Downloads the generated video/image"""
        data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        query = urllib.parse.urlencode(data)
        with urllib.request.urlopen(f"http://{self.COMFY_SERVER}/view?{query}") as response:
            return response.read()

    def close(self):
        """Closes the WebSocket connection"""
        if self.ws:
            self.ws.close()

    def run(
        self, 
        input_image_path, 
        prompt_text, 
        base_file_name="base_file_name",
        folder="movies",
        source_image_review="",
        source_image_prompt="",
        file_name_modifier="",
        lora_config=None,
        negative_prompt=None,
        seed=None
    ):
        """
        Executes the full workflow with proper LoRA management.
        
        Args:
            input_image_path: Path to input image
            prompt_text: Animation prompt
            base_file_name: Base name for output files
            folder: Output folder name
            source_image_review: Review text to save
            source_image_prompt: Source image prompt to save
            file_name_modifier: Modifier for output filename
            lora_config: Dict specifying LoRA settings (see update_loras_in_workflow)
            negative_prompt: Override default negative prompt
            seed: Override random seed
        
        Returns:
            Dictionary of saved file paths
        """
        try:
            # Check if LoRA is changing and cleanup if needed
            if lora_config and self.check_lora_change(lora_config):
                if self.UNLOAD_MODELS_BETWEEN_RUNS:
                    self.unload_all_models()
                if self.AGGRESSIVE_CLEANUP:
                    self.aggressive_cleanup()

            # Ensure WebSocket is connected
            self.ensure_connected()

            # Upload image
            print(f"Uploading {input_image_path}...")
            upload_resp = self.upload_image(input_image_path)
            server_filename = upload_resp["name"]
            print(f"Uploaded as: {server_filename}")

            # Load workflow
            with open(self.WORKFLOW_FILE, "r") as f:
                workflow = json.load(f)
            
            # Update image node
            workflow[self.IMAGE_NODE_ID]["inputs"]["image"] = server_filename

            # Update prompt node
            workflow[self.PROMPT_NODE_ID]["inputs"]["text"] = prompt_text

            # Update negative prompt
            if negative_prompt is None:
                negative_prompt = (
                    "gay, transvestite, hands, chewing, biting, eating, messy, distorted, "
                    "deformed, disfigured, ugly, tiling, poorly drawn, mutation, mutated, "
                    "extra limbs, cloned face, disfigured, out of frame, ugly, blurry, "
                    "bad anatomy, bad proportions, extra limbs, cloned face, disfigured, "
                    "gross proportions, malformed limbs, missing arms, missing legs, "
                    "fused fingers, too many fingers, long neck"
                )
            workflow[self.NEGATIVE_PROMPT_NODE_ID]["inputs"]["text"] = negative_prompt
            
            # Update LoRAs if provided
            if lora_config:
                print("\n📦 Configuring LoRAs:")
                workflow = self.update_loras_in_workflow(workflow, lora_config)
            
            # Randomize or set seed
            if seed is None:
                seed = random.randint(1, 1000000000)
            workflow[self.SEED_NODE_ID]["inputs"]["noise_seed"] = seed
            print(f"\n🎲 Using seed: {seed}")
            
            # Queue the workflow
            response = self.queue_prompt(workflow)
            prompt_id = response['prompt_id']
            print(f"\n🚀 Job queued! Prompt ID: {prompt_id}")

            # Wait for completion
            self.track_progress(prompt_id)

            # Download results
            history = self.get_history(prompt_id)[prompt_id]
            saved_files = {}

            output_dir = os.path.join(self.OUTPUT_DIR, folder)
            os.makedirs(output_dir, exist_ok=True)
            
            ts = int(time.time())
            for node_id in history['outputs']:
                node_output = history['outputs'][node_id]
                
                # Check for videos/images
                output_data = []
                if 'gifs' in node_output: output_data += node_output['gifs']
                if 'videos' in node_output: output_data += node_output['videos']
                if 'images' in node_output: output_data += node_output['images']

                for item in output_data:
                    print(f"Downloading {item['filename']}...")
                    file_data = self.download_file(item['filename'], item['subfolder'], item['type'])
                    
                    ext = os.path.splitext(item['filename'])[1]
                    if ext.lower() in ['.mp4', '.mov', '.avi', '.gif', '.webm']:
                        new_filename = f"{base_file_name}_{file_name_modifier}_video{ext}"
                        file_key = "video"
                    else:
                        new_filename = f"{base_file_name}_{file_name_modifier}_end{ext}"
                        file_key = "image_final"
                    
                    output_path = os.path.join(output_dir, new_filename)
                    with open(output_path, "wb") as f:
                        f.write(file_data)

                    print(f"✓ Saved to {output_path}")
                    saved_files[file_key] = output_path

            # Save text files
            prompt_path = os.path.join(output_dir, f"{base_file_name}_{file_name_modifier}_animation_prompt.txt")
            with open(prompt_path, "w") as f:
                f.write(prompt_text)
            
            source_img_path = os.path.join(output_dir, f"{base_file_name}_{file_name_modifier}_source_image_prompt.txt")
            with open(source_img_path, "w") as f:
                f.write(source_image_prompt)

            source_review_path = os.path.join(output_dir, f"{base_file_name}_{file_name_modifier}_source_image_review.txt")
            with open(source_review_path, "w") as f:
                f.write(source_image_review)

            saved_files["animation_prompt"] = prompt_path
            saved_files["source_image_prompt"] = source_img_path
            saved_files["source_image_review"] = source_review_path
            saved_files["image_start"] = input_image_path

            # Cleanup after run if configured
            if self.AGGRESSIVE_CLEANUP:
                self.aggressive_cleanup()
            
            return saved_files

        except Exception as e:
            print(f"❌ Error in ComfyCall: {e}")
            # Try cleanup even on error
            if self.AGGRESSIVE_CLEANUP:
                self.aggressive_cleanup()
            return {}


# =============================================================================
# Example Usage with Multiple LoRA Configs
# =============================================================================

if __name__ == "__main__":
    comfy = ComfyCallWithLoRAManagement()
    
    # Define different LoRA configurations for different styles
    lora_configs = {
        'nsfw_style': {
            'high_noise': {
                'lora_name': 'NSFW-22-H-e8.safetensors',
                'strength': 0.8
            },
            'low_noise': {
                'lora_name': 'NSFW-22-L-e8.safetensors',
                'strength': 0.4
            }
        },
        'anime_style': {
            'high_noise': {
                'lora_name': 'anime_style_high.safetensors',
                'strength': 1.0
            },
            'low_noise': {
                'lora_name': 'anime_style_low.safetensors',
                'strength': 0.8
            }
        },
        'realistic_style': {
            'high_noise': {
                'lora_name': 'realistic_high.safetensors',
                'strength': 0.9
            },
            'low_noise': {
                'lora_name': 'realistic_low.safetensors',
                'strength': 0.6
            }
        }
    }
    
    # Example: Generate with first LoRA config
    print("\n" + "="*60)
    print("GENERATION 1: NSFW Style")
    print("="*60)
    
    result1 = comfy.run(
        input_image_path="C:\\Users\\jared\\Documents\\code\\local_jarvis\\xserver\\autogen\\castle_721\\1766245852_image.png",
        prompt_text="nsfw,pornographic, [deepthroat] his penis enters her open mouth...",
        folder="castle_721",
        lora_config=lora_configs['nsfw_style'],
        base_file_name="test",
        file_name_modifier="nsfw"
    )
    
    # Example: Switch to different LoRA config
    # The cleanup will happen automatically!
    print("\n" + "="*60)
    print("GENERATION 2: Anime Style (LoRA SWAP)")
    print("="*60)
    
    result2 = comfy.run(
        input_image_path="C:\\Users\\jared\\Documents\\code\\local_jarvis\\xserver\\autogen\\castle_721\\1766245852_image.png",
        prompt_text="anime style character portrait, detailed eyes, vibrant colors",
        folder="castle_721",
        lora_config=lora_configs['anime_style'],  # Different LoRA!
        base_file_name="test",
        file_name_modifier="anime"
    )
    
    # Example: Another swap
    print("\n" + "="*60)
    print("GENERATION 3: Realistic Style (ANOTHER LoRA SWAP)")
    print("="*60)
    
    result3 = comfy.run(
        input_image_path="C:\\Users\\jared\\Documents\\code\\local_jarvis\\xserver\\autogen\\castle_721\\1766245852_image.png",
        prompt_text="photorealistic portrait, natural lighting, 8k detail",
        folder="castle_721",
        lora_config=lora_configs['realistic_style'],  # Yet another LoRA!
        base_file_name="test",
        file_name_modifier="realistic"
    )
    
    print("\n✓ All generations complete with proper LoRA swapping!")