
#pip install requests websocket-client

import json
import time
import urllib.request
import urllib.parse
import requests
import websocket # pip install websocket-client
import uuid
import random
import os

# ================= CONFIGURATION =================
COMFY_SERVER = "127.0.0.1:8000" # Local ComfyUI address
WORKFLOW_FILE = "video_wan2_2_14B_i2v_nsfw(v2)_API_3.json" # The file you saved via "Export (API)"
                                                                                                                    #sex from behind, doggystyle                                                                                #nude sitting on rose petals                                         
#INPUT_IMAGE = "C:\\Users\\jared\\Documents\\code\\local_jarvis\\xserver\\autogen\\anime_test\\1766372002_end.png"#"C:\\Users\\jared\\Documents\\code\\local_jarvis\\xserver\\autogen\\shoots\\1766195969_image.png"#"C:\\Users\\jared\\Documents\\ComfyUI\\input\\astronaut.jpg" # The local image you want to send
INPUT_IMAGE = "C:\\Users\\jared\Documents\\code\\local_jarvis\\xserver\demetra\\tartarus\\demetra_in_tartarus-p2_a7_f6_c3.png"
CLIENT_ID = str(uuid.uuid4())

# IDS FROM YOUR JSON (Open video_wan2_2_14B_i2v_nsfw(v2)_API.json to check these!)
#Key of each node in the workflow JSON
# Look for class_type: "LoadImage"
IMAGE_NODE_ID = "97"
# Look for class_type: "CLIPTextEncode" (the one for positive prompt)
PROMPT_NODE_ID = "93"
# Look for class_type: "KSampler" (to randomize seed)
SEED_NODE_ID = "85"
#HIGH MODEL LORA: nsfwsks, and strength
#high model lora "lopi999/Wan2.2-I2V_General-NSFW-LoRA"
HIGH_MODEL_LORA_ID = "143"
LOW_MODEL_LORA_ID = "139"
# =================================================

def upload_image(file_path):
    """Uploads the local image to ComfyUI"""
    with open(file_path, "rb") as file:
        files = {"image": file}
        data = {"overwrite": "true"}
        response = requests.post(f"http://{COMFY_SERVER}/upload/image", files=files, data=data)
    return response.json()

def queue_prompt(workflow):
    """Sends the workflow to the queue"""
    p = {"prompt": workflow, "client_id": CLIENT_ID}
    data = json.dumps(p).encode('utf-8')
    req = urllib.request.Request(f"http://{COMFY_SERVER}/prompt", data=data)
    return json.loads(urllib.request.urlopen(req).read())

def track_progress(ws, prompt_id):
    """Listens to WebSocket for completion"""
    while True:
        out = ws.recv()
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

def get_history(prompt_id):
    """Fetches the final results (filenames)"""
    with urllib.request.urlopen(f"http://{COMFY_SERVER}/history/{prompt_id}") as response:
        return json.loads(response.read())
    
def download_file(filename, subfolder, folder_type):
    """Downloads the generated video/image"""
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    query = urllib.parse.urlencode(data)
    with urllib.request.urlopen(f"http://{COMFY_SERVER}/view?{query}") as response:
        return response.read()
    
# ================= MAIN LOGIC =================
#TODO: unload models after inference to free up VRAM when implementing Mixture of LORA approach
## After your inference - assuming that is where COMFY_SERVER is defined
#requests.post('http://localhost:8188/free', json={'unload_models': True})
#
if __name__ == "__main__":
    # 1. Connect to WebSocket
    ws = websocket.WebSocket()
    ws.connect(f"ws://{COMFY_SERVER}/ws?clientId={CLIENT_ID}")

    
    # 2. Upload Your Image
    print(f"Uploading {INPUT_IMAGE}...")
    upload_resp = upload_image(INPUT_IMAGE)
    server_filename = upload_resp["name"] # The name ComfyUI assigned to it
    print(f"Uploaded as: {server_filename}")

    # 3. Load and Modify Workflow
    with open(WORKFLOW_FILE, "r") as f:
        workflow = json.load(f)
    # Update Image Node
    workflow[IMAGE_NODE_ID]["inputs"]["image"] = server_filename

    #Set LORA strengths
    workflow[HIGH_MODEL_LORA_ID]["inputs"]["strength_model"] = 1.0
    workflow[LOW_MODEL_LORA_ID]["inputs"]["strength_model"] = 0.7

    # Update Prompt Node
    """
    Guidance for writing better motion prompts for Wan2.2-I2V_General-NSFW-LoRA:
    huggingface: lopi999/Wan2.2-I2V_General-NSFW-LoRA
    
    LORA Key Phrase: nsfwsks

    5. Tips for Better Motion
    Describe the motion explicitly in natural language (not tags)
    Use temporal language: "first... then... repeating...", "slow/fast", "rhythmic", "continuously"
    Camera descriptions help: "close-up shot", "camera follows", "POV angle"
    Describe physical details: body parts moving, expressions changing
    Use prompt extension (if available in your workflow) — Wan models benefit from expanded descriptive prompts
    """

    #workflow[PROMPT_NODE_ID]["inputs"]["text"] = "nsfwsks,They are having sex, fucking, thrusts in and out. [penis moves in and out of her vagina], dripping white cum and ejaculation. her ass bouncing, breast jiggle, she maintains eye contact with camera. The hips of the man and woman move in rythem opposite directions. Camera [dolly in] slow"
    #workflow[PROMPT_NODE_ID]["inputs"]["text"] ="nsfwsks, she lifts arms unties string behind neck, top slides off revealing nude breast [uncensored nipples] high detail skin, hips swey, gentle breeze, [Camera slow dolly in] low angle."
    #workflow[PROMPT_NODE_ID]["inputs"]["text"] ="nsfwsks, she bends forward and pulls down her skirt belt and leggings all the way to her feet, she is now full nude, bends back up, reveals uncensored [vagina]. breasts bouncing, hips moving. Focus on full body"
    workflow[PROMPT_NODE_ID]["inputs"]["text"] ="nsfwsks, POV perspective, a woman performs oral sex with rhythmic head movements. She slowly moves her head down, taking him deep into her mouth, then lifts up smoothly before repeating the motion in a steady pace. Her expression shows pleasure and focus. Close-up camera angle capturing the intimate details of the repetitive bobbing motion."
    
    # Randomize Seed (so you don't get the exact same video every time)
    if SEED_NODE_ID in workflow:
        workflow[SEED_NODE_ID]["inputs"]["seed"] = random.randint(1, 1000000000)
    # 4. Queue the Workflow
    response = queue_prompt(workflow)
    prompt_id = response['prompt_id']
    print(f"Job queued! Prompt ID: {prompt_id}")


    # 5. Wait for completion
    track_progress(ws, prompt_id)

    # 6. Download Result
    history = get_history(prompt_id)[prompt_id]
    for node_id in history['outputs']:
        node_output = history['outputs'][node_id]
        
        # Check for videos (gifs/mp4) or images
        output_data = []
        if 'gifs' in node_output: output_data += node_output['gifs']
        if 'videos' in node_output: output_data += node_output['videos']
        if 'images' in node_output: output_data += node_output['images']

        if output_data:
            ts = int(time.time())
            #output_dir = "C:\\Users\\jared\\Documents\\code\\local_jarvis\\xserver\\autogen\\anime_test"
            output_dir = "C:\\Users\\jared\\Documents\\code\\local_jarvis\\xserver\\demetra\\tartarus"
            #save prompt
            prompt_file = os.path.join(output_dir, f"{ts}_prompt.txt")
            with open(prompt_file, "w") as f:
                f.write(workflow[PROMPT_NODE_ID]["inputs"]["text"])
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Save input image
            input_ext = os.path.splitext(INPUT_IMAGE)[1]
            start_filename = f"{ts}_start{input_ext}"
            try:
                with open(INPUT_IMAGE, "rb") as f_in, open(os.path.join(output_dir, start_filename), "wb") as f_out:
                    f_out.write(f_in.read())
                print(f"Saved input image to {os.path.join(output_dir, start_filename)}")
            except Exception as e:
                print(f"Error saving input image: {e}")

            for item in output_data:
                print(f"Downloading {item['filename']}...")
                file_data = download_file(item['filename'], item['subfolder'], item['type'])
                
                ext = os.path.splitext(item['filename'])[1]
                if ext.lower() in ['.mp4', '.mov', '.avi', '.gif', '.webm']:
                    new_filename = f"{ts}_video{ext}"
                else:
                    new_filename = f"{ts}_end{ext}"
                
                output_path = os.path.join(output_dir, new_filename)
                with open(output_path, "wb") as f:
                    f.write(file_data)
                print(f"Saved to {output_path}")

    ws.close()
    """
    ### Critical Setup Check:
    1.  **Node IDs:** Open your `workflow_api.json` in a text editor.
        *   Find the text of your prompt. The number above it (e.g., `"6"`) is your `PROMPT_NODE_ID`.[ 2 (https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGGpWRKBb6nbq5lG5efYia8COLc7i6qMtjYcELAnrrCFa3tXFw8J-1PxgwQlTVO4uBBhSVRn3QC_ROVyaUmpRbqlRHzX8u2BeHrjw3woeh_WN6Tm1Mxn8vBkRQ-tMCHxNqW1E-QDMF8mgJb6CTZUQ==)]
        *   Find `"class_type": "LoadImage"`. The number above it is your `IMAGE_NODE_ID`.
    2.  **Custom Nodes:** If your video workflow uses a specific "Load Image" node (like "Load Image Batch" or a URL loader), the input key might be `directory` or `url` instead of `image`.[ 2 (https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGGpWRKBb6nbq5lG5efYia8COLc7i6qMtjYcELAnrrCFa3tXFw8J-1PxgwQlTVO4uBBhSVRn3QC_ROVyaUmpRbqlRHzX8u2BeHrjw3woeh_WN6Tm1Mxn8vBkRQ-tMCHxNqW1E-QDMF8mgJb6CTZUQ==)] Check the JSON inputs section for that node to be sure.
    """