# Load model directly - Optimized for single GPU CUDA with 4-bit quantization
import os
import logging
import torch
import gc
from threading import Thread
from transformers import AutoProcessor, AutoModelForImageTextToText, TextIteratorStreamer, BitsAndBytesConfig
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

#when running in terminal use this to set the GPU
#$env:CUDA_VISIBLE_DEVICES="0"

class Qwen3VLChat:
    """Qwen3 Vision-Language Chat with 4-bit quantization for efficient CUDA inference."""
    
    def __init__(self, model_id="huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated"):
        """Initialize the model with 4-bit quantization."""
        logging.debug(f"Initializing Qwen3VLChat with model_id: {model_id}")
        self.model_id = model_id
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        logging.info(f"Using device: {self.device}, dtype: {self.dtype}")
        
        # Enable TF32 for faster computation on Ampere+ GPUs
        if torch.cuda.is_available():
            logging.debug("Enabling TF32 for faster computation")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # Initialize model and processor
        self._setup_model()
        logging.debug("Initialization complete")
        
    def _setup_model(self):
        """Load model with 4-bit quantization configuration."""
        logging.debug("Setting up model with 4-bit quantization")
        # Configure 4-bit quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=self.dtype,
            bnb_4bit_use_double_quant=True,  # Nested quantization for extra memory savings
            bnb_4bit_quant_type="nf4",  # NormalFloat4 - best for LLMs
        )
        
        # Load processor
        logging.debug(f"Loading processor from {self.model_id}")
        self.processor = AutoProcessor.from_pretrained(self.model_id, local_files_only=True)
        
        # Load model with 4-bit quantization (~4GB VRAM instead of ~16GB)
        logging.debug(f"Loading model from {self.model_id}")
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            quantization_config=quantization_config,
            device_map="auto",  # Required for bitsandbytes
            attn_implementation="sdpa",  # Use SDPA (built into PyTorch) instead of flash_attention_2
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        
        # Optimize for inference
        logging.debug("Setting model to eval mode and clearing cache")
        self.model.eval()
        torch.cuda.empty_cache()

    def _unload_model(self):
        """Unload model and processor to free up VRAM."""
        logging.debug("Unloading model and processor...")
        if hasattr(self, 'model'):
            del self.model
            self.model = None
        if hasattr(self, 'processor'):
            del self.processor
            self.processor = None
        
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.debug("Model unloaded and CUDA cache cleared")
        return True
    
    def generate(self, prompt, images=None, max_new_tokens=2048, streaming=False):
        """
        Generate a response from the model.
        
        Args:
            prompt: Text prompt
            images: Single or list of PIL Images or paths to image files (optional)
            max_new_tokens: Maximum tokens to generate
            streaming: If True, returns a generator for streaming output
            
        Returns:
            Generated text (or generator if streaming=True)
        """
        # Check if model components are loaded
        if not hasattr(self, 'model') or self.model is None or not hasattr(self, 'processor') or self.processor is None:
            logging.info("Model components not loaded. Reloading...")
            self._setup_model()

        logging.debug(f"Generating response for prompt: '{prompt[:50]}...' with streaming={streaming}")
        # Normalize images to a list
        if images is None:
            image_list = []
        elif not isinstance(images, list):
            image_list = [images]
        else:
            image_list = images

        # Build message content
        content = []
        for img in image_list:
            if isinstance(img, str):
                if os.path.exists(img):
                    try:
                        logging.debug(f"Loading image from path: {img}")
                        pil_img = Image.open(img).convert("RGB")
                        content.append({"type": "image", "image": pil_img})
                    except Exception as e:
                        logging.error(f"Error loading image {img}: {e}")
                else:
                    # If it's a string but not a file, maybe it's a URL or something else, 
                    # but for now we assume local paths. 
                    # If the user passed a non-existent path, we might want to warn.
                    if img: # Only warn if string is not empty
                         logging.warning(f"Warning: Image not found: {img}")
            elif isinstance(img, Image.Image):
                logging.debug("Using provided PIL Image object")
                content.append({"type": "image", "image": img})
        
        content.append({"type": "text", "text": prompt})
        
        messages = [{"role": "user", "content": content}]
        
        # Prepare inputs
        with torch.inference_mode():
            logging.debug("Preparing inputs and applying chat template")
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)
            
            if streaming:
                return self._generate_streaming(inputs, max_new_tokens)
            else:
                return self._generate_standard(inputs, max_new_tokens)
    
    def _generate_standard(self, inputs, max_new_tokens):
        """Non-streaming generation."""
        logging.debug("Starting standard generation")
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=True)
        logging.debug("Decoding output")
        result = self.processor.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            logging.debug("Clearing CUDA cache")
            torch.cuda.empty_cache()
        
        return result
    
    def _generate_streaming(self, inputs, max_new_tokens):
        """Streaming generation with thread-based approach."""
        logging.debug("Starting streaming generation")
        streamer = TextIteratorStreamer(
            self.processor.tokenizer, 
            skip_prompt=True, 
            skip_special_tokens=True
        )
        
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            use_cache=True,
        )
        
        # Run generation in a separate thread
        logging.debug("Starting generation thread")
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Yield tokens as they're generated
        for text in streamer:
            yield text
        
        thread.join()
        logging.debug("Generation thread finished")
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            logging.debug("Clearing CUDA cache")
            torch.cuda.empty_cache()
    
    def chat_loop(self):
        """Interactive terminal chat loop with optional image input."""
        print("=" * 50)
        print("Qwen3 VL Chat (type 'quit' or 'exit' to stop)")
        print(f"Running on {self.device} with {self.dtype}")
        print("=" * 50)
        logging.info("Starting chat loop")
        
        while True:
            # Get image path
            image_input = input("\nImage path(s) (comma separated, or press Enter for no image): ").strip()
            
            if image_input.lower() in ['quit', 'exit']:
                print("Goodbye!")
                logging.info("Exiting chat loop")
                break
            
            # Process multiple images
            image_paths = [p.strip() for p in image_input.split(',')] if image_input else []
            valid_images = []

            # Validate image paths
            for img_path in image_paths:
                if img_path and os.path.exists(img_path):
                    valid_images.append(img_path)
                elif img_path:
                    print(f"Error: File not found: {img_path}")
                    logging.warning(f"File not found: {img_path}")
            
            # Get prompt
            prompt = input("Prompt: ").strip()
            
            if prompt.lower() in ['quit', 'exit']:
                print("Goodbye!")
                logging.info("Exiting chat loop")
                break
            
            if not prompt:
                print("Error: Please enter a prompt.")
                continue
            
            print("\nThinking...")
            try:
                # Generate with streaming
                print("\nAssistant: ", end="", flush=True)
                logging.debug("Calling generate with streaming=True")
                for text in self.generate(
                    prompt=prompt,
                    images=valid_images,
                    max_new_tokens=1024,
                    streaming=True
                ):
                    print(text, end="", flush=True)
                print()  # Newline at the end
                    
            except Exception as e:
                print(f"\nError: {e}")
                logging.error(f"Error in chat loop: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    def demo_inference(self, image_path, prompt):
        """Run a demo inference with the given image and prompt."""
        logging.info("Starting demo inference")
        print(f"\nDemo Inference:")
        print(f"Image: {image_path}")
        print(f"Prompt: {prompt}")
        print(f"Response: ", end="", flush=True)
        
        try:
            response = self.generate(prompt=prompt, images=image_path, max_new_tokens=1024)
            print(response)
            return response
        except Exception as e:
            print(f"Error: {e}")
            logging.error(f"Error in demo inference: {e}")
            return None

if __name__ == "__main__":
    # Initialize the chat model
    chat_model = Qwen3VLChat()
    
    # Optional: Run demo inference
    chat_model.demo_inference([], """
    You create pornographic NSFW content.  Create a female character and develop solo and heterosexual prompts.  Some mostly nude, some clothed or bits of clothing.

    1. **Underscores**: Always use underscored Danbooru tags for: sexual_position, sex_acts, perspective,  mixing with natural language for the rest of the prompt
    2. **Front-load important concepts**:only describe the woman, man will auto generate simply using '1boy' as needed 
    PROMPT STRUCTURE:\nFacial features, eyewear, hair color and style, iris color, accets like mascara, long lashes, eye shadow.\nClothing or nude, nake\nBody type fit, athletic, curvy, thin\ncharacter count, either 1girl, or 1girl 1boy,\nSexual Postion, Act and Perspective tags\nLocation in 1 or 2 words hotel room, hot tub, bed room, forest, cabin, etc.\nLighting 
    3. **Photography tags have major visual impact**: Camera types and lighting dramatically affect the output
    4. **Use commas to separate concepts**
    5. **Parentheses/weight syntax doesn't work** in raw diffusers - they're treated as literal characters
    6. **Quality matters less than content**: Focus on describing what you want rather than quality tags
    7. **Experiment with hybrid approaches**: Mix tags and natural language for best results

    ### Body Features & Modifiers
    - nude, naked, topless, bottomless
    - breasts, small_breasts, large_breasts
    - nipples, pussy, penis, erection
    - spread_legs, legs_apart
    - straddling
    - arched_back


    ### Clothing States
    - lingerie, underwear, panties, bra
    - torn_clothes, clothes_pull
    - partially_undressed
    - stockings, thigh_highs, pantyhose
    - sheer_legwear

    ### Intimacy & Expression
    - sex, hetero
    - kissing, french_kiss
    - looking_at_viewer, eye_contact
    - seductive_smile, open_mouth
    - sweat, saliva
    
    ### Character Count
    - 1girl, 1boy 
    - 1girl, solo 

    ### Common Sexual Positions
    - missionary, missionary_position
    - sex_from_behind, doggystyle
    - cowgirl_position, girl_on_top, woman_on_top
    - reverse_cowgirl
    - standing_sex
    - spooning
    - 69_position

    ### Sexual Acts
    - fellatio, oral, blowjob, deepthroat
    - vaginal, penetration, sex
    - handjob
    - titjob, paizuri
    - anal


    ### Perspectives & Focus
    - pov, pov_crotch
    - from_behind, from_below, from_above
    - close-up, wide_shot
    - male_focus, female_focus


    ### Lighting Types
    - cinematic lighting
    - soft lighting
    - warm golden hour lighting
    - dramatic lighting
    - low key lighting
    - neon lighting
    - bright flash photography
    - radiant god rays

    First - Come up with a femal description - it must be consistent throughout.  Then come up with a combination of 10 sexual positions, acts and perspectives. Return XML schema with each comma separated sting image prompts, inside of 10 <scene></scene> tags
    """)

    # Optional: Run demo inference
    demo_image_path = r"C:\Users\jared\Documents\code\local_jarvis\xserver\autogen\anime_test\1766358994_image.png"
    if os.path.exists(demo_image_path):
        chat_model.demo_inference(demo_image_path, "create a detailed description of the character, 25 words, using tags suitable for image generation on Stable Diffusion. Focus on appearance, clothing, features, and style. Be concise and specific. since the image prompt space is limited.")
    
    # Start interactive chat loop
    chat_model.chat_loop()