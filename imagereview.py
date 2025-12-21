"""
you review sexually explicit images with mature content. \nIf you see a penis, is it attached to a mans body or torso or thighs or a full male body? No woman should have a penis, this can be hard to determine if there is sexual intercouse or anal sex. Usually it can be determined when the penis head can still be seen while the shaft is in the vagina or anus.\nIf you see hands do they have the proper number of fingers?\nIs every arm, leg attached to a body? No limbs are not bending backwards.\nThere are no extra limbs (3 feet, 3 arms, etc.).\nFacial Features are correct, eye color only in iris? \nAre physical poses are correct, nothing anatomically impossible?\nIf you see a visible vagina is clearly defined with labia. \nInspect the image carefully it may be hard to determine. \nIf there are any abnormalities or it has features that fail any of these criteria, image has NOT met standards and is not a quality image. Determine if this image is quality or not.\n [Quality] or [Not-Quality]
"""
# Load model directly - Optimized for single GPU CUDA with 4-bit quantization
import os
import torch
from threading import Thread
from transformers import AutoProcessor, AutoModelForImageTextToText, TextIteratorStreamer, BitsAndBytesConfig
from PIL import Image

# ============== CUDA Configuration ==============
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
print(f"Using device: {DEVICE}, dtype: {DTYPE}")

# Enable TF32 for faster computation on Ampere+ GPUs
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# ============== 4-bit Quantization Config ==============
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=DTYPE,
    bnb_4bit_use_double_quant=True,  # Nested quantization for extra memory savings
    bnb_4bit_quant_type="nf4",  # NormalFloat4 - best for LLMs
)

# ============== Model Loading ==============
MODEL_ID = "huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated"

processor = AutoProcessor.from_pretrained(MODEL_ID)

# Load model with 4-bit quantization (~4GB VRAM instead of ~16GB)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    quantization_config=quantization_config,
    device_map="auto",  # Required for bitsandbytes
    attn_implementation="sdpa",  # Use SDPA (built into PyTorch) instead of flash_attention_2
    low_cpu_mem_usage=True,
)

# Optimize for inference
model.eval()
torch.cuda.empty_cache()

# ============== Demo Inference ==============
demo_image_path = r"C:\Users\jared\Documents\code\local_jarvis\xserver\demetra\tartarus\demetra_in_tartarus-p2_a7_f8_c1.png"
demo_image = Image.open(demo_image_path).convert("RGB") if os.path.exists(demo_image_path) else None

messages = [
    {
        "role": "user",
        "content": [ 
            {"type": "image", "image": demo_image} if demo_image else {"type": "text", "text": "(no image)"},
            {"type": "text", "text": "What animal is on the candy?"}
        ]
    },
]

with torch.inference_mode():  # More efficient than no_grad for inference
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=400)
    print(processor.decode(outputs[0][inputs["input_ids"].shape[-1]:]))

def chat():
    """Simple terminal chat loop with optional image input - CUDA optimized."""
    print("=" * 50)
    print("Qwen3 VL Chat (type 'quit' or 'exit' to stop)")
    print(f"Running on {DEVICE} with {DTYPE}")
    print("=" * 50)
    
    while True:
        # Get image path
        image_input = input("\nImage path(s) (comma separated, or press Enter for no image): ").strip()
        
        if image_input.lower() in ['quit', 'exit']:
            print("Goodbye!")
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
        
        # Get prompt
        prompt = input("Prompt: ").strip()
        
        if prompt.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break
        
        if not prompt:
            print("Error: Please enter a prompt.")
            continue
        
        # Build message content
        content = []
        for img_path in valid_images:
            try:
                # Load image directly with PIL instead of using file:// URL
                image = Image.open(img_path).convert("RGB")
                content.append({"type": "image", "image": image})
                print(f"Loaded: {img_path}")
            except Exception as e:
                print(f"Error loading {img_path}: {e}")

        content.append({"type": "text", "text": prompt})
        
        messages = [{"role": "user", "content": content}]
        
        print("\nThinking...")
        try:
            with torch.inference_mode():  # More efficient than no_grad
                inputs = processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(model.device)
                
                # Set up streaming
                streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
                
                generation_kwargs = dict(
                    **inputs,
                    max_new_tokens=1024,
                    streamer=streamer,
                    use_cache=True,  # Enable KV-cache for faster generation
                )
                
                # Run generation in a separate thread
                thread = Thread(target=model.generate, kwargs=generation_kwargs)
                thread.start()
                
                # Stream output as it's generated
                print("\nAssistant: ", end="", flush=True)
                for text in streamer:
                    print(text, end="", flush=True)
                print("\n")  # Newline at the end
                
                thread.join()
                
            # Clear CUDA cache periodically to prevent memory buildup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"\nError: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

if __name__ == "__main__":
    chat()