import os
import logging
import torch
import gc
from threading import Thread
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer, BitsAndBytesConfig

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class Qwen25Chat:
    """Qwen2.5 Chat with 4-bit quantization for efficient CUDA inference."""
    
    def __init__(self, model_id="huihui-ai/Qwen2.5-7B-Instruct-abliterated-v2"):
        """Initialize the model with 4-bit quantization."""
        logging.debug(f"Initializing Qwen25Chat with model_id: {model_id}")
        self.model_id = model_id
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        logging.info(f"Using device: {self.device}, dtype: {self.dtype}")
        
        # Enable TF32 for faster computation on Ampere+ GPUs
        if torch.cuda.is_available():
            logging.debug("Enabling TF32 for faster computation")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # Initialize model and tokenizer
        self._setup_model()
        logging.debug("Initialization complete")
        
    def _setup_model(self):
        """Load model with 4-bit quantization configuration."""
        logging.debug("Setting up model with 4-bit quantization")
        # Configure 4-bit quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=self.dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        
        # Load tokenizer
        logging.debug(f"Loading tokenizer from {self.model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id,local_files_only=True,)
        
        # Load model with 4-bit quantization
        logging.debug(f"Loading model from {self.model_id}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=quantization_config,
            device_map="auto",
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        
        # Optimize for inference
        logging.debug("Setting model to eval mode and clearing cache")
        self.model.eval()
        torch.cuda.empty_cache()

    def _unload_model(self):
        """Unload model and tokenizer to free up VRAM."""
        logging.debug("Unloading model and tokenizer...")
        if hasattr(self, 'model'):
            del self.model
            self.model = None
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
            self.tokenizer = None
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.debug("Model unloaded and CUDA cache cleared")
        return True
    
    def generate(self, prompt, max_new_tokens=2048, streaming=False):
        """
        Generate a response from the model.
        
        Args:
            prompt: Text prompt
            max_new_tokens: Maximum tokens to generate
            streaming: If True, returns a generator for streaming output
            
        Returns:
            Generated text (or generator if streaming=True)
        """
        # Check if model components are loaded
        if not hasattr(self, 'model') or self.model is None or not hasattr(self, 'tokenizer') or self.tokenizer is None:
            logging.info("Model components not loaded. Reloading...")
            self._setup_model()

        logging.debug(f"Generating response for prompt: '{prompt[:50]}...' with streaming={streaming}")
        
        messages = [{"role": "user", "content": prompt}]
        
        # Prepare inputs
        with torch.inference_mode():
            logging.debug("Preparing inputs and applying chat template")
            inputs = self.tokenizer.apply_chat_template(
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
        result = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            logging.debug("Clearing CUDA cache")
            torch.cuda.empty_cache()
        
        return result
    
    def _generate_streaming(self, inputs, max_new_tokens):
        """Streaming generation with thread-based approach."""
        logging.debug("Starting streaming generation")
        streamer = TextIteratorStreamer(
            self.tokenizer, 
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
        """Interactive terminal chat loop."""
        print("=" * 50)
        print("Qwen2.5 Chat (type 'quit' or 'exit' to stop)")
        print(f"Running on {self.device} with {self.dtype}")
        print("=" * 50)
        logging.info("Starting chat loop")
        
        while True:
            # Get prompt
            prompt = input("\nPrompt: ").strip()
            
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
                    max_new_tokens=1024,
                    streaming=True
                ):
                    print(text, end="", flush=True)
                print("\n")  # Newline at the end
                    
            except Exception as e:
                print(f"\nError: {e}")
                logging.error(f"Error in chat loop: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

if __name__ == "__main__":
    # Initialize the chat model
    chat_model = Qwen25Chat()
    
    # Start interactive chat loop
    chat_model.chat_loop()

