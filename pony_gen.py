#!/usr/bin/env python3
"""
Pony Diffusion V6 XL Text-to-Image Generation Class
Uses Bakanayatsu/Pony-Diffusion-V6-XL-for-Anime with LyliaEngine LoRA
"""

import sys
import subprocess
from pathlib import Path
import torch
from PIL import Image
import uuid
import os
from diffusers import DiffusionPipeline

gpu_count = torch.cuda.device_count()
print(f"Number of GPUs available: {gpu_count}")


class PonyImageGenerator:
    """
    A class for generating images using Pony Diffusion V6 XL with LoRA.
    Optimized for anime-style generation.
    """
    
    def __init__(self, 
                 use_mps=True,
                 use_cuda=None):
        """
        Initialize the PonyImageGenerator with model loading and dependency checking.
        
        Args:
            use_mps (bool): Use Metal Performance Shaders on Mac (if available)
            use_cuda (bool): Use CUDA if available (None=auto-detect)
        """
        self.base_model = "Bakanayatsu/Pony-Diffusion-V6-XL-for-Anime"
        self.lora_model = "LyliaEngine/Pony_Diffusion_V6_XL"
        
        # Enhanced device configuration for Mac and CUDA
        self.device = self._get_best_device(use_mps, use_cuda)
        self.is_cpu = self.device == "cpu"
        self.is_mps = self.device == "mps"
        self.is_cuda = self.device == "cuda"
        
        # Install dependencies and check
        self._install_dependencies()
        self._check_dependencies()
        
        # Initialize pipeline (will be loaded on demand)
        self.pipe = None
        
        print(f"✅ PonyImageGenerator initialized")
        print(f"Device: {self.device}")
        if self.is_cuda:
            print("🚀 Using CUDA GPU acceleration!")
        elif self.is_mps:
            print("🚀 Using Metal Performance Shaders for acceleration!")
    
    def _get_best_device(self, use_mps=True, use_cuda=None):
        """Determine the best available device with CUDA priority"""
        if use_cuda is None:
            use_cuda = torch.cuda.is_available()
        
        if use_cuda and torch.cuda.is_available():
            return "cuda"
        elif use_mps and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _install_dependencies(self):
        """Install required packages automatically"""
        required_packages = [
            "torch",
            "diffusers", 
            "pillow",
            "transformers",
            "accelerate"
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package if package != "pillow" else "PIL")
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            print("Missing required packages. Installing automatically...")
            print(f"Installing: {', '.join(missing_packages)}")
            
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "--upgrade"
                ] + missing_packages)
                print("✓ Dependencies installed successfully!")
                
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to install dependencies: {e}")
                print("Please install manually:")
                print(f"pip install {' '.join(missing_packages)}")
                sys.exit(1)
            except Exception as e:
                print(f"❌ Unexpected error during installation: {e}")
                print("Please install manually:")
                print(f"pip install {' '.join(missing_packages)}")
                sys.exit(1)
    
    def _check_dependencies(self):
        """Verify all dependencies are properly installed"""
        try:
            import torch
            import diffusers
            from PIL import Image
            import transformers
            import accelerate
            print("✓ All dependencies verified")
        except ImportError as e:
            print(f"❌ Dependency check failed: {e}")
            print("Please run: pip install torch diffusers pillow transformers accelerate")
            sys.exit(1)
    
    def _load_pipeline(self):
        """
        Load the Pony Diffusion V6 XL pipeline with LoRA weights.
        
        Returns:
            Pipeline object with LoRA weights loaded
        """
        print("Loading Pony Diffusion V6 XL pipeline with LoRA...")
        
        try:
            # Determine dtype based on device
            if self.is_cuda:
                dtype = torch.bfloat16
            elif self.is_mps:
                dtype = torch.float32  # MPS doesn't support bfloat16 well
            else:
                dtype = torch.float32
            
            print(f"Loading base model: {self.base_model}")
            pipe = DiffusionPipeline.from_pretrained(
                self.base_model,
                torch_dtype=dtype,
                use_safetensors=True,
            )
            pipe = pipe.to(self.device)
            
            # Load LoRA weights
            print(f"Loading LoRA weights: {self.lora_model}")
            pipe.load_lora_weights(self.lora_model)
            print("✓ LoRA weights loaded successfully")
            
            # Apply device-specific optimizations
            if self.is_cuda:
                print("🔧 Applying CUDA optimizations...")
                if hasattr(pipe, 'enable_attention_slicing'):
                    pipe.enable_attention_slicing()
                try:
                    if hasattr(pipe, 'enable_xformers_memory_efficient_attention'):
                        pipe.enable_xformers_memory_efficient_attention()
                        print("✓ XFormers memory efficient attention enabled")
                except Exception as e:
                    print(f"⚠️  XFormers not available: {e}")
            elif self.is_mps:
                print("🔧 Applying MPS optimizations...")
                if hasattr(pipe, 'enable_attention_slicing'):
                    pipe.enable_attention_slicing(1)
            else:
                # CPU optimizations
                if hasattr(pipe, 'enable_attention_slicing'):
                    pipe.enable_attention_slicing(1)
                torch.set_num_threads(torch.get_num_threads())
                print(f"🔧 Using {torch.get_num_threads()} CPU threads")
            
            print("✅ Pony Diffusion V6 XL pipeline loaded successfully!")
            return pipe
            
        except Exception as e:
            print(f"❌ Error loading Pony pipeline: {e}")
            print("Make sure you have sufficient disk space and internet connection.")
            raise
    
    def _get_pipeline(self):
        """Get or load the Pony Diffusion pipeline"""
        if self.pipe is None:
            self.pipe = self._load_pipeline()
        return self.pipe
    
    def _enhance_prompt(self, prompt):
        """
        Enhance prompts with Pony Diffusion-specific quality tags.
        Pony models respond well to score-based quality tags.
        """
        quality_prefix = "score_9, score_8_up, score_7_up, score_6_up, score_5_up, score_4_up"
        
        # Check if prompt already has score tags
        if "score_" in prompt.lower():
            return prompt
        
        enhanced = f"{quality_prefix}, {prompt}"
        return enhanced
    
    def text_to_image(self,
                      prompt,
                      output_file="pony_output.png",
                      main_output_dir="C:\\Users\\jared\\Documents\\code\\local_jarvis\\xserver",
                      sub_folder="autogen",
                      shoot_folder="pony_shoots",
                      output_dir=None,
                      num_inference_steps=30,
                      guidance_scale=7.0,
                      width=1024,
                      height=1024,
                      use_enhanced_prompting=True,
                      **kwargs):
        """
        Generate image from text prompt using Pony Diffusion V6 XL with LoRA.
        
        Args:
            prompt (str): Text description of desired image
            output_file (str): filename to save generated image
            main_output_dir (str): Main directory to save images
            sub_folder (str): Sub-folder within main directory
            shoot_folder (str): Shoot-specific folder within sub-folder
            output_dir (str): Override full output directory path
            num_inference_steps (int): Number of denoising steps
            guidance_scale (float): Guidance scale for generation
            width (int): Image width (1024 recommended for SDXL-based models)
            height (int): Image height (1024 recommended for SDXL-based models)
            use_enhanced_prompting (bool): Add Pony-specific quality score tags
            **kwargs: Additional generation parameters
        
        Returns:
            str: Path to generated image, or None on failure
        """
        print(f"🎨 [Pony] Generating image from text: '{prompt}'")
        
        # Enhance prompt with Pony quality tags if requested
        if use_enhanced_prompting:
            enhanced_prompt = self._enhance_prompt(prompt)
            print(f"🎨 [Pony] Enhanced prompt: '{enhanced_prompt}'")
            prompt = enhanced_prompt
        
        # Get pipeline
        pipe = self._get_pipeline()
        
        # Device-specific adjustments
        if self.is_cpu:
            width = min(width, 768)
            height = min(height, 768)
            num_inference_steps = min(num_inference_steps, 20)
            print(f"⚠️  Running on CPU - optimized settings: {width}x{height}, {num_inference_steps} steps")
        elif self.is_mps:
            num_inference_steps = min(num_inference_steps, 25)
            print(f"🚀 Running on MPS: {width}x{height}, {num_inference_steps} steps")
        elif self.is_cuda:
            print(f"🚀 Running on CUDA: {width}x{height}, {num_inference_steps} steps")
        
        # Default negative prompt for Pony
        negative_prompt = kwargs.pop('negative_prompt', None)
        if not negative_prompt:
            negative_prompt = "score_1, score_2, score_3, blurry, low quality, distorted, deformed, extra limbs, bad anatomy, ugly, worst quality"
        
        # Generation parameters
        generation_kwargs = {
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "height": height,
            "width": width,
            "negative_prompt": negative_prompt,
        }
        generation_kwargs.update({k: v for k, v in kwargs.items() if v is not None})
        
        try:
            print("[Pony] Generating... (this may take a minute)")
            with torch.no_grad():
                if self.is_mps:
                    with torch.autocast(device_type='cpu', enabled=False):
                        result = pipe(prompt, **generation_kwargs)
                elif self.is_cuda:
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        result = pipe(prompt, **generation_kwargs)
                else:
                    result = pipe(prompt, **generation_kwargs)
                image = result.images[0]
            
            # Construct the target directory path
            if output_dir:
                target_dir = Path(output_dir)
            else:
                target_dir = Path(main_output_dir) / sub_folder / shoot_folder
            
            # Ensure directory exists
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Construct final file path
            final_output_path = target_dir / Path(output_file).name
            
            image.save(final_output_path)
            print(f"✅ [Pony] Image saved to: {final_output_path}")
            
            # Save prompt to text file for reference
            text_file_name = os.path.splitext(str(final_output_path))[0] + "_prompt.txt"
            with open(text_file_name, 'w') as f:
                f.write(prompt)
            
            return str(final_output_path)
            
        except Exception as e:
            print(f"❌ [Pony] Error generating image: {e}")
            return None
    
    def unload_model(self):
        """Unload the pipeline to free up memory"""
        print("Unloading Pony Diffusion model...")
        try:
            if self.pipe:
                del self.pipe
                self.pipe = None
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print("✓ CUDA cache cleared")
                
                print("✅ Pony model unloaded successfully")
        except Exception as e:
            print(f"❌ Error unloading Pony model: {e}")


if __name__ == "__main__":
    # Example usage
    generator = PonyImageGenerator()
    prompt = "beautiful female anthro portrait, dramatic lighting, dark background"
    output_path = generator.text_to_image(prompt, output_file="pony_test.png")
    print(f"Generated image saved at: {output_path}")
    generator.unload_model()
