#!/usr/bin/env python3
"""
Text-to-Image Generation Class
Uses diffusers library to generate images from text prompts
Updated to use LUSTIFY-SDXL-NSFW-checkpoint-v2-0-INPAINTING model
"""

import sys
import subprocess
from pathlib import Path
import time
import torch
from PIL import Image
import uuid
import os
from diffusers import DiffusionPipeline, AutoPipelineForInpainting, AutoPipelineForImage2Image
#from lustify_xwork import ImageGenerator

gpu_count = torch.cuda.device_count()
print(f"Number of GPUs available: {gpu_count}")

#when running in terminal use this to set the GPU
#$env:CUDA_VISIBLE_DEVICES="0"
#python lustify_xwork.py


class ImageGenerator:
    """
    A class for generating images using diffusion models.
    Supports text-to-image, image-to-image, and inpainting capabilities.
    """
    
    def __init__(self, 
                 model_name="TheImposterImposters/LUSTIFY-v4.0",#"UnfilteredAI/NSFW-GEN-ANIME","TheImposterImposters/LUSTIFY-v2.0",,#"AI-Porn/pornworks-anime-desire-NSFW-Anime-and-hentai-sdxl-checkpoint",
                 use_mps=True,
                 use_cuda=True,
                 animated=False):
        """
        Initialize the ImageGenerator with model loading and dependency checking.
        
        Args:
            model_name (str): Main model for text-to-image and image-to-image
            use_mps (bool): Use Metal Performance Shaders on Mac (if available)
            use_cuda (bool): Use CUDA if available (None=auto-detect)
        """
        self.animated = animated
        if self.animated:
            self.model_name="UnfilteredAI/NSFW-GEN-ANIME"
        else:  
            self.model_name="TheImposterImposters/LUSTIFY-v4.0"#"TheImposterImposters/LUSTIFY-v2.0"
        
        # Enhanced device configuration for Mac and CUDA
        self.device = self._get_best_device(use_mps, use_cuda)
        self.is_cpu = self.device == "cpu"
        self.is_mps = self.device == "mps"
        self.is_cuda = self.device == "cuda"
        
        # Install dependencies and check
        self._install_dependencies()
        self._check_dependencies()
        
        # Initialize pipelines (will be loaded on demand)
        self.text2img_pipe = None
        self.img2img_pipe = None
        self.inpaint_pipe = None
        
        print(f"✅ ImageGenerator initialized")
        print(f"Device: {self.device}")
        if self.is_cuda:
            print("🚀 Using CUDA GPU acceleration!")
        elif self.is_mps:
            print("🚀 Using Metal Performance Shaders for acceleration!")
    
    def _get_best_device(self, use_mps=True, use_cuda=None):
        """Determine the best available device with CUDA priority"""
        # Auto-detect CUDA if not explicitly specified
        if use_cuda is None:
            use_cuda = torch.cuda.is_available()
        
        if use_cuda and torch.cuda.is_available():
            return "cuda"
        elif use_mps and torch.backends.mps.is_available():
            # MPS (Metal Performance Shaders) for Mac acceleration
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
                # Install missing packages
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
    
    def _load_pipeline(self, pipeline_type="text-to-image"):
        """
        Load the appropriate diffusion pipeline
        
        Args:
            pipeline_type (str): "text-to-image", "image-to-image", or "inpainting"
        
        Returns:
            Pipeline object
        """
        if pipeline_type == "text-to-image":
            model_name = self.model_name
        elif pipeline_type == "image-to-image":
            model_name = self.model_name
        else:
            model_name = self.model_name
        
        print(f"Loading {pipeline_type} pipeline: {model_name}")
        
        try:
            # Configure model loading parameters with device-specific optimizations
            kwargs = {
                "use_safetensors": True,
            }
            
            # Set torch_dtype based on device - SDXL was trained in float16
            if self.is_cuda:
                kwargs["torch_dtype"] = torch.float16  # Native SDXL precision, best quality + speed
            elif self.is_mps:
                kwargs["torch_dtype"] = torch.float32  # MPS works better with float32
            else:
                kwargs["torch_dtype"] = torch.float32  # CPU needs float32
            
            # Load the appropriate pipeline
            print("Downloading/loading model (this may take a few minutes on first run)...")
            if pipeline_type == "inpainting":
                pipe = AutoPipelineForInpainting.from_pretrained(model_name, **kwargs)
            elif pipeline_type == "image-to-image":
                pipe = AutoPipelineForImage2Image.from_pretrained(model_name, **kwargs)
            else:
                pipe = DiffusionPipeline.from_pretrained(model_name, **kwargs)
            
            pipe = pipe.to(self.device)
            
            # Configure scheduler per documentation recommendations (DPM++ 2M SDE with Karras)
            try:
                from diffusers import DPMSolverMultistepScheduler
                pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                    pipe.scheduler.config,
                    use_karras_sigmas=True,
                    algorithm_type="sde-dpmsolver++"  # For DPM++ SDE variant
                )
                print("✓ Configured DPM++ 2M SDE scheduler with Karras sigmas")
            except Exception as e:
                print(f"⚠️  Could not configure DPM++ scheduler, using default: {e}")
            
            # Device-specific optimizations
            if self.is_cuda:
                # CUDA optimizations
                print("🔧 Applying CUDA optimizations...")
                if hasattr(pipe, 'enable_attention_slicing'):
                    pipe.enable_attention_slicing()
                
                # Enable memory efficient attention if available
                try:
                    if hasattr(pipe, 'enable_xformers_memory_efficient_attention'):
                        pipe.enable_xformers_memory_efficient_attention()
                        print("✓ XFormers memory efficient attention enabled")
                except Exception as e:
                    print(f"⚠️ XFormers not available: {e}")
            
            elif self.is_mps:
                # MPS optimizations
                print("🔧 Applying MPS optimizations...")
                if hasattr(pipe, 'enable_attention_slicing'):
                    pipe.enable_attention_slicing(1)  # More aggressive slicing for MPS
                
                # Enable sequential CPU offload for better memory management on Mac
                try:
                    if hasattr(pipe, 'enable_sequential_cpu_offload'):
                        pipe.enable_sequential_cpu_offload()
                        print("Sequential CPU offload enabled for MPS")
                except Exception as e:
                    print(f"⚠️ Sequential CPU offload not available: {e}")
            
            else:
                # CPU optimizations
                if hasattr(pipe, 'enable_attention_slicing'):
                    pipe.enable_attention_slicing(1)
                
                # Set number of threads for CPU inference
                torch.set_num_threads(torch.get_num_threads())
                print(f" Using {torch.get_num_threads()} CPU threads")
            
            print(f"{pipeline_type.title()} pipeline loaded successfully!")
            return pipe
            
        except Exception as e:
            print(f"❌ Error loading {pipeline_type} pipeline: {e}")
            print("Make sure you have sufficient disk space and internet connection.")
            sys.exit(1)
    
    def _get_pipeline(self, pipeline_type):
        """Get or load the requested pipeline"""
        if pipeline_type == "text-to-image":
            if self.text2img_pipe is None:
                self.text2img_pipe = self._load_pipeline("text-to-image")
            return self.text2img_pipe
        elif pipeline_type == "image-to-image":
            if self.img2img_pipe is None:
                self.img2img_pipe = self._load_pipeline("image-to-image")
            return self.img2img_pipe
        elif pipeline_type == "inpainting":
            if self.inpaint_pipe is None:
                self.inpaint_pipe = self._load_pipeline("inpainting")
            return self.inpaint_pipe
        else:
            raise ValueError(f"Unknown pipeline type: {pipeline_type}")
    
    def _count_tokens(self, text):
        """Count CLIP tokens in a prompt (approximate)"""
        try:
            from transformers import CLIPTokenizer
            tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
            tokens = tokenizer(text, truncation=False, return_tensors=None)
            return len(tokens['input_ids'])
        except Exception:
            # Fallback: rough estimate (~1.3 tokens per word)
            return int(len(text.split()) * 1.3)
    
    def text_to_image(self, 
                 prompt, 
                 output_file="output.png",
                 main_output_dir = "C:\\Users\\jared\\Documents\\code\\local_jarvis\\xserver",
                 sub_folder = "autogen",
                 shoot_folder = "shoots",
                 output_dir=None,
                 num_inference_steps=30,  # SDXL: 50-60 steps for better face detail
                 guidance_scale=5.5,      # Lower CFG (5-7) reduces face deformations
                 width=1024,
                 height=1024,
                 use_enhanced_prompting=True,  # adds LUSTIFY-specific tags
                 **kwargs):
        """
        Generate image from text prompt
        
        Args:
            prompt (str): Text description of desired image
            output_file (str): filename to save generated image
            main_output_dir (str): Main directory to save images
            sub_folder (str): Sub-folder within main directory
            shoot_folder (str): Shoot-specific folder within sub-folder
            num_inference_steps (int): Number of denoising steps
            guidance_scale (float): Guidance scale for generation
            width (int): Image width
            height (int): Image height
            use_enhanced_prompting (bool): Add LUSTIFY-specific style tags
            **kwargs: Additional generation parameters
        
        Returns:
            PIL.Image: Generated image
        """
        print(f"🎨 Generating image from text: '{prompt}'")
        
        # Enhance prompt for LUSTIFY model if requested
        if use_enhanced_prompting:
            if not self.animated:
                enhanced_prompt = self._enhance_prompt_for_lustify(prompt)
                print(f"🎨 Enhanced prompt: '{enhanced_prompt}'")
                prompt = enhanced_prompt
        
        # Count and display token usage
        token_count = self._count_tokens(prompt)
        print(f"CLIP input tokens: {token_count}/77 (max for SDXL)")
        
        # Get pipeline
        pipe = self._get_pipeline("text-to-image")
        
        # Adjust parameters for device capabilities while respecting SDXL's optimal resolution
        if self.is_cpu:
            # Even on CPU, try to maintain closer to SDXL's native resolution
            width = min(width, 768)  # Compromise between speed and quality
            height = min(height, 768)
            num_inference_steps = min(num_inference_steps, 20)  # Reduce steps for CPU
            print(f"⚠️  Running on CPU - using optimized settings: {width}x{height}, {num_inference_steps} steps")
        elif self.is_mps:
            # MPS can handle full SDXL resolution better, but reduce steps slightly
            # Keep 1024x1024 as SDXL works best at this resolution
            num_inference_steps = min(num_inference_steps, 25)
            print(f"🚀 Running on MPS - SDXL optimized: {width}x{height}, {num_inference_steps} steps")
        elif self.is_cuda:
            # CUDA can handle full resolution and steps efficiently
            print(f"🚀 Running on CUDA - Full SDXL resolution: {width}x{height}, {num_inference_steps} steps")
        
        # Add LUSTIFY-specific negative prompt for better quality (especially faces)
        negative_prompt = kwargs.pop('negative_prompt', None)
        if not negative_prompt:
            negative_prompt = "blurry, low quality, distorted, deformed, extra limbs, bad anatomy, hands"
        
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
            print("Generating... (this may take a minute)")
            with torch.no_grad():
                # Device-specific inference optimizations
                if self.is_mps:
                    # MPS sometimes has issues with certain operations
                    with torch.autocast(device_type='cpu', enabled=False):
                        result = pipe(prompt, **generation_kwargs)
                elif self.is_cuda:
                    # float16 is SDXL's native precision - best quality and speed
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        result = pipe(prompt, **generation_kwargs)
                else:
                    # CPU inference
                    result = pipe(prompt, **generation_kwargs)
                image = result.images[0]
            
            # Save image
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
            print(f"✅ Image saved to: {final_output_path}")
            # Also save the prompt to a text file for reference
            #prompt_file = final_output_path.with_suffix('.txt')
            text_file_name = os.path.splitext(str(final_output_path))[0]+"_prompt" + '.txt'
            with open(text_file_name, 'w') as f:
                f.write(prompt)

            return str(final_output_path)
            
        except Exception as e:
            print(f"❌ Error generating image: {e}")
            print(f"💡 Try lowering strength (current: strength) for more consistency")
            print(f"💡 Or enhance your prompt with camera/lighting details")
            return None
    
    def _enhance_prompt_for_lustify(self, prompt):
        """
        Enhance prompts with LUSTIFY-specific tags for better results
        Based on model documentation and community findings
        """
        # LUSTIFY responds well to photography-style prompting
        photo_enhancers = [
            "photograph",
            "shot on Canon EOS 5D", 
            "cinematic lighting",
            "professional photography"
        ]
        
        # Check if prompt already has photography terms
        prompt_lower = prompt.lower()
        has_photo_terms = any(term in prompt_lower for term in [
            "shot on", "photograph", "photo", "camera", "lighting", 
            "shot with", "taken with"
        ])
        
        if not has_photo_terms:
            # Add basic photography enhancement
            enhanced = f"photograph, {prompt}, shot on Canon EOS 5D, professional photography"
        else:
            # Prompt already has photo terms, just clean it up
            enhanced = prompt
            
        return enhanced
    
    def _unload_model(self):
        """Unload all pipelines to free up memory"""
        print("Unloading all models to free up memory...")
        try:
            if self.text2img_pipe:
                del self.text2img_pipe
                self.text2img_pipe = None
            if self.img2img_pipe:
                del self.img2img_pipe
                self.img2img_pipe = None
            if self.inpaint_pipe:
                del self.inpaint_pipe
                self.inpaint_pipe = None
            
            # Clear CUDA cache if applicable
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("✓ CUDA cache cleared")
            
            print("✅ All models unloaded successfully")
            return True
        except Exception as e:
            print(f"❌ Error unloading models: {e}")
    
 
if __name__ == "__main__":
    # Example usage
    ts = int(time.time())
    generator = ImageGenerator(use_mps=False, animated=False)
    prompt = "photograph, real photo missionary_position,1girl, a sexy white skin woman, short face framing red hair, glasses, high cheekbones, full lips, white-eye blue iris, long lashes,32B breast, black mascara, she is lying on her back, 1boy, POV, soft lighting, 8k"
    #prompt = "nsfw, low_squat hips almost touching the ground,long straight black hair, iris is light purple, angular features, thin black choker necklace, wearing black lace braw, black knee high nylons, pleated black vinyl minis kirt, black high heels. 2.5D realistic animation, front view"
    #prompt = "nsfw, cosmic background, wide_stance feet planted far apart, chest forward, hips back, long straight black hair, iris is light purple, angular features, thin black choker, black lace braw, black knee high nylons, pleated purple vinyl minis kirt, black high heels. hentai realistic animation, front view"

    output_path = generator.text_to_image(prompt, output_file=f"{ts}_lustify_gen_test.png")
    #save prompt
    prompt_file = output_path.replace(".png", f"{ts}_prompt.txt")
    with open(prompt_file, 'w') as f:
        f.write(prompt)
    print(f"Generated image saved at: {output_path}")
    generator._unload_model()
    
