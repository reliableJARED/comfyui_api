#!/usr/bin/env python3
"""
Text-to-Image Generation Class using ChikMix_V3
Uses diffusers library to generate images from text prompts
Based on digiplay/ChikMix_V3 - a Stable Diffusion 1.5 based model
"""

import sys
import subprocess
from pathlib import Path
import time
import torch
from PIL import Image
import uuid
import os
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline

gpu_count = torch.cuda.device_count()
print(f"Number of GPUs available: {gpu_count}")

# When running in terminal use this to set the GPU
# $env:CUDA_VISIBLE_DEVICES="0"
# python chikmix3_gen.py


class ImageGenerator:
    """
    A class for generating images using ChikMix_V3 diffusion model.
    Supports text-to-image, image-to-image, and inpainting capabilities.
    """
    
    def __init__(self, 
                 model_name="digiplay/ChikMix_V3",
                 use_mps=True,
                 use_cuda=True,
                 photorealistic=True):
        """
        Initialize the ImageGenerator with model loading and dependency checking.
        
        Args:
            model_name (str): Main model for text-to-image and image-to-image
            use_mps (bool): Use Metal Performance Shaders on Mac (if available)
            use_cuda (bool): Use CUDA if available (None=auto-detect)
            photorealistic (bool): Use photorealistic enhancement in prompts
        """
        self.photorealistic = photorealistic
        self.model_name = model_name
        
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
        
        print(f"✅ ImageGenerator (ChikMix_V3) initialized")
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
        model_name = self.model_name
        
        print(f"Loading {pipeline_type} pipeline: {model_name}")
        
        try:
            # Configure model loading parameters with device-specific optimizations
            kwargs = {
                "use_safetensors": True,
            }
            
            # Set torch_dtype based on device - SD1.5 works well with float16 on CUDA
            if self.is_cuda:
                kwargs["torch_dtype"] = torch.float16
            else:
                kwargs["torch_dtype"] = torch.float32
            
            # Load the appropriate pipeline
            print("Downloading/loading model (this may take a few minutes on first run)...")
            if pipeline_type == "inpainting":
                # For inpainting, we need to load from the base model and convert
                pipe = StableDiffusionInpaintPipeline.from_pretrained(model_name, **kwargs)
            elif pipeline_type == "image-to-image":
                pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_name, **kwargs)
            else:
                pipe = StableDiffusionPipeline.from_pretrained(model_name, **kwargs)
            
            pipe = pipe.to(self.device)
            
            # Configure scheduler - DPM++ 2M Karras works well for SD1.5 models
            try:
                from diffusers import DPMSolverMultistepScheduler
                pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                    pipe.scheduler.config,
                    use_karras_sigmas=True,
                    algorithm_type="dpmsolver++"
                )
                print("✓ Configured DPM++ 2M scheduler with Karras sigmas")
            except Exception as e:
                print(f"⚠️  Could not configure DPM++ scheduler, using default: {e}")
            
            # Device-specific optimizations
            if self.is_cuda:
                # CUDA optimizations
                print("🔧 Applying CUDA optimizations...")
                if hasattr(pipe, 'enable_attention_slicing'):
                    pipe.enable_attention_slicing()
                
                # Enable TensorFloat32 for better performance on Ampere+ GPUs
                torch.set_float32_matmul_precision('high')
                print("✓ TensorFloat32 precision enabled for faster matmul")
                
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
                 main_output_dir="C:\\Users\\jared\\Documents\\code\\local_jarvis\\xserver",
                 sub_folder="autogen",
                 shoot_folder="shoots",
                 output_dir=None,
                 num_inference_steps=30,  # SD1.5 typically needs 20-30 steps
                 guidance_scale=7.0,      # 7-8 range works well for SD1.5
                 width=512,               # SD1.5 native resolution
                 height=512,              # SD1.5 native resolution
                 use_enhanced_prompting=True,  # adds ChikMix-specific tags
                 upscale_to_1024=False,   # upscale output to 1024x1024
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
            width (int): Image width (512 or 768 recommended for SD1.5)
            height (int): Image height (512 or 768 recommended for SD1.5)
            use_enhanced_prompting (bool): Add ChikMix-specific style tags
            upscale_to_1024 (bool): Upscale output to 1024x1024 using Lanczos
            **kwargs: Additional generation parameters
        
        Returns:
            str: Path to the generated image, or None if failed
        """
        print(f"🎨 Generating image from text: '{prompt}'")
        
        # Enhance prompt for ChikMix model if requested
        if use_enhanced_prompting:
            if self.photorealistic:
                enhanced_prompt = self._enhance_prompt_for_chikmix(prompt)
                print(f"🎨 Enhanced prompt: '{enhanced_prompt}'")
                prompt = enhanced_prompt
        
        # Count and display token usage
        token_count = self._count_tokens(prompt)
        print(f"CLIP input tokens: {token_count}/77 (max for SD1.5)")
        
        # Get pipeline
        pipe = self._get_pipeline("text-to-image")
        
        # Adjust parameters for device capabilities
        if self.is_cpu:
            # CPU: reduce resolution and steps
            width = min(width, 512)
            height = min(height, 512)
            num_inference_steps = min(num_inference_steps, 20)
            print(f"⚠️  Running on CPU - using optimized settings: {width}x{height}, {num_inference_steps} steps")
        elif self.is_mps:
            # MPS can handle 512x512 well, 768x768 with some optimization
            width = min(width, 768)
            height = min(height, 768)
            num_inference_steps = min(num_inference_steps, 25)
            print(f"🚀 Running on MPS - optimized: {width}x{height}, {num_inference_steps} steps")
        elif self.is_cuda:
            # CUDA can handle higher resolutions
            # SD1.5 works best at 512x512 or 768x768
            print(f"🚀 Running on CUDA - resolution: {width}x{height}, {num_inference_steps} steps")
        
        # Add ChikMix-specific negative prompt for better quality
        negative_prompt = kwargs.pop('negative_prompt', None)
        if not negative_prompt:
            negative_prompt = (
                "lowres, bad anatomy, bad hands, text, error, missing fingers, "
                "extra digit, fewer digits, cropped, worst quality, low quality, "
                "normal quality, jpeg artifacts, signature, watermark, username, "
                "blurry, artist name, deformed, disfigured, mutation, mutated, "
                "ugly, extra limbs, poorly drawn hands, poorly drawn face"
            )
        
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
                    # Use autocast for CUDA to improve performance
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        result = pipe(prompt, **generation_kwargs)
                else:
                    # CPU inference
                    result = pipe(prompt, **generation_kwargs)
                image = result.images[0]
            
            # Upscale to 1024 if requested
            if upscale_to_1024:
                original_size = image.size
                image = image.resize((1024, 1024), Image.Resampling.LANCZOS)
                print(f"🔍 Upscaled from {original_size} to {image.size}")
            
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
            text_file_name = os.path.splitext(str(final_output_path))[0] + "_prompt" + '.txt'
            with open(text_file_name, 'w') as f:
                f.write(prompt)

            return str(final_output_path)
            
        except Exception as e:
            print(f"❌ Error generating image: {e}")
            print(f"💡 Try lowering resolution or steps for more stability")
            print(f"💡 Or enhance your prompt with quality/style details")
            return None
    
    def image_to_image(self,
                       init_image,
                       prompt,
                       output_file="output_i2i.png",
                       main_output_dir="C:\\Users\\jared\\Documents\\code\\local_jarvis\\xserver",
                       sub_folder="autogen",
                       shoot_folder="shoots",
                       output_dir=None,
                       num_inference_steps=30,
                       guidance_scale=7.0,
                       strength=0.75,
                       use_enhanced_prompting=False,
                       **kwargs):
        """
        Generate image from an initial image and text prompt
        
        Args:
            init_image: PIL Image or path to image file
            prompt (str): Text description of desired modifications
            output_file (str): filename to save generated image
            main_output_dir (str): Main directory to save images
            sub_folder (str): Sub-folder within main directory
            shoot_folder (str): Shoot-specific folder within sub-folder
            output_dir (str): Override output directory
            num_inference_steps (int): Number of denoising steps
            guidance_scale (float): Guidance scale for generation
            strength (float): How much to transform the image (0.0-1.0)
            use_enhanced_prompting (bool): Add ChikMix-specific style tags
            **kwargs: Additional generation parameters
        
        Returns:
            str: Path to the generated image, or None if failed
        """
        print(f"🎨 Generating image-to-image: '{prompt}'")
        
        # Load image if path is provided
        if isinstance(init_image, (str, Path)):
            init_image = Image.open(init_image).convert("RGB")
        
        # Enhance prompt for ChikMix model if requested
        if use_enhanced_prompting and self.photorealistic:
            enhanced_prompt = self._enhance_prompt_for_chikmix(prompt)
            print(f"🎨 Enhanced prompt: '{enhanced_prompt}'")
            prompt = enhanced_prompt
        
        # Get pipeline
        pipe = self._get_pipeline("image-to-image")
        
        # Add negative prompt
        negative_prompt = kwargs.pop('negative_prompt', None)
        if not negative_prompt:
            negative_prompt = (
                "lowres, bad anatomy, bad hands, text, error, missing fingers, "
                "extra digit, fewer digits, cropped, worst quality, low quality, "
                "normal quality, jpeg artifacts, signature, watermark, username, "
                "blurry, artist name, deformed, disfigured, mutation, mutated"
            )
        
        # Generation parameters
        generation_kwargs = {
            "image": init_image,
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "strength": strength,
            "negative_prompt": negative_prompt,
        }
        generation_kwargs.update({k: v for k, v in kwargs.items() if v is not None})
        
        try:
            print("Generating... (this may take a minute)")
            with torch.no_grad():
                if self.is_cuda:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        result = pipe(**generation_kwargs)
                else:
                    result = pipe(**generation_kwargs)
                image = result.images[0]
            
            # Save image
            if output_dir:
                target_dir = Path(output_dir)
            else:
                target_dir = Path(main_output_dir) / sub_folder / shoot_folder
            #make directories if they don't exist
            target_dir.mkdir(parents=True, exist_ok=True)
            final_output_path = target_dir / Path(output_file).name
            
            image.save(final_output_path)
            print(f"✅ Image saved to: {final_output_path}")
            
            # Save prompt
            text_file_name = os.path.splitext(str(final_output_path))[0] + "_prompt" + '.txt'
            with open(text_file_name, 'w') as f:
                f.write(prompt)

            return str(final_output_path)
            
        except Exception as e:
            print(f"❌ Error generating image: {e}")
            print(f"💡 Try lowering strength (current: {strength}) for more consistency")
            return None
    
    def inpaint(self,
                init_image,
                mask_image,
                prompt,
                output_file="output_inpaint.png",
                main_output_dir="C:\\Users\\jared\\Documents\\code\\local_jarvis\\xserver",
                sub_folder="autogen",
                shoot_folder="shoots",
                output_dir=None,
                num_inference_steps=30,
                guidance_scale=7.0,
                use_enhanced_prompting=True,
                **kwargs):
        """
        Inpaint an image using a mask and text prompt
        
        Args:
            init_image: PIL Image or path to image file
            mask_image: PIL Image or path to mask file (white=inpaint, black=keep)
            prompt (str): Text description of what to paint in masked area
            output_file (str): filename to save generated image
            main_output_dir (str): Main directory to save images
            sub_folder (str): Sub-folder within main directory
            shoot_folder (str): Shoot-specific folder within sub-folder
            output_dir (str): Override output directory
            num_inference_steps (int): Number of denoising steps
            guidance_scale (float): Guidance scale for generation
            use_enhanced_prompting (bool): Add ChikMix-specific style tags
            **kwargs: Additional generation parameters
        
        Returns:
            str: Path to the generated image, or None if failed
        """
        print(f"🎨 Inpainting image: '{prompt}'")
        
        # Load images if paths are provided
        if isinstance(init_image, (str, Path)):
            init_image = Image.open(init_image).convert("RGB")
        if isinstance(mask_image, (str, Path)):
            mask_image = Image.open(mask_image).convert("RGB")
        
        # Enhance prompt for ChikMix model if requested
        if use_enhanced_prompting and self.photorealistic:
            enhanced_prompt = self._enhance_prompt_for_chikmix(prompt)
            print(f"🎨 Enhanced prompt: '{enhanced_prompt}'")
            prompt = enhanced_prompt
        
        # Get pipeline
        pipe = self._get_pipeline("inpainting")
        
        # Add negative prompt
        negative_prompt = kwargs.pop('negative_prompt', None)
        if not negative_prompt:
            negative_prompt = (
                "lowres, bad anatomy, bad hands, text, error, missing fingers, "
                "extra digit, fewer digits, cropped, worst quality, low quality, "
                "normal quality, jpeg artifacts, signature, watermark, username, "
                "blurry, artist name, deformed, disfigured, mutation, mutated"
            )
        
        # Generation parameters
        generation_kwargs = {
            "image": init_image,
            "mask_image": mask_image,
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "negative_prompt": negative_prompt,
        }
        generation_kwargs.update({k: v for k, v in kwargs.items() if v is not None})
        
        try:
            print("Generating... (this may take a minute)")
            with torch.no_grad():
                if self.is_cuda:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        result = pipe(**generation_kwargs)
                else:
                    result = pipe(**generation_kwargs)
                image = result.images[0]
            
            # Save image
            if output_dir:
                target_dir = Path(output_dir)
            else:
                target_dir = Path(main_output_dir) / sub_folder / shoot_folder
            
            target_dir.mkdir(parents=True, exist_ok=True)
            final_output_path = target_dir / Path(output_file).name
            
            image.save(final_output_path)
            print(f"✅ Image saved to: {final_output_path}")
            
            # Save prompt
            text_file_name = os.path.splitext(str(final_output_path))[0] + "_prompt" + '.txt'
            with open(text_file_name, 'w') as f:
                f.write(prompt)

            return str(final_output_path)
            
        except Exception as e:
            print(f"❌ Error inpainting image: {e}")
            return None
    
    def _enhance_prompt_for_chikmix(self, prompt):
        """
        Enhance prompts with ChikMix-specific tags for better results
        Based on model documentation and sample prompts
        ChikMix excels at photorealistic portraits
        """
        # ChikMix responds well to quality and photography-style prompting
        quality_enhancers = [
            "masterpiece",
            "best quality",
            "photorealistic",
            "detailed"
        ]
        
        # Check if prompt already has quality terms
        prompt_lower = prompt.lower()
        has_quality_terms = any(term in prompt_lower for term in [
            "masterpiece", "best quality", "photorealistic", "8k", "detailed",
            "high quality", "professional"
        ])
        
        if not has_quality_terms:
            # Add quality enhancement (as seen in sample prompts)
            enhanced = f"masterpiece, best quality, photorealistic, {prompt}, detailed, 8k"
        else:
            # Prompt already has quality terms, just clean it up
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
        except Exception as e:
            print(f"❌ Error unloading models: {e}")


if __name__ == "__main__":
    # Example usage
    
    generator = ImageGenerator(use_mps=False, photorealistic=True)
    output_path = generator.text_to_image("low_squat hips almost touching the ground,long straight black hair, iris is light purple, angular features, thin black choker necklace, wearing black lace braw, black knee high nylons, pleated black vinyl minis kirt, black high heels. 2.5D realistic animation, front view",
        output_file="chikmix_test_image.png",
        width=512,
        height=512,
        shoot_folder='chikmix'
    )
    
    # Sample prompt based on ChikMix's strengths (photorealistic portraits)
    # Danbooru tag order: rating -> style -> subject -> features -> clothing -> action -> pose
    # Base character tags (reused across prompts)
    base_char = "1girl, freckles, long_hair, black_hair, green_eyes"
    base_char_solo = f"{base_char}, solo"
    base_char_hetero = f"1boy, {base_char}, hetero"
    base_outfit = "choker,black lace_trim, lingerie, black_thighhighs, green_skirt, pleated_skirt"
    style_tags = "rating:explicit, Realistic anime, comic"
    lighting_tags = "soft lighting"
    
    prompt = [
        f"{style_tags}, fellatio, {base_char_solo}, {base_outfit}, oral, penis, kneeling, pov,{lighting_tags}",
        f"{style_tags}, deepthroat, {base_char_solo}, {base_outfit}, irrumatio, penis, kneeling, pov, tears, saliva,{lighting_tags}",
        f"{style_tags}, cunnilingus, {base_char_hetero}, {base_outfit}, spread_legs, lying, on_back, pussy,{lighting_tags}",
        f"{style_tags}, cunnilingus, {base_char_hetero}, {base_outfit}, sitting_on_face, femdom, ass_focus,{lighting_tags}",
        f"{style_tags}, sex, {base_char_hetero}, {base_outfit}, vaginal, penis, spread_legs, lying, pov,{lighting_tags}",
        f"{style_tags}, anal, {base_char_hetero}, {base_outfit}, sex_from_behind, doggystyle, ass, from_behind,{lighting_tags}",
        f"{style_tags}, handjob, {base_char_solo}, {base_outfit}, penis, pov, kneeling,{lighting_tags}",
        f"{style_tags}, fellatio, {base_char_solo}, {base_outfit}, licking_penis, tongue_out, penis, pov, saliva,{lighting_tags}",
        f"{style_tags}, sex, cowgirl_position, {base_char_hetero}, {base_outfit},  girl_on_top, straddling, pov,{lighting_tags}",
        f"{style_tags}, sex, missionary, {base_char_hetero}, {base_outfit},  lying, on_back, spread_legs, pov,{lighting_tags}",
    ]
    """for p in prompt:
        ts = int(time.time())
        output_path = generator.text_to_image(
            p, 
            output_file=f"{ts}_chikmix_image.png",
            width=512,
            height=512,  # Portrait orientation
            shoot_folder='chikmix'
        )
        #prompt_file = f"{ts}_chikmix_prompt.txt"
        #with open(prompt_file, 'w') as f:
        #    f.write(prompt)
        
        if output_path:
            print(f"Generated image saved at: {output_path}")"""
    
    generator._unload_model()