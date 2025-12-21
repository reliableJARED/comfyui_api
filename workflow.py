"""
FIRST: run qwen3VL_UI.py to start the local API server.

Then run this workflow.py script to execute the workflow.

Orchestration workflow for full prompt to video.
1) Create initial character model traits from text prompt.
2) Create scene concept description from text prompt.
3) Generate action, camera angles, and shots from scene concept from text prompt for the character
4) Generate Images for each shot using the character model traits and scene concept.
5) review images, reject and regenerate as needed.
6) Compile images into video.
7) compare start image, end image, prompt to determine if final video is acceptable.
"""

#Workflow steps use local API endpoints to perform each step.
# e.g. http://localhost:5055/generate
import requests
import time
import re
from PIL import Image
import os
import random
from comfy_call import ComfyCall
import json
from join_mp4 import combine_videos

COMFY = ComfyCall()

def i2v(image_path="", prompt="",folder="neon", source_image_review="", source_image_prompt="",file_name_modifier="1"):
    global COMFY
    """
    Generate a video from an image and prompt using ComfyCall.
    """
    #create a base file name from image_path
    base_filename = os.path.basename(image_path)
    #remove file extension
    base_filename_noext = os.path.splitext(base_filename)[0]

    result = COMFY.run(
        input_image_path=image_path,
        prompt_text=prompt,
        folder=folder,
        source_image_review=source_image_review,
        source_image_prompt=source_image_prompt,
        base_file_name=base_filename_noext,
        file_name_modifier=file_name_modifier
    )
    return result
    
def videogen(clips=1,image_path="", prompt="",folder="neon",source_image_review="", source_image_prompt=""):
    #this function will generate successive videos from a single image and prompt attempting to create a longer animation
    animation_prompt = prompt
    start_image = image_path
    counter = 0
    all_files = []
    video_clips = []
    while counter < clips:
        counter += 1
        print(f"Generating clip {counter} of {clips}...")
        #first generate initial video
        result = i2v(image_path=start_image, prompt=prompt, folder=folder,source_image_review=source_image_review, source_image_prompt=source_image_prompt,file_name_modifier=f"{counter}")
        """result = {'video': 'path_to_video',
                    'image_final': 'path_to_final_image',
                    'animation_prompt': 'path_to_animation_prompt.txt',
                    'source_image_prompt': 'path_to_source_image_prompt.txt',
                    'source_image_review': 'path_to_source_image_review.txt',
                    'image_start': 'path_to_input_image'}
        """
        all_files.append(result)
        video_clips.append(result['video'])

        #check if we have at least one more clip to generate, else break we are done
        if counter >= clips:
            break

        #read the animation prompt from file
        with open(result['animation_prompt'], 'r') as f:
            animation_prompt = f.read()
        #get the final image path
        end_image = result['image_final']
        #create a new prompt for VLM model
        prompt = f"The image you are looking at is the LAST FRAME of a video that was generated with this prompt: {animation_prompt}. Continue the clip using the image as inspiration for how to continue. First, Review the image. Then create a new prompt for the text-to-video AI to Extend the video and create a longer video clip.  Here is an example prompt: Focus strictly on subject Motion/activity + Camera motion, in that order. You do Not need to describe the subject visually as the animation model takes that from the input image. use brackets for emphasis [ ]. here is an example \nThey are having sex, fucking, thrusts in and out. [penis moves in and out of her vagina], dripping white cum and ejaculation. her ass bouncing, breast jiggle, she maintains eye contact with camera. The hips of the man and woman move in rythem opposite directions. Camera [dolly in] slow\n. Create an appropriate erotic sex movie prompt for the input image.\n\nCreate a NEW prompt that continues the image action/motion + camera view"
        #set next image and prompt if we need it
        animation_prompt = qwen_generate(prompt, images=[end_image])
        start_image = end_image

    # Extract base filename from first video (all share same unique number prefix)
    base_file_name = os.path.splitext(os.path.basename(video_clips[0]))[0].rsplit('_', 1)[0]
    output_dir = os.path.dirname(video_clips[0])
    #merge all the video clips
    full_video_filename = os.path.join(output_dir, f"{base_file_name}_full_video.mp4")
    combine_videos(video_clips, full_video_filename)
    return all_files    


def workflow(seed="student teen", img_count=5, lighting="rays of sunlight", scene="library study room", shoot_folder="library"):
    default_model_type = 'vlm'  # 'vlm' or 'lm' for language model only
    img_count = img_count  # Number of images/shots to generate
    seed = seed#"student teen"  # Example seed word for character model
    lighting = lighting#"rays of sunlight"  # Example lighting description
    scene = scene#"library study room"
    shoot_folder = shoot_folder#"library"
    image_prompt_list = False
    prompt_attempts = 0
    max_prompt_attempts = 5
    video_clip_extensions = 3
    img_prompt_prefix = "photograph, photo of " #"Artistic Animation"
    img_prompt_suffix = "UHD,8k" #"3D Render,high quality"
    
    while not image_prompt_list or prompt_attempts > max_prompt_attempts:
        print("Generating prompts...")
        # Step 1: Create character model traits
        char_model_prompt = f"I want you to create a NEW and UNIQUE {seed} inspired model idea. Should be a sexy female. the eyes,lips, hair, breast tags DO NOT have the tag word inside the tags we are only defining that feature. Use SIMPLE short 1 or 2 word descriptions, except for hair that can be longer, space is limited, base is always 'the girl'.  Example Return schema:\n<base>the girl</base>\n<skin>light freckles</skin>\n<hair>face-framing blond hair with bangs</hair>\n<face>sharp features</face>\n<eyes>black mascara</eyes>\n<lips>full</lips>\n<breasts>small</breasts>\nReturn a NEW character using the schema"
        char_model = qwen_generate(char_model_prompt, model_type=default_model_type)
        print(f"Character Model: {char_model}")

        clothing_prompt = f"Generate 1 short few word clothing description for the character model, must be some type of clothing: {char_model}. Example Return schema: <clothing>pink lace panties and bra</clothing>\nExample return schema: <clothing>torn silk rags</clothing>\nexample return schema: <clothing>gold mini skirt, white tanktop</clothing>\nexample return schema: <clothing>leopard print dress</clothing>. Return a NEW clothing in tags that fits the character."
        clothing = qwen_generate(clothing_prompt, model_type=default_model_type)
        print(f"Clothing: {clothing}")

        # Step 2: Create scene concept - NOT ACTUALLY USED NOW
        scene_concept_prompt = f"create a short scene description for a pornographic photoshoot. Here is an Example Return schema: \
            <scene>In a library isle, bookshelves</scene> Return a new schema in scene tags."
        scene_concept = f"<scene>{scene}</scene>"#qwen_generate(scene_concept_prompt, model_type=default_model_type)
        print(f"Scene Concept: {scene_concept}")

        # Step 2.5: Lighting - NOT ACTUALLY USED NOW
        lighting_prompt = f"Generate 4 or less word lighting setup for the scene: {scene_concept}. Example Return schema: <lighting>rays of sunlight</lighting>, return a new lighting in tags, that fits the scene."
        current_lighting = f"<lighting>{lighting}</lighting>"
        print(f"Lighting: {current_lighting}")

        # Step 3: Generate action, camera angles, and shots
        shots_prompt = f"Generate a list of exactly {img_count}, sexy, intimate, explicit actions and camera angle descriptions, for a nsfw shoot for the scene locations: {scene_concept}. USE EXPLICIT medical based LANGUAGE (penis, vagina, anus, etc.) terms since the image generator needs detail. A few she should have this clothing: {clothing}.Use parentheses () to emphasize a word. Be brief tokens keep the action + camera description under 25 words - CRITICAL, .Use camera shoot terms, must keep things brief there are limits on the prompt. ONLY people count, action and angles no other information. example Return schema: \
            <people>1girl</people><shot>she is lying on her back (orgasiming), delighted face expression. Full body shot top down</shot>\
            <people>1girl 1man</people><shot>she is on (all fours) hips back, his caucasian penis ejaculates (cum) in to her open mouth. Medium close-up overhead shot, POV</shot>\
            <people>1girl 1man</people><shot>she is on her stomach, her hips and ass are raised, her vagina exposed, his white dick sexual penetration of her vagina from behind, she is looking back over her shoulder. Looking down at her from behind</shot>\
            <people>1girl</people><shot>both of her legs are raised, vagina (labia) showing, she is pulling her buttocks cheeks spreading her vagina. Extreme close up, focus on her vagina</shot>\
            <people>1girl 1man</people><shot>she is lying on her side, one leg raised, the man is having sex with her, he penis is inside her vagina. Side view</shot>\
            <people>1girl</people><shot>she is sitting one knee is raised, leaning, (chest forward), wearing pink lace panties and bra, she is looking at viewer. Front view Close-up</shot>\
            <people>1girl 1man</people><shot>she is sitting on heals,with butt next to heals, his legs are spread and his penis is in her mouth (deepthroat) blowjob. Low view</shot>\
            <people>1girl</people><shot>she is sitting in a (provocative) pose, arms back, chest forward. Medium shot front view </shot>\
            <people>1girl 1man</people><shot>she is squating, arms between her legs, chin raised mouth open, his penis is (ejaculating) loads of semen in to her mouth and on her chest. Side profile shot</shot>\
            <people>1girl</people><shot>she is standing, one hand on her hip wearing pink lace panties and bra, she is looking (seductively) at viewer. Full body shot</shot>\
        ...generate {img_count} new shot"
        shots = qwen_generate(shots_prompt,model_type=default_model_type)
        print(f"Shots: {shots}")

        print("="*50)
        print(char_model,"\n",clothing,"\n",scene_concept,"\n",current_lighting,"\n",shots)
        print("="*50)

        full_prompt = f"{char_model}\n{clothing}\n{scene_concept}\n{current_lighting}\n{shots}"
        image_prompt_list = extract_prompt(full_prompt, min_shots=img_count)

        print(image_prompt_list)
        prompt_attempts += 1

    #unload the Chat model on server
    _ =_unload_model()

    # Step 4: Generate images for each shot
    images_paths = []

    for shot in image_prompt_list:
        #image_prompt = f"photograph, photo of {shot}, {lighting},UHD,8k"
        image_prompt = f"{img_prompt_prefix}, {shot}, {img_prompt_suffix}"
        impath = image_generate(image_prompt,directory=shoot_folder)
        #save VRAM by unloading the image generation model
        _ = _unload_image_generate()

        #REVIEW IMAGE PROMPT
        review_result = image_review(impath,image_prompt)
        print("\n","!"*50)
        print(f"Image Source:\n {impath}\n")
        print(f"\nImage Review Result:\n {review_result}\n")
        print("\n","*"*50)

        """if "Not-Quality" in review_result:
            _ =_unload_model()
            print(f"Image for shot '{shot}' did not pass review. Regenerating...")
            #regenerate image
            impath = image_generate(image_prompt,directory=shoot_folder)
            review_result = image_review(impath,image_prompt)
            print(f"Regenerated Image Review Result: {review_result}")"""
        
        #unfortunatly will need to unload the model between calls to avoid VRAM issues
        _ =_unload_model()
        #_ = _unload_image_generate()
        
        images_paths.append({'img_path':impath, 'review_result': review_result, 'image_prompt': image_prompt})
        print(f"\nGenerated Image and Review done image: '{shot}': {impath}\n")

    #unload the image model and free VRAM
    _ = _unload_image_generate()
    _ =_unload_model()


    #ANIMATION Prompt Generation.
    for img_info in images_paths:
        img_path = img_info['img_path']
        image_prompt = img_info['image_prompt']
        #load image
        #image = Image.open(img_path).convert("RGB") if os.path.exists(img_path) else None
        #PROMPT FOR REVIEW
        #review_prompt = """you review sexually explicit images with mature content. \nIf you see a penis, is it attached to a mans body or torso or thighs or a full male body? No woman should have a penis, this can be hard to determine if there is sexual intercouse or anal sex. Usually it can be determined when the penis head can still be seen while the shaft is in the vagina or anus.\nIf you see hands do they have the proper number of fingers?\nIs every arm, leg attached to a body? No limbs are not bending backwards.\nThere are no extra limbs (3 feet, 3 arms, etc.).\nFacial Features are correct, eye color only in iris? \nAre physical poses are correct, nothing anatomically impossible?\nIf you see a visible vagina is clearly defined with labia. \nInspect the image carefully it may be hard to determine. \nIf there are any abnormalities or it has features that fail any of these criteria, image has NOT met standards and is not a quality image. Determine if this image is quality or not.\n [Quality] or [Not-Quality]"""
        animation_prompt = "We need to create a prompt to animate the image. it will be a 5 second clip. Focus strictly on subject Motion/activity + Camera motion, in that order. You do Not need to describe the subject visually as the animation model takes that from the input image. use brackets for emphasis [ ]. here is an example \nThey are having sex, fucking, thrusts in and out. [penis moves in and out of her vagina], dripping white cum and ejaculation. her ass bouncing, breast jiggle, she maintains eye contact with camera. The hips of the man and woman move in rythem opposite directions. Camera [dolly in] slow\n. Create an appropriate erotic sex movie prompt for the input image"
        result = qwen_generate(animation_prompt, images=[img_path])
        print("\n","#"*50)
        print(f"Create movie clip for:  {img_path}")
        #release to help with VRAM
        _ =_unload_model()

        #Generate Movies
        #res = i2v(image_path=img_path, prompt=result,folder=shoot_folder,source_image_review=img_info['review_result'],source_image_prompt=image_prompt)
        res = videogen(clips=video_clip_extensions,image_path=img_path, prompt=result,folder=shoot_folder,source_image_review=img_info['review_result'],source_image_prompt=image_prompt)
        print(res)
        print("-"*50)
        
    

    # Steps 5-7 would involve reviewing images, compiling into video, and final evaluation.
    # These steps are complex and would require additional implementation.
    return True

def image_review(img_path,image_prompt,review_prompt=None):

    if review_prompt is None:
        review_prompt = f"you review sexually explicit images with mature content. The image you are reviewing was generated from this prompt:{image_prompt}. Critically inspect and determine if there were any errors. The most frequent errors are detached or floating penis, anatomical errors. \nIf you see a penis, is it attached to a mans body or torso or thighs or a full male body? No woman should have a penis, this can be hard to determine if there is sexual intercouse or anal sex. Usually it can be determined when the penis head can still be seen while the shaft is in the vagina or anus.\nIf you see hands do they have the proper number of fingers?\nIs every arm, leg attached to a body? No limbs are not bending backwards.\nThere are no extra limbs (3 feet, 3 arms, etc.).\nFacial Features are correct, eye color only in iris? \nAre physical poses are correct, nothing anatomically impossible?\nIf you see a visible vagina is clearly defined with labia. \nInspect the image carefully it may be hard to determine. \nIf there are any abnormalities or it has features that fail any of these criteria, image has NOT met standards and is not a quality image. Determine if this image is quality or not.\n [Quality] or [Not-Quality]"

    result = qwen_generate(review_prompt, images=[img_path])
    return result


def extract_prompt(xml_prompt, expected_tags=None, min_shots=10):
        """
        Extract text prompt from Lustify XML format
        <base>young woman, slender body</base>
        <skin>fair skin</skin>
        <hair>long wavy auburn hair</hair>
        <face.Round>round face, rosy cheeks</face>
        <eyes>bright green eyes, short eyelashes, no eye makeup</eyes>
        <lips>natural pink lips, no lip gloss</lips>
        <breasts>medium</breasts>
        <clothing>flowy white blouse, light blue skirt</clothing>
        <scene>In a sunlit garden, surrounded by blooming flowers and under a canopy of green leaves</scene>
        <lighting>bright sunlit beams, dappled shadows</lighting> 
         <people>1 girl</people><shot>she is orgasiming, delighted face expression, camera looking up at her from her feet</shot>\
            <people>1 girl 1 man</people><shot>she is kneeling, the man's penis cums in to her open mouth, she is looking up at the camera, POV</shot>\
            <people>1 girl 1 man</people><shot>she is naked on all fours, arched back, the man's penis is entering her vagina from behind, she is looking back over her shoulder, camera looking down at her from above</shot>\
            <people>1 girl</people><shot>she is sitting legs slightly spread apart, camera is focused on her vagina</shot>\
            <people>1 girl 1 man</people><shot>she is lying on her back, the man is having sex with her missionary style, she looks at camera, POV from over her</shot>
        """
        # Extract clothing options (multiple)
        clothing_options = re.findall(r'<clothing>(.*?)</clothing>', xml_prompt, re.IGNORECASE | re.DOTALL)
        #temp no clothing
        print("NO CLOTHING PROMPT BEING USED ---------- ")
        clothing_options = []
        clothing_options.append("nude, naked")
        
        # Extract people and shot pairs
        shot_pairs = re.findall(r'<people>(.*?)</people>\s*<shot>(.*?)</shot>', xml_prompt, re.IGNORECASE | re.DOTALL)
        
        if len(shot_pairs) < min_shots:
            print(f"Extraction failed: Found {len(shot_pairs)} shots, expected at least {min_shots}")
            return False

        # Extract other single tags
        tag_values = {}
        # Tags to extract once
        single_tags = ['base', 'skin', 'hair', 'face', 'eyes', 'lips', 'breasts', 'scene', 'lighting']
        
        for tag in single_tags:
            # Match <tag>content</tag> or <tag.something>content</tag>
            # Expecting closing tag to match opening tag name (e.g. </face> for <face.Round>)
            pattern = f"<{tag}(?:\\.[^>]*)?>(.*?)</{tag}>"
            match = re.search(pattern, xml_prompt, re.IGNORECASE | re.DOTALL)
            if match:
                tag_values[tag] = match.group(1).strip()
            else:
                tag_values[tag] = ""

        prompts = []
        
        for people, shot in shot_pairs:
            people = people.strip()
            shot = shot.strip()
            
            # Select random clothing
            clothing = random.choice(clothing_options).strip()
            
            # Build the prompt parts in specific order
            # 'people','base', 'skin', 'hair', 'face', 'eyes', 'lips', 'breasts', 'clothing', 'scene', 'lighting'
            parts = []
            
            if people: parts.append(people)
            
            for key in ['base', 'skin', 'hair', 'face', 'eyes', 'lips', 'breasts']:
                if tag_values.get(key):
                    parts.append(tag_values[key]+" "+key)
            
            #if clothing: parts.append(clothing) #CLOTHING INJECTED IN SCENE DIRECTLY NOW
            
            for key in ['scene', 'lighting']:
                if tag_values.get(key):
                    parts.append(tag_values[key])
            
            # Append shot action at the end
            if shot: parts.append(shot)
            
            full_prompt = ", ".join(parts)
            prompts.append(full_prompt)
            
        return prompts

def qwen_generate(prompt, images=None, unload=False, model_type='vlm'):
    url = "http://localhost:5055/generate"
    payload = {
        "prompt": prompt,
        "images": images or [],
        "unload": unload,
        "model_type": model_type  # or "lm" depending on the desired model
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json().get("response")
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

def image_generate(prompt,directory="image"):
    url = "http://localhost:5055/t2i"
    payload = {
        "prompt": prompt,
        "output_dir": directory
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json().get("image_path")
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

def _unload_model():
    url = "http://localhost:5055/unload_chat"
   
    response = requests.post(url)
    if response.status_code == 200:
        return True
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return False
    
def _unload_image_generate():
    url = "http://localhost:5055/unload_t2i"
   
    response = requests.post(url)
    if response.status_code == 200:
        return True
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return False
    
def _random_scene():
    scenes = [
        "beach at sunset",
        "rooftop penthouse",
        "luxury spa",
        #"strip club",
        "library study room",
        "hotel room with view",
        "private yacht deck",
        "tropical waterfall",
        #"deserted island shore",
        #"modern art gallery",
        "cozy mountain cabin",
        #"urban loft apartment",
        "mideval castle courtyard",
        #"futuristic cityscape",
        "secluded forest glade",
        #"vintage diner",
        #"tavern with fireplace",
        #"sunlit garden",
    ]
    return random.choice(scenes)

def _random_lighting():
    lightings = [
        "natural lighting",
        "soft morning light",
        "dramatic spotlight",
        "candlelight glow",
        #"golden hour sunlight",
        "dim lighting",
        "twilight ambiance",
        "moonlit night",
        "studio lighting setup",
        "sun rays through window",
        "warm indoor lighting",
    ]
    return random.choice(lightings)

def _random_woman():
    women = [
        "young woman",
        "teen girl",
        "adult female",
        #"exotic beauty",
        #"sensual lady",
        #"prostitue nightwalker",
        "stripper dancer",
        "fashion model",
        "roman goddess",
        "fairy tale princess",
        "beach babe",
        "fitness model",
        #"artistic model",
        #"bohemian beauty",
        #"elegant lady",
        #"sultry siren",
        #"mysterious femme fatale",
        #"classic pin-up girl",
        "confident businesswoman",
        "free-spirited traveler",
    ]
    return random.choice(women)

if __name__ == "__main__":
    start_time = time.time()
    scenes_count = 5
    images_per_run = 8

    # Run multiple workflow iterations with different seeds
    last_result = {"subject":f"{_random_woman()}", "lighting":f"{_random_lighting()}", "scene":f"{_random_scene()}","name":"lush_425"}
    second_last_result = {"subject":f"{_random_woman()}", "lighting":f"{_random_lighting()}", "scene":f"{_random_scene()}","name":"juciy_861"}
    for run_iter in range(scenes_count):

        try:
            print("="*60)
            print("="*20)
            print("="*20,f" > > > RUNNING SCENE {run_iter+1}")
            print("="*20)
            print("="*60)

            seed_prompt = f"""You need to come up with pornographic arrousing ideas (beach, hotel, rooftop penthouse, couch, van, etc.) I need a SUBJECT of 3 or 4 words, just them NOT what they are wearing. A LIGHTTING style of 2 or 3 words, and a SCENE of 3 to 5 words, lastly I need a reference name for the idea. Here is an example return schema:\
                    {json.dumps(last_result)}. Here is another example: {json.dumps(second_last_result)}. Now create a new unique creative and sexy idea return the schema."""
            char_model = qwen_generate(seed_prompt, model_type='lm')
            print(f"Shoot Idea: {char_model}")
            try:
                seed_data = json.loads(char_model)
            except json.JSONDecodeError:
                import ast
                seed_data = ast.literal_eval(char_model)

            seed_word = seed_data.get("subject","student teen")
            lighting = seed_data.get("lighting","rays of sunlight")
            scene = seed_data.get("scene","library study room")
            shoot_folder = seed_data.get("name","library")
            print(f"Using seed: {seed_word}, lighting: {lighting}, scene: {scene}, folder: {shoot_folder}")
            _ = workflow(seed=seed_word, lighting=lighting, scene=scene, shoot_folder=shoot_folder, img_count=images_per_run)
            if run_iter % 2 == 0:
                last_result = seed_data
            else:
                second_last_result = seed_data
        except Exception as e:
            print(f"Error generating seed data: {e}")
            continue

     
    #_ = workflow()
    end_time = time.time()
    print(f"Workflow completed in {end_time - start_time:.2f} seconds.")