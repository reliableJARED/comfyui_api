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
from deployed import gstorage

COMFY = ComfyCall()

def i2v(image_path="", prompt="",folder="neon", source_image_review="", source_image_prompt="",file_name_modifier="1",lora="sex",clear_vram=False):
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
        file_name_modifier=file_name_modifier,
        lora=lora,
        clear_vram=clear_vram
    )
    return result

def extract_lora_tag(prompt):
    """
    Extracts the content within <lora>...</lora> tags from the prompt.
    Returns the extracted lora name and the prompt with the tag removed.
    """
    match = re.search(r'<lora>(.*?)</lora>', prompt, re.IGNORECASE)
    if match:
        lora_value = match.group(1).strip()
        clean_prompt = prompt.replace(match.group(0), "").strip()
        return lora_value, clean_prompt
    return None, prompt
    
def videogen(clips=1,image_path="", prompt="",folder="neon",source_image_review="", source_image_prompt="", lora="sex"):
    #this function will generate successive videos from a single image and prompt attempting to create a longer animation
    animation_prompt = prompt
    start_image = image_path
    counter = 0
    all_files = []
    video_clips = []
    lora = lora #default LORA model to use for video generation based on content
    clear_vram = True #default when starting video gen to clear vram between clips
    while counter < clips:
        counter += 1
        print(f"Generating clip {counter} of {clips}...")
        #Check the initial prompt to determine if LORA should be overridden
        extracted_lora, prompt = extract_lora_tag(prompt)
        if extracted_lora != lora and extracted_lora is not None:
            lora = extracted_lora
            print(f"Overriding LORA to '{lora}' based on prompt tag. Also sending request to clear vram because model change.")
            clear_vram = True
            
        #first generate initial video
        result = i2v(image_path=start_image, prompt=prompt, folder=folder,source_image_review=source_image_review, source_image_prompt=source_image_prompt,file_name_modifier=f"{counter}", lora=lora, clear_vram=clear_vram)
        """result = {'video': 'path_to_video',
                    'image_final': 'path_to_final_image',
                    'animation_prompt': 'path_to_animation_prompt.txt',
                    'source_image_prompt': 'path_to_source_image_prompt.txt',
                    'source_image_review': 'path_to_source_image_review.txt',
                    'image_start': 'path_to_input_image'}
        """
        # Check if result is empty or missing required keys
        if not result:
            print(f"ERROR: i2v returned empty result for clip {counter}")
            raise ValueError(f"Video generation failed for clip {counter} - empty result returned")
        if 'video' not in result:
            print(f"ERROR: i2v result missing 'video' key for clip {counter}. Keys present: {list(result.keys())}")
            raise ValueError(f"Video generation failed for clip {counter} - no video in result")
        
        all_files.append(result)
        video_clips.append(result['video'])

        # Upload video and animation prompt to GCS
        bucket_name = "xserver"
        if 'video' in result and os.path.exists(result['video']):
            video_path = result['video']
            video_filename = os.path.basename(video_path)
            gstorage.upload_blob(bucket_name, video_path, f"{folder}/{video_filename}")
            
        if 'animation_prompt' in result and os.path.exists(result['animation_prompt']):
            anim_prompt_path = result['animation_prompt']
            anim_prompt_filename = os.path.basename(anim_prompt_path)
            gstorage.upload_blob(bucket_name, anim_prompt_path, f"{folder}/{anim_prompt_filename}")

        #check if we have at least one more clip to generate, else break we are done
        if counter >= clips:
            break

        #read the animation prompt from file
        with open(result['animation_prompt'], 'r') as f:
            animation_prompt = f.read()

        #get the final image path
        end_image = result['image_final']
        #create a new prompt for VLM model - OPTIMIZED FOR WAN 2.2 I2V MODEL
        last_frame_prompt_modifier = "You need to animate the FINAL clip of the video.  It should end in an orgaism, or she should move in to a position (if she isn't in one already) to receive a facial cumshot.  The image should clearly show the action and camera angle to be animated in the next clip."
        prompt = f"""The image you are looking at is the LAST FRAME of a video generated with this prompt: {animation_prompt}

Your task: Create a NEW continuation prompt for the Wan 2.2 I2V video model. This model requires NATURAL LANGUAGE descriptions, NOT tag-based prompts.

{last_frame_prompt_modifier if counter == clips else ""}

CRITICAL RULES FOR WAN 2.2:
1. Use descriptive sentences, NOT tags or brackets like [word] or (word) - those do nothing in Wan
2. Describe motion explicitly with temporal language: "slowly", "rhythmically", "then", "continuously", "repeating"
3. Describe physical movements in detail: body parts moving, expressions changing, physical interactions
4. Camera terms that work: "close-up shot", "POV perspective", "camera slowly dollies in", "low angle view", "the camera follows"
5. Keep prompts 2-4 sentences, focusing on MOTION and CAMERA only
6. The model takes visual details from the input image - describe only ACTION and CAMERA
7. If you are creating a 'facial or 'cum on face' scene, use those terms explicitly in the prompt. If you are creating a blowjob scene, use 'blowjob', 'fellatio', or 'oral' terms explicitly.

MOTION VOCABULARY:
- Thrusting rhythmically, hips moving back and forth, bouncing motion
- Head bobbing up and down, taking deep then pulling back
- Body trembling, muscles tensing, breathing heavily
- Arching back, gripping tightly, pressing closer

CAMERA VOCABULARY:
- Close-up shot focusing on the action
- POV perspective from his viewpoint
- Camera slowly dollies in closer
- Low angle looking up, overhead shot looking down
- Side profile capturing the motion

PROGRESSION: Your prompts should continue the progression of the scene, and head toward climax (facial). The source image will determine if it's appropriate to attempt a facial scene.

EXAMPLE PROMPTS:
"She rhythmically moves her hips, riding him with increasing intensity. Her body bounces up and down as she grips his chest for support. The camera slowly dollies in to capture the intimate motion from a low angle<lora>sex</lora>."

"He thrusts deeply in a steady rhythm, her body responding to each movement. She arches her back and moans as the intensity builds toward climax. POV perspective capturing her expressions of pleasure<lora>sex</lora>."

{"Her head bobs up and down rhythmically, taking him deep into her mouth then pulling back slowly. She maintains eye contact while her hand strokes the base. Close-up shot from his POV.<lora>facial</lora>" if counter == clips else "she puts the penis in her mouth and slowly moves her lips down the penis shaft to his stomach. maintains eye contact with viewer. her lips touch his torso. Then starts bobbing her head sliding penis shaft in and out of her mouth. she is giving him a deepthroat blowjob. Camera close-up POV<lora>blowjob</lora>"}

{"cum shot facial. The penis ejaculates semen on her face, in her mouth and on her chest. POV.<lora>facial</lora>" if counter == clips else "sex, The mans penis thrusts all the way inside the vagina, then pulls back out of the vagina, repeatedly, full insertion. Her thighs and breasts jiggle on impacts.<lora>sex</lora>"}


Now analyze the image and create a NEW 2-4 sentence prompt describing the MOTION to animate and CAMERA angle. The image should guide you. if it's already a blowjob stick with that style, if it's sex or any penis entering a vagina or anus stick with that. Natural language only, no brackets or tags.

Finally, let the model know which LORA to use based on the content by including one of these three options: {"<lora>blowjob</lora> or <lora>sex</lora>. Sex is for all secene types not blowjob or facial." if counter != clips else "<lora>facial</lora>, or <lora>sex</lora> . Sex is for any scene not ending in a facial."}"""
        #set next image and prompt if we need it
        animation_prompt = qwen_generate(prompt, images=[end_image])
        start_image = end_image
        prompt = animation_prompt

        clear_vram = False #only clear vram on first clip or if model change

    # Extract base filename from first video (all share same unique number prefix)
    base_file_name = os.path.splitext(os.path.basename(video_clips[0]))[0].rsplit('_', 1)[0]
    output_dir = os.path.dirname(video_clips[0])
    #merge all the video clips
    full_video_filename = os.path.join(output_dir, f"{base_file_name}_full_video.mp4")
    combine_videos(video_clips, full_video_filename)
    
    # Upload full video to GCS
    bucket_name = "xserver"
    if os.path.exists(full_video_filename):
        full_video_basename = os.path.basename(full_video_filename)
        gstorage.upload_blob(bucket_name, full_video_filename, f"{folder}/{full_video_basename}")

    return all_files    

def workflow_anime(seed="student teen", img_count=5, lighting="rays of sunlight", scene="library study room", shoot_folder="library"):
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
    img_prompt_prefix = "photograph, 8k, " #"Artistic Animation"
    img_prompt_suffix = " 8k" #"3D Render,high quality"
    #Prompt Examples
    #1766358994
    a = "Realistic anime, illustrious comic, meadow, real skin irish freckles smiling, at viewer, long straight black hair, white-eyes green iris, choker, black lace lingerie top, exposed stomach, black knee high nylons, pleated green mini skirt, high heels. leaning forward clevage, hand on knees, hips back. soft lighting, front view"
    #1766359527
    b = "Realistic anime, illustrious comic, grass meadow, leaning forward clevage, standing, hand on knees, hips back, real skin irish freckles, smiling at viewer, long straight black hair, white-eyes green iris, choker, black lace lingerie top, exposed stomach, black knee high nylons, pleated green mini skirt, stilettos. soft lighting, front view"
    #1766360114
    c = "Realistic anime, illustrious comic, grass meadow, wide stance, leaning forward clevage, real skin irish freckles, laughing, long straight black hair, white-eyes green iris, choker, black bikini top, exposed stomach, black knee high nylons, pleated green mini skirt, stilettos. soft lighting, front view"
    #1766360295
    d = "Realistic anime, illustrious comic, grass meadow, hands on hips, wide stance, leaning forward clevage, real skin irish freckles, laughing, long straight black hair, white-eyes green iris, choker, black lingerie top, exposed stomach, black knee high nylons, pleated green mini skirt, stilettos. soft lighting, close-up"
    #1766361485
    e = "Realistic anime, illustrious comic, grass meadow, squatting, knees apart, clevage, real skin irish freckles, intense, long straight black hair, white-eyes green iris, choker, (black lace lingerie) top, exposed stomach, black knee high nylons, pleated green mini skirt, stilettos. sun ray lighting"
    #1766364692
    f = "Realistic anime, illustrious comic, grass meadow, laying on stomach, elbows bent, chin in hands, legs back, clevage, real skin irish freckles, intense, long straight black hair, white-eyes green iris, choker, (black lace lingerie) top, knees bent legs in air behind her, black knee high nylons, pleated green mini skirt, stilettos. sunset ray lighting"
    #1766364847
    g = "Realistic anime, illustrious comic, grass meadow, sitting on heels, arms stretched on knees, clevage, real skin irish freckles, intense, long straight black hair, white-eyes green iris, choker, (black lace lingerie) top, black knee high nylons, pleated green mini skirt, stilettos. pollen in wind, sunset ray lighting"

    return [a,b,c,d,e,f,g]

def missionary_workflow():
    return
def fellatio_workflow():
    return
def rear_doggystyle_workflow():
    return
def front_doggystyle_workflow():
    return

def workflow_v2(shoot_folder="dec24_workflow_test6"):
    # Create shoot folder on GCS if it doesn't exist
    bucket_name = "xserver"
    gstorage.check_or_create_folder(bucket_name, shoot_folder)
    characters = 3
    photos = 10 #how many from EACH model, will generate 2x this in images
    video_clip_extensions = 3 #how many successive frame to frame clips to try and make

    #PROMPT INSTRUCTIONS FOR AI IMAGE GEN PROMPTS
    #This creates different people, ai does not create the same person
    #FANTASY ELF, DRUID or MAGE
    prompt = f"""
    You create pornographic NSFW content in the FANTASY genre that depicts ELF, DRUID or MAGE woman.  Create a women and develop solo and heterosexual prompts. Use variety.  mostly nude, some clothed or bits of clothing.

    1. **Underscores**: Always use underscored Danbooru tags for: sexual_position, sex_acts, perspective,  mixing with natural language for the rest of the prompt
    2. **Front-load important concepts**:only describe the woman, man will auto generate simply using '1boy' as needed 
    PROMPT STRUCTURE:\nFacial features, accessories like glasses, choker, hair color and style, iris color, accents like mascara, long lashes, eye shadow.\nClothing or nude, naked\nBody type fit, athletic, curvy, thin\ncharacter count, either 1girl, or 1girl 1boy,\nSexual Postion, Act and Perspective tags\nLocation in 1 or 2 words hotel room, hot tub, bed room, forest, cabin, etc.\nLighting 
    3. **Photography tags have major visual impact**: Camera types and lighting dramatically affect the output
    4. **Use commas to separate concepts**
    5. **Parentheses/weight syntax doesn't work** in raw diffusers - they're treated as literal characters
    6. **Quality matters less than content**: Focus on describing what you want rather than quality tags
    7. **Experiment with hybrid approaches**: Mix tags and natural language for best results

    ### Body Features & Modifiers
    nude, naked, topless, bottomless
    breasts, small_breasts, large_breasts
    nipples, pussy, penis, erection
    spread_legs, legs_apart
    straddling
    arched_back

    ### Clothing States
    lingerie, underwear, panties, bra
    torn_clothes, clothes_pull
    partially_undressed
    stockings, thigh_highs, pantyhose
    sheer_legwear

    ### Intimacy & Expression
    sex, hetero
    kissing, french_kiss
    looking_at_viewer, eye_contact
    seductive_smile, open_mouth
    sweat, saliva
    cum_on_face, cum
    
    ### Character Count
    1girl, 1boy 
    1girl, solo 

    ### Common Sexual Positions
    missionary, missionary_position
    sex_from_behind, doggystyle
    cowgirl_position, girl_on_top, woman_on_top
    reverse_cowgirl
    standing_sex
    spooning
    69_position

    ### Sexual Acts
    fellatio, oral, blowjob, deepthroat
    vaginal, penetration, sex
    handjob
    titjob, paizuri
    anal

    ### Perspectives & Focus
    pov, pov_crotch
    from_behind, from_below, from_above
    close-up, wide_shot
    male_focus, female_focus


    ### Lighting Types
    cinematic lighting
    soft lighting
    warm golden hour lighting
    dramatic lighting
    low key lighting
    neon lighting
    bright flash photography
    radiant god rays

    First - Come up with a femal description - it must be consistent throughout.  
    Then come up with a combination of {photos} sexual positions, acts and perspectives. 
    Return XML schema for the image prompts, put each string prompt inside <scene></scene> tags. ONLY use scene tag to contain prompt strings, no other tags.
    """
    
    #PROMPT INSTRUCTIONS FOR AI IMAGE GEN PROMPTS
    #This creates different people, ai does not create the same person
    prompt = f"""
    You create pornographic NSFW content.  First Come up with a look for our model. She will be described THE SAME WAY each time, except for clothing since she could be nude.
    
    DESCRIPTION STRUCTURE for the WOMAN ONLY:\n
    ###Facial Structure
    - round face, sharp features, high cheekbones

    ### Hair Style
    - long, short, ponytail, bixi-cut, wavy, straight, curly, bangs, face-framing, wafting, flowing, braided
    
    ### Hair Color
    - chestnut, black, blond, platinum blonde, red, pink, silver
    
    ### Iris Color
    - blue, green, brown, hazel, grey

    ### Eye Accents
    - long lashes, eye shadow, eyeliner, mascara, dark eyeliner, dark mascara
    
    ### Accessories
    -  glasses, choker, hair color and style, iris color, accents like mascara

    ### Clothing (except when nude)
    - lingerie, underwear, panties, bra, torn_clothes, clothes_pull, partially_undressed, stockings, thigh_highs, pantyhose, sheer_legwear

    ### Body type
    - fit, athletic, curvy, thin

    Then develop solo and heterosexual prompts. Use variety of sexual position, sexual act and perspectives.  mostly nude, some clothed or bits of clothing.

    combine natural language appearance description with danbooru tags for sexual positions, acts and perspectives.    
    
    1. **Underscores**: Always use underscored Danbooru tags for: sexual_position, sex_acts, perspective,  mixing with natural language for the rest of the prompt
    2. **Front-load important concepts**:only describe the woman, man will auto generate simply using '1boy' as needed 
    PROMPT STRUCTURE:\nFacial features, accessories like glasses, choker, hair color and style, iris color, accents like mascara, long lashes, eye shadow.\nClothing or nude, naked\nBody type fit, athletic, curvy, thin\ncharacter count, either 1girl, or 1girl 1boy,\nSexual Postion, Act and Perspective tags\nLocation in 1 or 2 words hotel room, hot tub, bed room, forest, cabin, etc.\nLighting 
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

    Follow the PROMPT STRUCTURE above - use your female description - keep her consistent throughout.  

    Then come up with a combination of sexual positions, acts and perspectives. Create {photos} prompts for our female

    Return XML schema for the image prompts, put each string prompt inside <scene></scene> tags. ONLY use scene tag to contain prompt strings, no other tags.

    YOU MUST describe the woman exactly the same (except clothing or nude) only the position, act and perspective should change.  
    Keep prompting short there is a hard limit of about 35 words (77 tokens).
    """
    
    
    #Qwen 2.5 is Not nearly as good at image prompt generation as Qwen 3 VLM. Switch to Qwen 3 VLM for image prompt generation
    
    scenes = []
    for attempt in range(3):
        tries = 0
        while tries < 3:
            tries += 1
            q3_results = qwen_generate(prompt, model_type='vlm')
            q3_results = extract_scenes(q3_results)
            if q3_results:
                scenes += q3_results
                break
            print(f"Attempt {tries} failed, retrying...")
    print(f"Q3 Generated: {len(q3_results)} scenes, was asked for {photos}")

    _unload_model()
    time.sleep(10)
    
    # Generate ALL images first
    generated_images = []
    for shot in scenes:
        image_prompt = f"photograph, 8k photo, {shot}"
        print(f"\nGenerating image for shot: {shot}")
        impath = image_generate(image_prompt, directory=shoot_folder)
        generated_images.append({'img_path': impath, 'image_prompt': image_prompt})
        print(f"Generated: {impath}")
        print("PUSH to image and prompt GCS storage...   ")
        
        # Upload image
        bucket_name = "xserver"
        filename = os.path.basename(impath)
        gstorage.upload_blob(bucket_name, impath, f"{shoot_folder}/{filename}")
        
        # Upload prompt
        prompt_filename = f"{os.path.splitext(filename)[0]}.txt"
        gstorage.upload_blob_from_memory(bucket_name, image_prompt, f"{shoot_folder}/{prompt_filename}")

    
    # Unload image generation model after all images are done
    print(f"\nAll {len(generated_images)} images generated. Starting reviews...")

    #unload the image model and free VRAM
    _ = _unload_image_generate()
    
    #NO IMAGE REVIEW for this workflow version
    #Hold Generated images and their reviews and prompts file path to send to video gen
    images_paths = []

    print(f"\nSKIP Reviewing generated images...DISABLED FOR SPEED")
    for img_info in generated_images:
        impath = img_info['img_path']
        image_prompt = img_info['image_prompt']
        
        #review_result = image_review(impath, image_prompt)
        review_result = "[Quality]"  #skip review for speed
        print(f"\nSKIP Reviewing generated images...DISABLED FOR SPEED")
        
        print("\n", "!"*50)
        print(f"Image Source:\n {impath}\n")
        print(f"\nImage Review Result:\n {review_result}\n")
        print("\n", "*"*50)
        
        
        images_paths.append({'img_path': impath, 'review_result': review_result, 'image_prompt': image_prompt})
        print(f"\nReview complete for: {impath}\n")

    #ANIMATION Prompt Generation.
    for img_info in images_paths:
        img_path = img_info['img_path']
        image_prompt = img_info['image_prompt']
        img_info['review_result'] = review_result

        # INITIAL ANIMATION PROMPT - OPTIMIZED FOR WAN 2.2 I2V MODEL
        animation_prompt = f"""Analyze this image and created with this prompt: {image_prompt} 
        
        Create a prompt to animate the image as a 5 second video clip using the Wan 2.2 I2V model.  Understand what the image is depicting and create a NATURAL LANGUAGE description of how to animate the erotic sexual motion and camera angle to animate.

CRITICAL RULES FOR WAN 2.2:
1. Use NATURAL LANGUAGE descriptions only - NO brackets [word] or parentheses (word) - they have no effect in Wan
2. Describe motion with temporal words: "slowly", "rhythmically", "continuously", "then", "repeating the motion"
3. Focus on MOTION and CAMERA only - the model gets visual details from the input image
4. Keep it to 2-4 descriptive sentences

MOTION VOCABULARY:
- Thrusting rhythmically, hips rocking back and forth, bouncing up and down
- Head bobbing motion, taking deep then pulling back slowly
- Body trembling, muscles tensing, arching back, gripping tightly
- Breasts bouncing, ass jiggling, body swaying, squeezing

CAMERA VOCABULARY:
- Close-up shot, medium shot, full body view
- POV perspective, low angle view, overhead shot
- Camera slowly dollies in, camera follows the motion
- Side profile, front view, from behind

EXAMPLE PROMPTS:
"She bounces up and down rhythmically, riding with increasing intensity. Her breasts jiggle with each movement as she grips his chest. Close-up shot slowly dollying in from a low angle."

"He thrusts deeply in a steady rhythm while she moans with pleasure. Her body rocks with each motion as intensity builds. POV perspective capturing her face and bouncing breasts."

"Her head moves up and down in a steady rhythm, taking him deep then slowly pulling back. She maintains eye contact with a pleasured expression. Close-up POV shot."

Create a 2-4 sentence animation prompt for this image. Natural language only, describe the erotic sexual motion and camera angle.

Finally, there are two different animation models based on the content.  Add a tag at the end of your prompt for either: <lora>blowjob</lora> or <lora>sex</lora>. Sex is for ALL secene types that are not a form blowjob or situations where the woman's face is near a penis and may become a blowjob."""
        result = qwen_generate(animation_prompt, images=[img_path])
        print("\n","#"*50)
        print(f"Create movie clip for:  {img_path}")
        
        #Generate Movies
        res = videogen(clips=video_clip_extensions,image_path=img_path, prompt=result,folder=shoot_folder,source_image_review=img_info['review_result'],source_image_prompt=image_prompt)
        print(res)
        print("-"*50)
    
    #release to help with VRAM
    _ =_unload_model()

    return True

 
def workflow(seed="student teen", img_count=5, lighting="rays of sunlight", scene="library study room", shoot_folder="library"):
    # Create shoot folder on GCS if it doesn't exist
    bucket_name = "xserver"
    gstorage.check_or_create_folder(bucket_name, shoot_folder)

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
    img_prompt_prefix = "photograph, 8k, " #"Artistic Animation"
    img_prompt_suffix = " 8k" #"3D Render,high quality"
    hard_code_hack = ["<base>young woman</base>\n<skin>white</skin>\n<hair>short face-framing pink hair with bangs</hair>\n<face>thin eyebrows</face>\n<eyes>long eyelashes</eyes>\n<lips> </lips>\n<breasts>small</breasts>",
                      "<base>young woman</base>\n<skin>tan</skin>\n<hair>pony tail blond hair</hair>\n<face>sharp facial features</face>\n<eyes>sexy</eyes>\n<lips>full</lips>\n<breasts> </breasts>",
                      "<base>an Irish woman</base>\n<skin>freckles, tan</skin>\n<hair>short wavy black hair with bangs</hair>\n<face>hard facial features</face>\n<eyes>long eyelashes,light blue</eyes>\n<lips> </lips>\n<breasts>small</breasts>",
                      "<base>a sexy woman</base>\n<skin>white</skin>\n<hair>long flowing silver hair</hair>\n<face>hard facial features</face>\n<eyes>dark eye liner and mascara, blue</eyes>\n<lips>puckered</lips>\n<breasts>small</breasts>"]
    while not image_prompt_list or prompt_attempts > max_prompt_attempts:
        print("NOT Generating prompts...USING HARD CODE HACK")
        # Step 1: Create character model traits
        #char_model_prompt = f"I want you to create a NEW and UNIQUE {seed} inspired model idea, fun new hair stles and color. I WANT A thin or fit woman white lady. There are key attributes you need to assign. Base examples (the athletic woman, the girl, the skinny teen, the petite girl, the fit woman, ...), Skin examples: (white, fair, ...). Hair examples: (brown ponytails, bixi-cut face framing, short straight black with bangs, wafting chestnut, straight blond, platinum blonde, etc). Face examples: (high cheekbones, round features, angled features, dimples, etc.),  Breast examples (32A, 36D, 32B, small, perky, medium, large, etc). You are defining a sexy female. the eyes,lips, hair, breast tags ARE NOT stated inside the tags defining that feature. Use SIMPLE short 1 or 2 word descriptions, except for hair that can be longer, space is limited we are using SDXL which has 77 token limit.  Example Return schema:\n<base>the woman</base>\n<skin>tan skin</skin>\n<hair>pony tail blond hair</hair>\n<face>thin eyebrows</face>\n<eyes>sexy</eyes>\n<lips>full</lips>\n<breasts>small</breasts>\nReturn a NEW character using the schema"
        #char_model = qwen_generate(char_model_prompt, model_type=default_model_type)
        char_model = hard_code_hack[prompt_attempts % len(hard_code_hack)]
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
        shots_prompt = f"Generate a list of exactly {img_count}, sexy, intimate, explicit actions and camera angle descriptions, for a nsfw shoot for the scene locations: {scene_concept}. USE EXPLICIT medical based LANGUAGE (penis, vagina, anus, etc.) terms since the image generator needs detail. A few she should have this clothing: {clothing}. Be brief tokens keep the action + camera description under 25 words - CRITICAL, .Use camera shoot terms, must keep things brief there are limits on the prompt. ONLY people count, action and angles no other information. example Return schema: \
            <people>1girl</people><shot>she is lying on her back (orgasiming), delighted face expression. Full body shot top down</shot>\
            <people>1girl 1man</people><shot>she is on (all fours) hips back, his caucasian penis ejaculates (cum) in to her open mouth. Medium close-up overhead shot, POV</shot>\
            <people>1girl 1man</people><shot>she is on her stomach, her hips and ass are raised, her vagina exposed, caucasian dick sexual penetration of her vagina from behind, she is looking back over her shoulder. Looking down at her from behind</shot>\
            <people>1girl</people><shot>both of her legs are raised, vagina (labia) showing, she is pulling her buttocks cheeks spreading her vagina. Extreme close up, focus on her vagina</shot>\
            <people>1girl</people><shot>she is sitting in a (provocative) pose, arms back, chest forward. Medium shot front view </shot>\
            <people>1girl 1man</people><shot>she is squating, arms between her legs, chin raised mouth open, his caucasian penis is (ejaculating) loads of semen in to her mouth and on her chest. Side profile shot</shot>\
            <people>1girl</people><shot>she is standing, one hand on her hip wearing pink lace panties and bra, she is looking (seductively) at viewer. Full body shot</shot>\
            <people>1girl 1man</people><shot>she is lying on her side, one leg raised, the man is having sex with her, he penis is inside her vagina. Side view</shot>\
            <people>1girl</people><shot>she is sitting one knee is raised, leaning, (chest forward), wearing pink lace panties and bra, she is looking at viewer. Front view Close-up</shot>\
            <people>1girl 1man</people><shot>she is sitting on heals,with butt next to heals, his legs are spread and his caucasian penis is in her mouth (deepthroat) blowjob. Low view</shot>\
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

    # Step 4a: Generate ALL images first
    generated_images = []
    for shot in image_prompt_list:
        image_prompt = f"{img_prompt_prefix}, {shot}, {img_prompt_suffix}"
        print(f"\nGenerating image for shot: {shot}")
        impath = image_generate(image_prompt, directory=shoot_folder)
        generated_images.append({'img_path': impath, 'image_prompt': image_prompt})
        print(f"Generated: {impath}")
        
        # Upload image
        bucket_name = "xserver"
        filename = os.path.basename(impath)
        gstorage.upload_blob(bucket_name, impath, f"{shoot_folder}/{filename}")
        
        # Upload prompt
        prompt_filename = f"{os.path.splitext(filename)[0]}.txt"
        gstorage.upload_blob_from_memory(bucket_name, image_prompt, f"{shoot_folder}/{prompt_filename}")
    
    # Unload image generation model after all images are done
    _ = _unload_image_generate()
    print(f"\nAll {len(generated_images)} images generated. Starting reviews...")

    # Step 4b: Review ALL images
    print(f"\nSKIP Reviewing generated images...DISABLED FOR SPEED")
    for img_info in generated_images:
        impath = img_info['img_path']
        image_prompt = img_info['image_prompt']
        
        #review_result = image_review(impath, image_prompt)
        review_result = "[Quality]"  #skip review for speed
        print(f"\nSKIP Reviewing generated images...DISABLED FOR SPEED")
        
        print("\n", "!"*50)
        print(f"Image Source:\n {impath}\n")
        print(f"\nImage Review Result:\n {review_result}\n")
        print("\n", "*"*50)
        
        
        images_paths.append({'img_path': impath, 'review_result': review_result, 'image_prompt': image_prompt})
        print(f"\nReview complete for: {impath}\n")

    """
    # Step 4c: Regenerate failed images (commented out for now)
    for i, img_info in enumerate(images_paths):
        if "Not-Quality" in img_info['review_result']:
            print(f"Image '{img_info['img_path']}' did not pass review. Regenerating...")
            impath = image_generate(img_info['image_prompt'], directory=shoot_folder)
            _ = _unload_image_generate()
            review_result = image_review(impath, img_info['image_prompt'])
            _ = _unload_model()
            images_paths[i] = {'img_path': impath, 'review_result': review_result, 'image_prompt': img_info['image_prompt']}
            print(f"Regenerated Image Review Result: {review_result}")
    """

    #unload the image model and free VRAM
    _ = _unload_image_generate()
    


    #ANIMATION Prompt Generation.
    for img_info in images_paths:
        img_path = img_info['img_path']
        image_prompt = img_info['image_prompt']
        #load image
        #image = Image.open(img_path).convert("RGB") if os.path.exists(img_path) else None
        #PROMPT FOR REVIEW
        #review_prompt = """you review sexually explicit images with mature content. \nIf you see a penis, is it attached to a mans body or torso or thighs or a full male body? No woman should have a penis, this can be hard to determine if there is sexual intercouse or anal sex. Usually it can be determined when the penis head can still be seen while the shaft is in the vagina or anus.\nIf you see hands do they have the proper number of fingers?\nIs every arm, leg attached to a body? No limbs are not bending backwards.\nThere are no extra limbs (3 feet, 3 arms, etc.).\nFacial Features are correct, eye color only in iris? \nAre physical poses are correct, nothing anatomically impossible?\nIf you see a visible vagina is clearly defined with labia. \nInspect the image carefully it may be hard to determine. \nIf there are any abnormalities or it has features that fail any of these criteria, image has NOT met standards and is not a quality image. Determine if this image is quality or not.\n [Quality] or [Not-Quality]"""
        
        # INITIAL ANIMATION PROMPT - OPTIMIZED FOR WAN 2.2 I2V MODEL
        animation_prompt = """Analyze this image and create a prompt to animate it as a 5 second video clip using the Wan 2.2 I2V model.

CRITICAL RULES FOR WAN 2.2:
1. Use NATURAL LANGUAGE descriptions only - NO brackets [word] or parentheses (word) - they have no effect in Wan
2. Describe motion with temporal words: "slowly", "rhythmically", "continuously", "then", "repeating the motion"
3. Focus on MOTION and CAMERA only - the model gets visual details from the input image
4. Keep it 2-4 descriptive sentences

MOTION VOCABULARY:
- Thrusting rhythmically, hips rocking back and forth, bouncing up and down
- Head bobbing motion, taking deep then pulling back slowly
- Body trembling, muscles tensing, arching back, gripping tightly
- Breasts bouncing, ass jiggling, body swaying

CAMERA VOCABULARY:
- Close-up shot, medium shot, full body view
- POV perspective, low angle view, overhead shot
- Camera slowly dollies in, camera follows the motion
- Side profile, front view, from behind

EXAMPLE PROMPTS:
"She bounces up and down rhythmically, riding with increasing intensity. Her breasts jiggle with each movement as she grips his chest. Close-up shot slowly dollying in from a low angle."

"He thrusts deeply in a steady rhythm while she moans with pleasure. Her body rocks with each motion as intensity builds. POV perspective capturing her face and bouncing breasts."

"Her head moves up and down in a steady rhythm, taking him deep then slowly pulling back. She maintains eye contact with a pleasured expression. Close-up POV shot."

Create a 2-4 sentence animation prompt for this image. Natural language only, describe the erotic motion and camera angle."""
        result = qwen_generate(animation_prompt, images=[img_path])
        print("\n","#"*50)
        print(f"Create movie clip for:  {img_path}")
        

        #Generate Movies
        #res = i2v(image_path=img_path, prompt=result,folder=shoot_folder,source_image_review=img_info['review_result'],source_image_prompt=image_prompt)
        res = videogen(clips=video_clip_extensions,image_path=img_path, prompt=result,folder=shoot_folder,source_image_review=img_info['review_result'],source_image_prompt=image_prompt)
        print(res)
        print("-"*50)
    
    #release to help with VRAM
    _ =_unload_model()
        
    
    return True

def image_review(img_path,image_prompt,review_prompt=None):

    if review_prompt is None:
        review_prompt = f"you review sexually explicit images with mature content. The image you are reviewing was generated from this prompt:{image_prompt}. Critically inspect and determine if there were any errors. The most frequent errors are detached or floating penis, anatomical errors. \nIf you see a penis, is it attached to a mans body or torso or thighs or a full male body? No woman should have a penis, this can be hard to determine if there is sexual intercouse or anal sex. Usually it can be determined when the penis head can still be seen while the shaft is in the vagina or anus.\nIf you see hands do they have the proper number of fingers?\nIs every arm, leg attached to a body? No limbs are not bending backwards.\nThere are no extra limbs (3 feet, 3 arms, etc.).\nFacial Features are correct, eye color only in iris? \nAre physical poses are correct, nothing anatomically impossible?\nIf you see a visible vagina is clearly defined with labia. \nInspect the image carefully it may be hard to determine. \nIf there are any abnormalities or it has features that fail any of these criteria, image has NOT met standards and is not a quality image. Determine if this image is quality or not.\n [Quality] or [Not-Quality]"

    result = qwen_generate(review_prompt, images=[img_path])
    return result


def extract_scenes(xml_text):
    """
    Extract content from <scene></scene> tags and return as a list of strings.
    """
    scenes = re.findall(r'<scene>(.*?)</scene>', xml_text, re.IGNORECASE | re.DOTALL)
    return [scene.strip() for scene in scenes]

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
                    if key == 'base' or key == 'face':
                        parts.append(tag_values[key])
                    else:
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
        #"beach at sunset",
        #"rooftop penthouse",
        "luxury spa",
        #"strip club",
        "library",
        "hotel room with view",
        "private yacht",
        "forest",
        #"deserted island shore",
        #"modern art gallery",
        "mountain cabin",
        #"urban loft apartment",
        "mideval castle courtyard",
        #"futuristic cityscape",
        "secluded glade",
        #"vintage diner",
        #"tavern with fireplace",
        #"sunlit garden",
    ]
    return random.choice(scenes)

def _random_lighting():
    lightings = [
        "natural lighting",
        "soft morning light",
        #"dramatic spotlight",
        "candlelight glow",
        #"golden hour sunlight",
        #"dim lighting",
        #"twilight ambiance",
        #"moonlit night",
        #"studio lighting setup",
        "sun rays through window",
        "warm indoor lighting",
    ]
    return random.choice(lightings)

def _random_woman():
    women = [
        "young woman",
        "teen girl",
        #"adult female",
        #"exotic beauty",
        #"sensual lady",
        #"prostitue nightwalker",
        #"stripper dancer",
        "fashion model",
        #"roman goddess",
        #"fairy tale princess",
        #"beach babe",
        "fitness model",
        #"artistic model",
        #"bohemian beauty",
        #"elegant lady",
        #"sultry siren",
        #"mysterious femme fatale",
        #"classic pin-up girl",
        #"confident businesswoman",
        #"free-spirited traveler",
    ]
    return random.choice(women)

if __name__ == "__main__":
    start_time = time.time()
    #7.5 Hour RUN TIME LIMIT TEST
    #scenes_count = 5
    #images_per_run = 8
    _ = workflow_v2()
    end_time = time.time()
    print(f"Workflow completed in {end_time - start_time:.2f} seconds.")
    
    
    scenes_count = 0
    images_per_run = 0

    # Run multiple workflow iterations with different seeds
    last_result = {"subject":f"{_random_woman()}", "lighting":f"{_random_lighting()}", "scene":f"{_random_scene()}","name":"jade_027"}
    second_last_result = {"subject":f"{_random_woman()}", "lighting":f"{_random_lighting()}", "scene":f"{_random_scene()}","name":"monica_516"}
    for run_iter in range(scenes_count):

        try:
            print("="*60)
            print("="*20)
            print("="*20,f" > > > RUNNING SCENE {run_iter+1}")
            print("="*20)
            print("="*60)

            seed_prompt = f"""You need to come up with pornographic nsfw arrousing ideas (hotel room, bedroom, couch, kitchen, etc.) I need a SUBJECT of 3 or 4 words, just them NOT what they are wearing. A LIGHTTING style of 2 or 3 words, and a SCENE of 3 to 5 words, lastly I need a name for the idea. Here is an example return schema:\
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

            scene_time = time.time()
            print("="*60,"\n")
            print(f"Scene {run_iter+1} completed in {scene_time - start_time:.2f} seconds.")
            print("="*60,"\n")

        except Exception as e:
            print(f"Error generating seed data: {e}")
            continue

     
    #_ = workflow()
    end_time = time.time()
    print(f"Workflow completed in {end_time - start_time:.2f} seconds.")