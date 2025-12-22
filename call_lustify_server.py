#!/usr/bin/env python3
"""
Simple client to call the LUSTIFY image generation server
"""

import requests

SERVER_URL = "http://localhost:8052"


def generate_image(prompt, output_file="output.png", **kwargs):
    """Send a generation request to the server"""
    payload = {
        "prompt": prompt,
        "output_file": output_file,
        **kwargs
    }
    
    response = requests.post(f"{SERVER_URL}/generate", json=payload)
    return response.json()


if __name__ == "__main__":
    import time
    ts = int(time.time())
    #Anime girl, black lace lingerie, green pleated skirt, thigh-high stockings, long black hair, green eyes, elegant indoor pose, soft lighting, fantasy style.
    #Anime girl, black lace bustier, green pleated skirt, thigh-high stockings, green eyes, long black hair, elegant indoor pose, natural lighting, soft focus, detailed textures.
    # Anime girl, black hair with clover, green eyes, wears green lace lingerie, black thigh-highs, grassy field, soft lighting, detailed skin, fantasy style, elegant pose 
    #Anime girl, black hair, green eyes, black lace top, green pleated skirt, thigh-highs, choker, grassy hills, sunset, soft lighting, detailed, vibrant colors, elegant pose
    
    #prompt = "Realistic anime, illustrious comic, grass meadow, laying on side, ((clevage)), real skin (irish) freckles, kiss, long straight black hair, white-eyes green iris, choker, (black lace lingerie) top, knees bent legs in air behind her, black knee high nylons, pleated green mini skirt, stilettos. sunlight rays, side view"
    
    # SDXL/Lustify prompt - proper Danbooru tag order:
    # rating -> style -> subjects -> character features -> clothing -> action -> setting
    prompt = "rating:explicit, realistic, anime, 1boy, 1girl, hetero, kneeling, deepthroat, irrumatio, penis, oral, cum_in_mouth, freckles, long_hair, black_hair, green_eyes, breasts, nipples, choker, thighhighs, ejaculation, ahegao, saliva, grassy_field, outdoors, natural_lighting, pov"
    output_path = f"{ts}_image.png"
    """prompt_file = f"{ts}_prompt.txt"
    with open(prompt_file, 'w') as f:
        f.write(prompt)"""
    
    print(f"Sending request with prompt: {prompt}")
    result = generate_image(prompt, output_file=output_path,shoot_folder="anime_test")
    
    if result.get("success"):
        print(f"✅ Image generated: {result['output_path']}")
    else:
        print(f"❌ Failed: {result.get('error')}")
