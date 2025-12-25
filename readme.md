# Lustify V4 NSFW Prompt Guide


## Overview
Lustify is a photorealistic SDXL model trained on both **Danbooru-style tags** and **natural language captions** (via JoyCaption). This dual training means you have flexibility in how you prompt.
https://www.toolify.ai/ai-model/theimposterimposters-lustify-v2-0


## Best Prompting Approach

### Recommended: Hybrid Style (Natural Language + Strategic Tags)
Combine natural language descriptions with key Danbooru tags for optimal results:

```python
prompt = "intimate photograph, 1girl, 1boy, missionary position, bedroom setting, soft lighting, shot on Canon EOS 5D, warm golden hour lighting"
```

### Alternative: Pure Tag Style
Use comma-separated Danbooru tags with underscores:

```python
prompt = "photo (medium), 1girl, 1boy, missionary_position, spread_legs, bedroom, soft_lighting, candid_photo"
```

### Alternative: Pure Natural Language
Descriptive sentences work too:

```python
prompt = "A photograph of a couple engaged in intimate activity in missionary position, soft bedroom lighting, candid style"
```

## Essential NSFW Tags & Terminology

### Character Count
- `1girl, 1boy` - Standard heterosexual pairing
- `2girls` - Lesbian content
- `1girl, solo` - Solo female content

### Common Sexual Positions
- `missionary`, `missionary_position`
- `sex_from_behind`, `doggystyle`
- `cowgirl_position`, `girl_on_top`, `woman_on_top`
- `reverse_cowgirl`
- `standing_sex`
- `spooning`
- `69_position`

### Sexual Acts
- `fellatio`, `oral`, `blowjob`, `deepthroat`
- `cunnilingus`
- `vaginal`, `penetration`, `sex`
- `handjob`
- `titjob`, `paizuri`
- `anal`

### Body Features & Modifiers
- `nude`, `naked`, `topless`, `bottomless`
- `breasts`, `small_breasts`, `large_breasts`, `huge_breasts`
- `nipples`, `pussy`, `penis`, `erection`
- `spread_legs`, `legs_apart`
- `straddling`
- `arched_back`

### Perspectives & Focus
- `pov`, `pov_crotch`
- `from_behind`, `from_below`, `from_above`
- `close-up`, `wide_shot`
- `male_focus`, `female_focus`

### Clothing States
- `lingerie`, `underwear`, `panties`, `bra`
- `torn_clothes`, `clothes_pull`
- `partially_undressed`
- `stockings`, `thigh_highs`, `pantyhose`
- `sheer_legwear`

### Intimacy & Expression
- `sex`, `hetero`
- `kissing`, `french_kiss`
- `looking_at_viewer`, `eye_contact`
- `seductive_smile`, `open_mouth`
- `sweat`, `saliva`

## Photography & Style Tags (High Impact)

### Camera Types
- `shot on Polaroid SX-70`
- `shot on Kodak Funsaver`
- `shot on Canon EOS 5D`
- `shot on Leica T`
- `shot on GoPro Hero`

### Photography Styles
- `analog photo`
- `glamour photography`
- `candid photo`
- `amateur photo` (more realistic/less polished)
- `street fashion photography`

### Lighting Types
- `cinematic lighting`
- `soft lighting`
- `warm golden hour lighting`
- `dramatic lighting`
- `low key lighting`
- `neon lighting`
- `bright flash photography`
- `radiant god rays`

### Film Types
- `Ilford HP5 Plus`
- `Lomochrome color film`
- `Fujicolor Pro`

### Quality Modifiers
- For amateur/candid aesthetic: `lowres, low quality, candid photo, amateur`

### Subreddit Tags (via bigASP)
- `r/amateur`
- `r/gonewild`
- `r/traps`

## Settings for V4/V6

### CFG Scale
- **V4**: 4.5-6 typically
- **V6**: Lower CFG ~3-4.5 recommended

### Prompt Length
- Keep prompts reasonable, avoid "schizophrompoting" (overly long/complex)
- V6 specifically: **Less is more** - avoid unreasonably long prompts

### Negative Prompts
- V6: Try to avoid or minimize negative prompts
- If needed, keep them simple and targeted

## Example Prompts

### Example 1: Missionary (Hybrid Style)
```python
prompt = "intimate bedroom photograph, 1girl, 1boy, missionary position, spread legs, eye contact, soft lighting, shot on Canon EOS 5D, warm tones, candid photo"
cfg_scale = 5.0
```

### Example 2: Oral (Natural Language Heavy)
```python
prompt = "photograph of a woman performing fellatio, pov perspective, bedroom setting, low key lighting, shot on Polaroid SX-70, intimate atmosphere, amateur photo"
cfg_scale = 4.5
```

### Example 3: Cowgirl (Tag Heavy)
```python
prompt = "photo (medium), 1girl, 1boy, cowgirl_position, woman_on_top, straddling, arched_back, bedroom, cinematic_lighting, shot_on_Canon_EOS_5D"
cfg_scale = 5.5
```

### Example 4: Amateur Aesthetic
```python
prompt = "candid amateur photograph, 1girl, 1boy, sex from behind, bedroom, lowres, low quality, amateur photo, natural lighting, r/amateur"
cfg_scale = 4.0
```

### Example 5: Glamour Style
```python
prompt = "glamour photography, 1girl, nude, posing on bed, large breasts, seductive smile, looking at viewer, dramatic lighting, shot on Leica T, high quality"
cfg_scale = 5.0
```

## Key Tips

1. **Underscores vs Spaces**: Both work, but use underscores for pure Danbooru-style tags (`missionary_position`) and spaces when mixing with natural language
2. **Front-load important concepts**: Place key elements early in the prompt
3. **Photography tags have major visual impact**: Camera types and lighting dramatically affect the output
4. **Use commas to separate concepts**
5. **Parentheses/weight syntax doesn't work** in raw diffusers - they're treated as literal characters
6. **Quality matters less than content**: Focus on describing what you want rather than quality tags
7. **Experiment with hybrid approaches**: Mix tags and natural language for best results

## Technical Notes

- Model trained on both JoyCaption (natural language) and JoyTag (Danbooru tags extended to photos)
- bigASP is a major component, which was trained 90% on natural language captions
- Token limit is 75 tokens per CLIP encoder - keep this in mind for very long prompts
- Lustify is NOT Pony-based or Illustrious-based - don't use LoRAs from those models

## Version Notes

- **V2**: Struggled with doggystyle
- **V3**: Unpublished (poor quality)
- **V4**: Improved all-around, good for most positions
- **V5**: Most stable version
- **V6**: Latest, requires lower CFG and shorter prompts

---

*This guide is for educational purposes. Always ensure your use complies with applicable laws and platform terms of service.*