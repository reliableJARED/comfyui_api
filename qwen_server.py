import os
import time
from flask import Flask, request, jsonify, render_template_string
from qwen3VL_ablit import Qwen3VLChat
from qwen25_ablit import Qwen25Chat
from lustify_gen import ImageGenerator
import base64
from PIL import Image
app = Flask(__name__)

chat_model = None
image_model = None
LOW_VRAM_MODE = False  # Set to True to optimize for low VRAM usage

def load_model(type='vlm'):
    # Initialize the model globally
    # This will load the model when the script starts
    print("Initializing Chat model...")
    try:
        if type == 'vlm':
            # Option 1: Vision Language Model
            cm = Qwen3VLChat(model_id="huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated")
            model_type = 'vlm'

        elif type == 'lm':
            # Option 2: Text Only Model
            cm = Qwen25Chat(model_id="huihui-ai/Qwen2.5-7B-Instruct-abliterated-v2")
            model_type = 'lm'

        print(f"Model initialized successfully ({model_type}).")
        return {'model': cm, 'type': model_type}
    except Exception as e:
        print(f"Error initializing model: {e}")
        return None
    

# HTML Template for the simple UI
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Qwen3 VL Chat</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background-color: #f9f9f9; }
        .container { display: flex; flex-direction: column; gap: 15px; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        h1 { color: #333; }
        label { font-weight: bold; color: #555; }
        textarea { width: 100%; height: 100px; padding: 10px; border: 1px solid #ddd; border-radius: 4px; box-sizing: border-box; resize: vertical; }
        input[type="text"] { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 4px; box-sizing: border-box; }
        button { padding: 12px; cursor: pointer; background-color: #007bff; color: white; border: none; border-radius: 4px; font-size: 16px; transition: background 0.3s; }
        button:hover { background-color: #0056b3; }
        button:disabled { background-color: #ccc; cursor: not-allowed; }
        #response { white-space: pre-wrap; background: #fff; padding: 20px; border-radius: 8px; margin-top: 20px; border: 1px solid #eee; min-height: 50px; }
        .loading { color: #666; font-style: italic; }
        .error { color: #d9534f; }
    </style>
</head>
<body>
    <h1>Qwen3 VL Chat Interface</h1>
    <div class="container">
        <div>
            <label for="images">Image Paths (comma separated, optional):</label>
            <input type="text" id="images" placeholder="C:\\path\\to\\image1.jpg, C:\\path\\to\\image2.png">
            <small style="color: #777; display: block; margin-top: 5px;">Enter full local file paths to images.</small>
        </div>
        <div>
            <label for="prompt">Prompt:</label>
            <textarea id="prompt" placeholder="Describe the image..."></textarea>
        </div>
        <button id="generateBtn" onclick="generateResponse()">Generate Response</button>
    </div>
    <div id="response"></div>

    <script>
        async function generateResponse() {
            const promptInput = document.getElementById('prompt');
            const imagesInput = document.getElementById('images');
            const responseDiv = document.getElementById('response');
            const generateBtn = document.getElementById('generateBtn');
            
            const prompt = promptInput.value;
            const imagesStr = imagesInput.value;
            
            if (!prompt) {
                alert('Please enter a prompt');
                return;
            }

            // UI State: Loading
            responseDiv.innerHTML = '<div class="loading">Generating response... (this may take a moment)</div>';
            generateBtn.disabled = true;
            generateBtn.innerText = 'Generating...';
            
            // Parse comma-separated image paths
            const images = imagesStr ? imagesStr.split(',').map(s => s.trim()).filter(s => s.length > 0) : [];

            try {
                const res = await fetch('/generate', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        prompt: prompt,
                        images: images
                    })
                });

                const data = await res.json();
                
                if (data.error) {
                    responseDiv.innerHTML = '<div class="error">Error: ' + data.error + '</div>';
                } else {
                    responseDiv.innerText = data.response;
                }
            } catch (e) {
                responseDiv.innerHTML = '<div class="error">Network Error: ' + e.message + '</div>';
            } finally {
                // UI State: Ready
                generateBtn.disabled = false;
                generateBtn.innerText = 'Generate Response';
            }
        }
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/unload_t2i', methods=['POST'])
def unload_t2i_model():
    global image_model
    if image_model:
        print("Unloading Image Generation model...")
        image_model._unload_model()
        image_model = None
        print("Image Generation model unloaded.")
        time.sleep(10)# wait a bit to ensure VRAM is freed
        return jsonify({"status": "Image Generation model unloaded."}),200
    else:
        return jsonify({"status": "No Image Generation model loaded, nothing to unload"}),200
    
@app.route('/unload_chat', methods=['POST'])
def unload_chat_model():
    global chat_model
    if chat_model:
        print("Unloading Chat model...")
        chat_model['model']._unload_model()
        chat_model = None
        time.sleep(10)# wait a bit to ensure VRAM is freed
        print("Chat model unloaded.")
        return jsonify({"status": "Chat model unloaded."}),200
    else:
        return jsonify({"status": "No Chat model loaded, nothing to unload"}),200
    
@app.route('/t2i', methods=['POST'])
def generate_t2i():
    #generates an image from text prompt only
    global image_model
    print("Received Text-to-Image generation request.")
    if LOW_VRAM_MODE:
        #unload chat model to free up VRAM
        global chat_model
        if chat_model:
            print("Low VRAM mode: Unloading Chat model to free up VRAM...")
            chat_model['model']._unload_model()
            chat_model = None
            print("Chat model unloaded.")

    if not image_model:
        print("Loading Image Generation model...")
        image_model = ImageGenerator()
        print("Image Generation model loaded.")

    data = request.json
    if not data or 'prompt' not in data:
        return jsonify({"error": "No prompt provided"}), 400
    
    #get image prompt
    prompt = data['prompt']
    output_dir = data.get('output_dir', None)
    print(f"Generating image for prompt: {prompt}")

    try:
        print(f"Generating image for prompt: {prompt}")
        #time stamp to create unique filename
        timestamp = time.time()
        file_name = f"{int(timestamp)}_image.png"
        output_full = os.path.join(output_dir, file_name) if output_dir else file_name
        image_path = image_model.text_to_image(prompt,shoot_folder=output_dir,output_file=file_name)
        print(f"Image saved to: {image_path}")

        

        if LOW_VRAM_MODE:
            print("Low VRAM mode: Unloading Image Generation model to free up VRAM...")
            image_model._unload_model()
            image_model = None
            print("Image Generation model unloaded.")

        #encoded_image = base64.b64encode(image_bytes).decode('utf-8')
        #return jsonify({"image_base64": encoded_image})
        text_file_name = os.path.splitext(image_path)[0] + '.txt'
        return jsonify({'image_path': image_path, 'prompt_path': text_file_name})
    
    except Exception as e:
        print(f"Image generation error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/generate', methods=['POST'])
def generate():
    global chat_model
    

    #UNLOAD OPTION used to unload model after generation to save memory when used with api workflow
    """
    Handle generation request with optional image inputs and model unloading.
    send JSON with 'prompt', optional 'images' (list of paths), and optional 'unload' (bool).
    schema:
    {'prompt': str, 'images': [str], 'unload': bool}
    """
    data = request.json
    if not data:
        return jsonify({"error": "Invalid JSON data"}), 400
    prompt = data.get('prompt')
    images = data.get('images', [])
    unload = data.get('unload', False)
    model_type = data.get('model_type', 'vlm')  # default to 'vlm' if not provided

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
    

    if not chat_model:
        chat_model = load_model(type=model_type)
        if not chat_model:
            return jsonify({"error": "Model not initialized. Check server logs."}), 500
        
    if chat_model['type'] != model_type:
        #unload current model
        print("Unloading current model to switch types...")
        chat_model['model']._unload_model()
        #load new model type
        chat_model = load_model(type=model_type)
        if not chat_model:
            return jsonify({"error": "Model not initialized. Check server logs."}), 500

    
    
    try:
        model_instance = chat_model['model']
        model_type = chat_model['type']

        # The generate method expects a list of image paths or PIL images
        # We are passing the list of strings (paths) directly
        
        if model_type == 'vlm':
            #open the images if paths are provided
            if images and all(isinstance(img_path, str) for img_path in images):
                images = [Image.open(img_path).convert("RGB") if isinstance(img_path, str) else img_path for img_path in images]
            response_text = model_instance.generate(
                prompt=prompt,
                images=images,
                max_new_tokens=5000,
                streaming=False
            )
        else:
            # Text only model
            response_text = model_instance.generate(
                prompt=prompt,
                max_new_tokens=5000,
                streaming=False
            )

        if unload or LOW_VRAM_MODE:
            print("Unloading model as requested...")
            model_instance._unload_model()

        return jsonify({"response": response_text})
    except Exception as e:
        print(f"Generation error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    #set GPU available before starting flask app
    #os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # Run on 0.0.0.0 to be accessible, port 5055
    # debug=False is important to prevent the reloader from loading the model twice
    app.run(host='0.0.0.0', port=5055, debug=False)
