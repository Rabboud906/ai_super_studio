import os
import base64
from flask import Flask, request, jsonify, render_template
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from pypdf import PdfReader
from io import BytesIO
import requests
from huggingface_hub import InferenceClient
from PIL import Image
import io

app = Flask(__name__)
# Initialize Hugging Face Client
HF_API_KEY = os.getenv("HF_API_KEY")  # Replace with your actual API key
client = InferenceClient(api_key=HF_API_KEY)
# --- CONFIGURATION ---
# --- UPDATED CONFIGURATION (Use these models) ---
MODELS = {
    "chat": "Qwen/Qwen2.5-7B-Instruct",
    "image": "stabilityai/stable-diffusion-xl-base-1.0",
    "vision": "Salesforce/blip-image-captioning-large",
    "video": "cerspense/zeroscope_v2_576w"
}

@app.route('/')
def home():
    return render_template('index.html')

# 1. TEXT CHAT
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    messages = [{"role": "user", "content": user_input}]
    
    try:
        response = client.chat_completion(
            model=MODELS["chat"],
            messages=messages, 
            max_tokens=500
        )
        return jsonify({'response': response.choices[0].message.content})
    except Exception as e:
        return jsonify({'error': str(e)})

# 2. IMAGE GENERATION (Text-to-Image)
@app.route('/generate-image', methods=['POST'])
def generate_image():
    prompt = request.json.get('prompt')
    try:
        image = client.text_to_image(prompt, model=MODELS["image"])
        
        # Save image to a static folder so the frontend can display it
        if not os.path.exists('static'):
            os.makedirs('static')
        filename = f"static/generated_{os.urandom(4).hex()}.png"
        image.save(filename)
        
        return jsonify({'image_url': filename})
    except Exception as e:
        return jsonify({'error': str(e)})


# 3. FILE CHAT (PDF Upload)
@app.route('/chat-file', methods=['POST'])
def chat_file():
    if 'file' not in request.files:
        return jsonify({'error': "No file uploaded"})
    
    file = request.files['file']
    user_question = request.form.get('question')

    # Extract text from PDF
    try:
        reader = PdfReader(file)
        text_content = ""
        for page in reader.pages:
            text_content += page.extract_text() + "\n"
            
        # Limit text length to avoid token limits (approx 3000 chars for safety)
        text_context = text_content[:3000] 

        prompt = f"Here is the content of a document:\n\n{text_context}\n\nBased on this document, answer the user's question: {user_question}"
        
        messages = [{"role": "user", "content": prompt}]
        response = client.chat_completion(model=MODELS["chat"], messages=messages, max_tokens=500)
        
        return jsonify({'response': response.choices[0].message.content})
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True, port=5000)