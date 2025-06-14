from flask import Flask, request, jsonify
from deepface import DeepFace

app = Flask(__name__)

@app.route('/')
def home():
    return 'Emotion Detection API is Running!'

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    img = request.files['image']
    result = DeepFace.analyze(img_path=img, actions=['emotion'], enforce_detection=False)
    emotion = result[0]['dominant_emotion']
    return jsonify({'emotion': emotion})

if __name__ == '__main__':
    app.run(debug=True)