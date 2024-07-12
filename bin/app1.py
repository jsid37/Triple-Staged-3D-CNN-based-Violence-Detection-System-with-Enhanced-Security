from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import cv2
import os
from mamonfight22 import video_mamonreader, mamon_videoFightModel, pred_fight

app = Flask(__name__)

# Load the pre-trained model
model = mamon_videoFightModel()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/detect-violence', methods=['POST'])
def detect_violence():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Read video frames
        video_data = video_mamonreader(cv2, file_path)
        
        # Predict violence
        is_violence, probability = pred_fight(model, video_data)
        
        # Delete the uploaded file
        os.remove(file_path)
        
        # Prepare response
        response = {
            'violence_detected': is_violence,
            'probability': probability
        }
        return jsonify(response), 200

if __name__ == "__main__":
    app.config['UPLOAD_FOLDER'] = 'uploads'
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
