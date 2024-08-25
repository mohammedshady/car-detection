from flask import Flask, request, jsonify,send_file
from PIL import Image
from io import BytesIO
from flask_cors import CORS
import os
from script import process_image
import cv2
from flask_socketio import SocketIO, emit

app = Flask(__name__)
CORS(app)  # Apply CORS to all routes and all origins by default
socketio = SocketIO(app, cors_allowed_origins='*')

def emit_progress(message):
    socketio.emit('progress_update', {'message': message})

@app.route('/upload', methods=['POST'])
def handle_upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        filename = file.filename
        save_path = os.path.join("./temp", filename)
        file.save(save_path)
        return jsonify({"message": "File successfully uploaded", "filename": filename}), 200
    
    
@app.route('/process/<filename>', methods=['GET'])
def process(filename):
    original_path = os.path.join('./temp', filename)
    if not os.path.exists(original_path):
        return jsonify({"error": "File does not exist"}), 404
    
    # Define path for processed image
    output_filename = f"processed_{filename}"
    output_path = os.path.join('./temp', output_filename)

    # Process the image
    processed_image_array = process_image(original_path, output_path,emit_progress)
    processed_image_array = cv2.cvtColor(processed_image_array, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(processed_image_array.astype('uint8'))

    # Save the PIL Image to a buffer
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)

    # Return the processed image
    return send_file(buffer, mimetype='image/jpeg')



if __name__ == "__main__":
    app.run(debug=True)