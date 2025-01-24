from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import io

app = Flask(__name__)

# Load model YOLOv8
model = YOLO("best10epochs.pt")  # Ganti "model.pt" dengan path model YOLOv8 Anda

@app.route('/', methods=['POST'])
def test_get():
    return jsonify({
        "info": "success"
    })

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files['image']
    
    try:
        # Baca file gambar
        image = Image.open(io.BytesIO(file.read()))
        
        # Deteksi objek dengan YOLOv8
        results = model(image)
        
        # Ekstrak informasi deteksi
        detections = []
        for box in results[0].boxes:
            detections.append({
                "class": int(box.cls),  # Kelas objek
                "confidence": float(box.conf),  # Confidence score
                "bbox": box.xyxy.tolist()  # [xmin, ymin, xmax, ymax]
            })
        
        return jsonify({"detections": detections})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
