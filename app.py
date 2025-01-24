from flask import Flask, request, jsonify, send_from_directory
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
import io
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load model YOLOv8
model = YOLO("best.pt")

CLASS_LABELS = {
    0: "Memakai Helm",
    1: "Tidak Memakai Helm",
}

@app.route('/', methods=['GET'])
def test_get():
    return jsonify({
        "info": "success"
    })

@app.route('/output/<path:filename>', methods=['GET'])
def get_file(filename):
    try:
        return send_from_directory("./output", filename)
    except FileNotFoundError:
        return jsonify({"error": "File not found"}), 404

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files['image']

    try:
        # Baca file gambar
        file.seek(0)
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

        detections = process_detections(detections)

        file.seek(0)
        output_image = draw_bounding_boxes(file, detections)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"./output/{timestamp}-output.jpg"
        path = f"/output/{timestamp}-output.jpg"

        with open(filename, "wb") as f:
            f.write(output_image.getvalue())  # Simpan buffer ke file

        return jsonify({"image": path})

    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500

def process_detections(raw_detections):
    processed = []
    for det in raw_detections:
        if isinstance(det["bbox"], list) and len(det["bbox"]) == 1:
            det["bbox"] = det["bbox"][0]
        processed.append(det)
    return processed

def draw_bounding_boxes(file, detections):
    # Baca gambar dari buffer file
        image = Image.open(io.BytesIO(file.read()))
        draw = ImageDraw.Draw(image)

        # Load font default
        font = ImageFont.truetype("/font/Roboto-Black.ttf", 20)

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            class_id = det["class"]
            confidence = det["confidence"]

            # Ambil label deskriptif dari CLASS_LABELS
            label_name = CLASS_LABELS.get(class_id, "Unknown")
            label = f"{label_name} ({confidence * 100:.2f}%)"

            # Gambar bounding box
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

            # Hitung ukuran teks menggunakan textbbox()
            text_bbox = draw.textbbox((x1, y1), label, font=font)

            # Tambahkan latar belakang untuk teks
            draw.rectangle([text_bbox[0], text_bbox[1], text_bbox[2], text_bbox[3]], fill="red")

            # Tulis teks
            draw.text((x1, y1), label, fill="white", font=font)

        # Simpan gambar hasil ke buffer
        output_buffer = io.BytesIO()
        image.save(output_buffer, format="JPEG")  # Simpan gambar hasil ke buffer
        output_buffer.seek(0)  # Reset posisi pointer buffer
        return output_buffer


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
