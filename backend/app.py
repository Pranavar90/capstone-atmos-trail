from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
from inference.inference_engine import DehazeInference
import io

app = Flask(__name__)
CORS(app)

# Path to the best model
CHECKPOINT = "outputs/checkpoints/best_model.pth"
if not os.path.exists(CHECKPOINT):
    # Fallback to last model if best isn't available
    CHECKPOINT = "outputs/checkpoints/last_model.pth"

engine = None

def get_engine():
    global engine
    if engine is None and os.path.exists(CHECKPOINT):
        engine = DehazeInference(CHECKPOINT)
    return engine

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files["image"]
    temp_path = "outputs/inference/temp_input.png"
    file.save(temp_path)
    
    infra_engine = get_engine()
    if not infra_engine:
        return jsonify({"error": "Model checkpoint not found. Please train first."}), 404
    
    dehazed, t_map, a_light = infra_engine.predict(temp_path)
    
    # Save outputs
    dehazed_path = "outputs/inference/latest_dehazed.png"
    dehazed.save(dehazed_path)
    
    # For simplicity, we return the dehazed image as response
    # In a full app, we might return URLs or base64
    return send_file(dehazed_path, mimetype='image/png')

@app.route("/status", methods=["GET"])
def status():
    return jsonify({
        "model_loaded": get_engine() is not None,
        "checkpoint": CHECKPOINT
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
