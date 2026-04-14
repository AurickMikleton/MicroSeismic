from flask import Blueprint, jsonify, request
from PIL import Image
from src.MachineLearning import (
    device,
    get_training_state,
    start_training,
    predict_image,
    get_dataset_info,
)

homescreen_bp = Blueprint("homescreen", __name__)


@homescreen_bp.route("/", methods=["GET"])
def homescreen():
    return jsonify({"status": "ok", "device": device})


@homescreen_bp.route("/status", methods=["GET"])
def status():
    return jsonify({
        "device":   device,
        "training": get_training_state(),
        "dataset":  get_dataset_info(),
    })


@homescreen_bp.route("/train", methods=["POST"])
def train():
    data   = request.get_json(silent=True) or {}
    epochs = max(1, min(int(data.get("epochs", 10)), 100))
    started = start_training(epochs=epochs)
    if not started:
        return jsonify({"error": "Training is already running"}), 409
    return jsonify({"message": "Training started", "epochs": epochs})


@homescreen_bp.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    try:
        img    = Image.open(request.files["file"].stream).convert("RGB")
        result = predict_image(img)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500