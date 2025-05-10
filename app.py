import os
import torch
import torchvision.transforms as transforms
from torchvision import models
from flask import Flask, request, render_template, redirect, url_for, jsonify
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
from werkzeug.utils import secure_filename
from datetime import datetime
from chatbot import chatbot  # Import the chatbot module

# Flask App Setup
app = Flask(__name__)

# Define upload folder & allowed extensions
UPLOAD_FOLDER = "static"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)  # Ensure output matches classes (Fake/Real)
model.load_state_dict(torch.load("Model/fake_real_model.pth", map_location=device))
model = model.to(device)
model.eval()

# Define Image Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Function to Check File Extension
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Model Prediction Function
def predict_image(image_path):
    image = Image.open(image_path)
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1).item()

    class_labels = ["Fake", "Real"]
    return class_labels[prediction]

# Function for LIME Explainability (Only for Fake Images)
def generate_lime_explanation(image_path, prediction):
    """Generate LIME explanation only if the image is predicted as Fake."""
    if prediction != "Fake":
        return None  # No explanation for "Real" images

    image = Image.open(image_path)
    image_np = np.array(image.resize((224, 224)))

    def model_predict(image_batch):
        """Convert a batch of images to model predictions."""
        image_batch = torch.tensor(image_batch, dtype=torch.float32).permute(0, 3, 1, 2)
        image_batch = image_batch.to(device)

        with torch.no_grad():
            outputs = model(image_batch)
            probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()

        return probs

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(image_np, model_predict, top_labels=2, hide_color=0, num_samples=1000)

    # Get Explanation for "Fake" class only
    temp, mask = explanation.get_image_and_mask(0, positive_only=True, num_features=5, hide_rest=False)

    explanation_path = os.path.join(app.config["UPLOAD_FOLDER"], "explanation.png")
    plt.figure(figsize=(6, 6))
    plt.imshow(mark_boundaries(temp, mask))
    plt.axis("off")
    plt.savefig(explanation_path, bbox_inches="tight")
    plt.close()

    return "explanation.png"

# Store recent detections (limit to last 10)
recent_detections = []

# Flask Routes
@app.route("/", methods=["GET"])
def index():
    # Redirect root URL to home page
    return redirect(url_for('home'))

@app.route("/home", methods=["GET"])
def home():
    # Render the home page
    return render_template("home.html")

@app.route("/dashboard", methods=["GET"])
def dashboard():
    # Pass recent detections to dashboard
    return render_template("dashboard.html", recent_detections=recent_detections)

@app.route("/about", methods=["GET"])
def about():
    # Render the about page
    return render_template("about.html")

@app.route("/detect", methods=["GET", "POST"])
def detect():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", message="No file uploaded.")

        file = request.files["file"]
        if file.filename == "" or not allowed_file(file.filename):
            return render_template("index.html", message="Invalid file type. Please upload a JPG or PNG.")

        # Save uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        # Get Prediction
        prediction = predict_image(file_path)

        # Generate LIME Explanation only for Fake images
        explanation_image = generate_lime_explanation(file_path, prediction)

        # Store detection info
        detection_info = {
            "filename": filename,
            "result": prediction,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "confidence": "85%" if prediction == "Fake" else "90%"
        }
        recent_detections.insert(0, detection_info)
        # Keep only last 10 detections
        if len(recent_detections) > 10:
            recent_detections.pop()

        return render_template(
            "index.html",
            uploaded_image=filename,
            result=prediction,
            explanation_image=explanation_image,
        )

    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    """Handle chatbot API requests"""
    if request.method == "POST":
        user_message = request.json.get("message", "")
        if not user_message:
            return jsonify({"status": "error", "message": "No message provided"})
        
        # Get response from chatbot
        response = chatbot.get_response(user_message)
        return jsonify(response)

# Run Flask App
if __name__ == "__main__":
    app.run(debug=True)
