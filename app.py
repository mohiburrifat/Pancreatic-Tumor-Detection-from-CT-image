from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
import os
import time

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

# ============================================================
# CONFIGURATION
# ============================================================

UPLOAD_FOLDER = "static/uploads"
GRADCAM_FOLDER = "static/gradcam"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRADCAM_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["GRADCAM_FOLDER"] = GRADCAM_FOLDER

# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = os.path.join(
    os.getcwd(),
    "final_mobilenetv2_pancreatic_tumor.keras"
)

print("Model Path:", MODEL_PATH)
print("Model Exists:", os.path.exists(MODEL_PATH))

# ============================================================
# LOAD MODEL
# ============================================================

model = load_model(MODEL_PATH)

print("===================================")
print("MODEL LOADED SUCCESSFULLY")
print("===================================")

# ============================================================
# PARAMETERS
# ============================================================

IMG_SIZE = 224

class_names = [
    "No Tumor",
    "Tumor"
]

# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess_image(img_path):

    img = image.load_img(
        img_path,
        target_size=(IMG_SIZE, IMG_SIZE)
    )

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)

    img_array = preprocess_input(img_array)

    return img_array

# ============================================================
# EXPERT SYSTEM
# ============================================================

def expert_analysis(prediction, confidence):

    if prediction == "Tumor":

        if confidence >= 95:
            risk = "Critical Risk"

            recommendation = (
                "Immediate consultation with an oncologist is strongly recommended."
            )

        elif confidence >= 80:
            risk = "High Risk"

            recommendation = (
                "Tumor-like abnormalities detected. Further medical analysis recommended."
            )

        else:
            risk = "Moderate Risk"

            recommendation = (
                "Potential abnormality detected. Clinical verification advised."
            )

    else:

        risk = "Low Risk"

        recommendation = (
            "No tumor patterns detected by the AI model."
        )

    return risk, recommendation

# ============================================================
# GENERATE GRADCAM
# ============================================================

def generate_gradcam(img_path):

    last_conv_layer_name = "Conv_1"

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    processed_img = preprocess_image(img_path)

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(
            processed_img,
            training=False
        )

        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

    heatmap = tf.squeeze(heatmap)

    heatmap = heatmap.numpy()

    heatmap = np.maximum(heatmap, 0)

    max_heat = np.max(heatmap)

    if max_heat != 0:
        heatmap /= max_heat

    original_img = cv2.imread(img_path)

    original_img = cv2.resize(
        original_img,
        (IMG_SIZE, IMG_SIZE)
    )

    heatmap = cv2.resize(
        heatmap,
        (IMG_SIZE, IMG_SIZE)
    )

    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    superimposed_img = cv2.addWeighted(
        original_img,
        0.6,
        heatmap,
        0.4,
        0
    )

    filename = os.path.basename(img_path)

    gradcam_filename = "gradcam_" + filename

    gradcam_path = os.path.join(
        app.config["GRADCAM_FOLDER"],
        gradcam_filename
    )

    cv2.imwrite(
        gradcam_path,
        superimposed_img
    )

    return gradcam_filename

# ============================================================
# HOME ROUTE
# ============================================================

@app.route("/", methods=["GET", "POST"])

def index():

    if request.method == "POST":

        if "file" not in request.files:

            return render_template(
                "index.html",
                error="No file uploaded!"
            )

        file = request.files["file"]

        if file.filename == "":

            return render_template(
                "index.html",
                error="Please select an image!"
            )

        filepath = os.path.join(
            app.config["UPLOAD_FOLDER"],
            file.filename
        )

        file.save(filepath)

        processed_img = preprocess_image(filepath)

        start_time = time.time()

        prediction = model.predict(
            processed_img,
            verbose=0
        )[0][0]

        end_time = time.time()

        inference_time = end_time - start_time

        if prediction > 0.5:

            predicted_class = class_names[1]

            confidence = prediction * 100

        else:

            predicted_class = class_names[0]

            confidence = (1 - prediction) * 100

        risk_level, recommendation = expert_analysis(
            predicted_class,
            confidence
        )

        gradcam_filename = generate_gradcam(filepath)

        return render_template(
            "index.html",
            prediction=predicted_class,
            confidence=round(confidence, 2),
            inference_time=round(inference_time, 4),
            uploaded_image=file.filename,
            gradcam_image=gradcam_filename,
            risk_level=risk_level,
            recommendation=recommendation
        )

    return render_template("index.html")

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )