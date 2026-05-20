Pancreatic Tumor Detection from CT Images

A Deep Learning–Based Web Application for Detecting Pancreatic Tumors from CT Scan Images using MobileNetV2 and Grad-CAM Visualization.

📖 Overview

This project is a medical image classification web application that detects pancreatic tumors from CT scan images using a trained MobileNetV2 deep learning model.

The system allows users to upload CT images through a simple web interface and receive:

✅ Tumor prediction results
✅ Confidence score
✅ Grad-CAM heatmap visualization

The project is built with Flask, TensorFlow/Keras, and OpenCV.

✨ Features
📤 Upload CT scan images
🧠 Tumor detection using MobileNetV2
🔥 Grad-CAM visualization for explainability
🌐 User-friendly Flask web interface
⚡ Real-time prediction results
🎨 Clean frontend with HTML & CSS
🛠️ Technologies Used
Technology	Purpose
Python	Backend development
Flask	Web framework
TensorFlow / Keras	Deep learning
MobileNetV2	CNN model
OpenCV	Image processing
NumPy	Numerical operations
HTML/CSS	Frontend UI


📂 Project Structure
Pancreatic-Tumor-Detection-from-CT-image/
│
├── app.py
├── final_mobilenetv2_pancreatic_tumor.keras
│
├── templates/
│   └── index.html
│
├── static/
│   ├── style.css
│   ├── uploads/
│   └── gradcam/



🚀 Installation Guide
1️⃣ Clone the Repository
git clone https://github.com/your-username/Pancreatic-Tumor-Detection-from-CT-image.git
cd Pancreatic-Tumor-Detection-from-CT-image
2️⃣ Create Virtual Environment (Optional but Recommended)
Windows
python -m venv venv
venv\Scripts\activate
Linux / Mac
python3 -m venv venv
source venv/bin/activate
3️⃣ Install Required Packages
pip install -r requirements.txt
▶️ Run the Application
python app.py

Now open your browser and visit:

http://127.0.0.1:5000
