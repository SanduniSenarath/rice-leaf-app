import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ==========================
# Load TFLite Model
# ==========================
@st.cache_resource
def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# Path to your TFLite model (update if needed)
MODEL_PATH = "rice_leaf_model.tflite"
interpreter = load_tflite_model(MODEL_PATH)

# Get model input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Labels (update with your classes in correct order)
CLASS_NAMES = ["Bacterial Leaf Blight", "Brown Spot", "Leaf Smut"]

# ==========================
# Preprocess Function
# ==========================
def preprocess_image(image: Image.Image, target_size=(224, 224)):
    img = image.resize(target_size)
    img = np.array(img).astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ==========================
# Prediction Function
# ==========================
def predict(image: Image.Image):
    img_input = preprocess_image(image)
    interpreter.set_tensor(input_details[0]['index'], img_input)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])[0]
    pred_idx = np.argmax(preds)
    confidence = float(np.max(preds))
    return CLASS_NAMES[pred_idx], confidence, preds

# ==========================
# Streamlit UI
# ==========================
st.set_page_config(page_title="🌾 Rice Leaf Disease Classifier", layout="centered")

st.title("🌾 Rice Leaf Disease Classification")
st.write("Upload a rice leaf image to predict its disease class using a **MobileNetV2 (TFLite)** model.")

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("🔎 Running prediction...")
    label, confidence, preds = predict(image)

    st.success(f"✅ Predicted: **{label}** ({confidence*100:.2f}% confidence)")

    # Show probabilities
    st.subheader("Prediction Probabilities")
    prob_dict = {CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))}
    st.bar_chart(prob_dict)
