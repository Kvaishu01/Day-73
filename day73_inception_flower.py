import streamlit as st
import numpy as np
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

# -------------------------------
# 🎯 PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Day 73 - Inception Flower Classifier", page_icon="🌸", layout="centered")

st.title("🌸 Day 73: Inception Flower Classifier ")
st.markdown("""
This is a **demo Streamlit app** showing how InceptionV3 can classify flowers.  
Currently running **without a dataset** using random images, so no errors occur.
""")

# -------------------------------
# 🔹 LOAD MODEL
# -------------------------------
st.write("Loading InceptionV3 model...")
model = InceptionV3(weights='imagenet')
st.success("✅ Model loaded successfully!")

# -------------------------------
# 🔹 UPLOAD IMAGE OR USE RANDOM
# -------------------------------
uploaded_file = st.file_uploader("Upload a flower image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = image.load_img(uploaded_file, target_size=(299, 299))
else:
    # Generate dummy random image if no upload
    st.info("No image uploaded. Using a random dummy image for demo.")
    img_array = np.random.randint(0, 256, size=(299, 299, 3), dtype=np.uint8)
    img = image.array_to_img(img_array)

st.image(img, caption="Selected Image", use_column_width=True)

# -------------------------------
# 🔹 PREPROCESS & PREDICT
# -------------------------------
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
decoded = decode_predictions(preds, top=3)[0]

st.write("### 🔎 Top Predictions:")
for i, (imagenet_id, label, prob) in enumerate(decoded):
    st.write(f"{i+1}. **{label}** — Probability: {prob:.2f}")

# -------------------------------
# 📘 ABOUT
# -------------------------------
st.markdown("""
---
✅ **Note:** This app is currently a **demo template** running without a dataset.  
You can later replace the dummy/random images with your **own flower dataset**.  

🧠 **Model:** InceptionV3 (Pretrained on ImageNet)  
📈 **Use Case:** Flower classification / Image recognition demo
""")
