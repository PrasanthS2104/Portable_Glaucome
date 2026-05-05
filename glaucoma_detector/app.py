import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.applications import InceptionResNetV2
from PIL import Image
import time

# Define the custom layer
class CustomScaleLayer(tf.keras.layers.Layer):
    def __init__(self, scale=0.17, **kwargs):
        super(CustomScaleLayer, self).__init__(**kwargs)
        self.scale = scale
    def call(self, inputs):
        return tf.cast(inputs, tf.float32) * self.scale

# Load model by recreating architecture and loading weights
@st.cache_resource
def load_glaucoma_model():
    try:
        # Method 1: Try direct load first
        try:
            model = tf.keras.models.load_model(
                'inception_resnet_v2_glaucoma.h5',
                custom_objects={'CustomScaleLayer': CustomScaleLayer}
            )
            st.success("✅ Model loaded successfully!")
            return model
        except Exception as e:
            st.warning("⚠️ Direct load failed, recreating architecture...")
            
            # Method 2: Recreate architecture and load weights
            # Create the exact same architecture as your training
            base_model = InceptionResNetV2(
                weights="imagenet", 
                include_top=False,
                input_tensor=Input(shape=(224, 224, 3))  # Use input_tensor instead of batch_shape
            )
            
            # Add custom layers (same as your training)
            x = base_model.output
            x = CustomScaleLayer(scale=0.17)(x)
            x = Flatten(name="flatten")(x)
            x = Dropout(0.5)(x)
            outputs = Dense(2, activation="softmax")(x)
            
            # Create model
            model = Model(inputs=base_model.input, outputs=outputs)
            
            # Freeze base model layers
            for layer in base_model.layers:
                layer.trainable = False
                
            # Compile model (same as training)
            model.compile(
                loss='categorical_crossentropy', 
                optimizer='adam', 
                metrics=['accuracy']
            )
            
            # Load only weights
            model.load_weights('inception_resnet_v2_glaucoma.h5')
            st.success("✅ Model architecture recreated and weights loaded!")
            return model
            
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# Rest of your Streamlit app code remains the same...
st.set_page_config(page_title="Glaucoma Detector", page_icon="👁️", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #2E86AB; text-align: center; margin-bottom: 2rem; }
    .prediction-box { padding: 20px; border-radius: 10px; margin: 10px 0; text-align: center; }
    .normal { background-color: #d4edda; border: 1px solid #c3e6cb; color: #155724; }
    .glaucoma { background-color: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; }
</style>
""", unsafe_allow_html=True)

# App title
st.markdown('<div class="main-header">👁️ Glaucoma Detection System</div>', unsafe_allow_html=True)
st.write("Upload an eye fundus image below for glaucoma detection.")

# Load model
model = load_glaucoma_model()

# File upload
uploaded_file = st.file_uploader("📷 Choose an eye fundus image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Uploaded Image")
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_column_width=True)
    
    with col2:
        st.subheader("🔄 Processing...")
        with st.spinner("Analyzing image..."):
            time.sleep(1)
            
            # Preprocess image
            img_array = np.array(image)
            img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            img = cv2.resize(img, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255.0
            
            # Predict
            img_batch = np.expand_dims(img, axis=0)
            predictions = model.predict(img_batch)
            probability = predictions[0]
            
            # Determine result
            if probability[1] > 0.5:
                result = "Glaucoma Positive"
                confidence = probability[1] * 100
                result_class = "glaucoma"
            else:
                result = "Glaucoma Negative" 
                confidence = (1 - probability[1]) * 100
                result_class = "normal"
    
    # Display results
    st.markdown(f"""
    <div class="prediction-box {result_class}">
        <h2>{result}</h2>
        <h3>Confidence: {confidence:.2f}%</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Probability breakdown
    st.subheader("📊 Probability Breakdown")
    col1, col2 = st.columns(2)
    
    with col1:
        normal_prob = (1 - probability[1]) * 100
        st.metric("🟢 Glaucoma Negative", f"{normal_prob:.2f}%")
    
    with col2:
        glaucoma_prob = probability[1] * 100
        st.metric("🔴 Glaucoma Positive", f"{glaucoma_prob:.2f}%")
    
    st.warning("⚠️ **Medical Disclaimer:** Consult an ophthalmologist for professional diagnosis.")

elif model is None:
    st.error("❌ **Model file 'inception_resnet_v2_glaucoma.h5' not found in the current directory!**")
    st.info("💡 **Please make sure:**")
    st.info("1. The model file is in the same folder as this app")
    st.info("2. The filename is exactly 'inception_resnet_v2_glaucoma.h5'")
    st.info("3. You have file read permissions")