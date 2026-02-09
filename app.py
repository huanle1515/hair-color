import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import requests

# --- 1. Setup & Caching ---
# We cache the model loading so it doesn't reload every time you change a color
@st.cache_resource
def load_segmenter():
    model_path = "hair_segmenter.tflite"
    # Auto-download model if missing
    if not os.path.exists(model_path):
        url = "https://storage.googleapis.com/mediapipe-models/image_segmenter/hair_segmenter/float32/1/hair_segmenter.tflite"
        r = requests.get(url, allow_redirects=True)
        with open(model_path, 'wb') as f:
            f.write(r.content)
    
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.ImageSegmenterOptions(
        base_options=base_options,
        output_confidence_masks=True
    )
    return vision.ImageSegmenter.create_from_options(options)

segmenter = load_segmenter()

# --- 2. Your Processing Logic (Helper Functions) ---
def srgb_to_linear(x):
    x = np.clip(x, 0.0, 1.0)
    a = 0.055
    return np.where(x <= 0.04045, x / 12.92, ((x + a) / (1 + a)) ** 2.4)

def linear_to_srgb(x):
    x = np.clip(x, 0.0, 1.0)
    a = 0.055
    return np.where(x <= 0.0031308, x * 12.92, (1 + a) * (x ** (1/2.4)) - a)

def hex_to_rgb01(hex_color):
    s = hex_color.lstrip("#")
    return np.array([int(s[i:i+2], 16) for i in (0,2,4)], dtype=np.float32) / 255.0

def refine_mask(mask):
    m = (mask * 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
    m = cv2.GaussianBlur(m, (0, 0), 3)
    return m.astype(np.float32) / 255.0

def recolor_hair_unified(img_bgr, mask, target_hex, strength=0.7, mask_gamma=1.6):
    # Prep
    m = np.clip(mask, 0.0, 1.0).astype(np.float32)
    m = m ** mask_gamma
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb_lin = srgb_to_linear(rgb)

    # Luminance
    Y = 0.2126 * rgb_lin[...,0] + 0.7152 * rgb_lin[...,1] + 0.0722 * rgb_lin[...,2]
    hair = m > 0.35
    if not np.any(hair):
        return img_bgr

    # Target color analysis
    tgt = hex_to_rgb01(target_hex)
    tgt_lin = srgb_to_linear(tgt)
    Yt = 0.2126*tgt_lin[0] + 0.7152*tgt_lin[1] + 0.0722*tgt_lin[2]
    Yt = max(1e-6, float(Yt))
    tgt_sat = np.max(tgt_lin) - np.min(tgt_lin)
    
    # Coolness gate
    coolness = float(tgt_lin[2] - tgt_lin[0])
    cool_color = np.clip((coolness - 0.05) / 0.25, 0.0, 1.0)
    tgt_chroma = tgt_lin / Yt

    # Adaptive alpha
    Yh = Y[hair]
    if len(Yh) == 0: return img_bgr # Safety check
    p85 = np.percentile(Yh, 85)
    p20 = np.percentile(Yh, 20)

    hi_w = np.clip((Y - p85) / max(1e-6, (Yh.max() - p85)), 0, 1)
    sh_w = np.clip((p20 - Y) / max(1e-6, p20), 0, 1)

    alpha = m * strength
    alpha *= (1.0 - hi_w * 0.5)
    alpha *= (1.0 - sh_w * 0.3)
    alpha = np.clip(alpha, 0, 1)

    # Vivid Color Lift
    is_vivid = np.clip((tgt_sat - 0.20) / 0.35, 0.0, 1.0)
    base_blend = 0.10
    vivid_blend = 0.45 * is_vivid * cool_color
    blend_factor = base_blend + vivid_blend

    Y_mix = (Y * (1.0 - blend_factor) + Yt * blend_factor)[..., None]
    dark_hair = np.clip((0.30 - Y) / 0.30, 0.0, 1.0)
    Y_mix += 0.10 * dark_hair[..., None] * is_vivid * cool_color
    Y_mix = np.clip(Y_mix, 0.0, 1.0)

    chroma_strength = 0.85
    tgt_scaled = tgt_chroma[None, None, :] * Y_mix * chroma_strength
    tgt_scaled = np.clip(tgt_scaled, 0.0, 1.2)

    out_lin = rgb_lin * (1 - alpha[...,None]) + tgt_scaled * alpha[...,None]

    # Root depth
    h, w = m.shape
    y = np.linspace(0, 1, h)[:,None]
    root = np.clip(1 - y * 3, 0, 1)
    out_lin *= (1 - root[...,None] * m[...,None] * 0.15)

    out_rgb = linear_to_srgb(out_lin)
    return (out_rgb * 255).astype(np.uint8)

# --- 3. The UI Layout ---
st.title("✨ Virtual Hair Color Try-On")
st.write("Upload a photo to test our new color blending algorithm.")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Controls")
    # File Uploader
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    # Interactive Widgets
    target_color = st.color_picker("Pick a Hair Color", "#5e3c58")
    strength = st.slider("Blend Strength", 0.0, 1.0, 0.75)
    
    st.info("💡 Tip: Try vivid colors to see the 'Bleach Simulation' effect.")

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    
    # Process
    with st.spinner('Segmenting hair...'):
        H, W = img_bgr.shape[:2]
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        )
        result = segmenter.segment(mp_image)
        
        # Get mask
        if len(result.confidence_masks) > 1:
            hair_mask = result.confidence_masks[1].numpy_view()
            hair_mask = cv2.resize(hair_mask, (W, H))
            hair_mask = refine_mask(hair_mask)
            
            # Recolor
            result_img = recolor_hair_unified(img_bgr, hair_mask, target_color, strength=strength)
            
            # Display Result in Main Column
            with col2:
                st.header("Result")
                st.image(result_img, caption="Recolored Result", use_container_width=True)
                
                # Show mask for debugging (optional)
                with st.expander("See Hair Mask (Debug)"):
                    st.image(hair_mask, caption="Segmentation Mask", clamp=True)
        else:
            st.error("Could not detect hair in this image!")