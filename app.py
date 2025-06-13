import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.models import load_model
import rasterio
import tempfile
import os

# Warna ke kelas
COLOR_MAP = {
    (0, 255, 255): 0,  # urban_land
    (255, 255, 0): 1,  # agriculture_land
    (255, 0, 255): 2,  # rangeland
    (0, 255, 0): 3,    # forest_land
    (0, 0, 255): 4,    # water
    (255, 255, 255): 5,  # barren_land
    (0, 0, 0): 6       # unknown
}

CLASS_LABELS = {
    0: "urban_land",
    1: "agriculture_land",
    2: "rangeland",
    3: "forest_land",
    4: "water",
    5: "barren_land",
    6: "unknown"
}

def class_to_color(class_mask, color_map):
    color_image = np.zeros((class_mask.shape[0], class_mask.shape[1], 3), dtype=np.uint8)
    for color, class_id in color_map.items():
        color_image[class_mask == class_id] = color
    return color_image

def predict_image(image_path, model, tile_size=256):
    with rasterio.open(image_path) as src:
        img_array = src.read().transpose(1, 2, 0)
        transform = src.transform
        crs = src.crs

    img_array_rgb = img_array[:, :, :3] / 255.0
    height, width = img_array_rgb.shape[:2]
    pred_classes_full = np.zeros((height, width), dtype=np.uint8)

    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            y_end = min(y + tile_size, height)
            x_end = min(x + tile_size, width)
            tile = img_array_rgb[y:y_end, x:x_end, :]

            tile_padded = np.zeros((tile_size, tile_size, 3), dtype=np.float32)
            tile_padded[:tile.shape[0], :tile.shape[1], :] = tile
            tile_input = np.expand_dims(tile_padded, axis=0)

            pred_tile = model.predict(tile_input, verbose=0)
            pred_classes_tile = np.argmax(pred_tile, axis=-1)[0]
            pred_classes_full[y:y_end, x:x_end] = pred_classes_tile[:tile.shape[0], :tile.shape[1]]

    return img_array_rgb, pred_classes_full

# === Streamlit App ===
st.set_page_config(page_title="UNet Satelit Inferensi", layout="wide")
st.title("🛰️ Segmentasi Citra Satelit dengan UNet")

uploaded_file = st.file_uploader("Upload file citra satelit (.tif)", type=["tif", "tiff"])

if uploaded_file:
    # Simpan file sementara
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Load model
    with st.spinner("🔄 Memuat model UNet..."):
        model = load_model("model/best_model_unet_optimasi.h5")

    # Prediksi
    with st.spinner("🧠 Melakukan prediksi..."):
        original_img, prediction_mask = predict_image(tmp_path, model)

    # Visualisasi
    pred_colored = class_to_color(prediction_mask, COLOR_MAP)

    st.subheader("📷 Hasil Visualisasi")
    col1, col2 = st.columns(2)
    with col1:
        st.image((original_img * 255).astype(np.uint8), caption="Gambar Asli", use_column_width=True)
    with col2:
        st.image(pred_colored, caption="Segmentasi UNet", use_column_width=True)

    # Cleanup
    os.remove(tmp_path)
