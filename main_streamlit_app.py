# main_streamlit_app.py (Untuk Model EfficientNetB0 Saja dengan Gemini API)

import streamlit as st
import os
import numpy as np
import gdown # Digunakan untuk mengunduh model dari Google Drive
from tensorflow.keras.models import load_model # Untuk memuat model Keras/TensorFlow
from PIL import Image # Digunakan untuk manipulasi gambar (misalnya, mengubah ukuran)
import requests # Digunakan untuk membuat permintaan HTTP ke Gemini API
import json # Digunakan untuk mengurai respons JSON dari Gemini API

# Import fungsi pra-pemrosesan spesifik untuk EfficientNetB0
# EfficientNetB0 biasanya menggunakan preprocessing yang mirip dengan ResNet atau Inception,
# atau cukup normalisasi 1./255. Jika ada masalah, kita bisa coba tf.keras.applications.efficientnet.preprocess_input
# Namun, untuk EfficientNetB0 yang dilatih dengan ImageDataGenerator rescale=1./255,
# normalisasi 1./255 sudah cukup.
# Kita akan menggunakan normalisasi 1./255 secara langsung di preprocess_image_for_model.

# --- Konfigurasi Model ---
# Direktori untuk menyimpan model yang diunduh
MODEL_DIR = "models"

# Google Drive URL untuk model EfficientNetB0 Anda
# PENTING: GANTI URL INI DENGAN URL MODEL efficientnetb0_fine_tuned_model.h5 ANDA!
MODEL_URLS = {
    "efficientnet": "https://drive.google.com/uc?id=1EIvOgg48eLrj6gZ0gwV5uzOL06xN21Pv" # URL MODEL ANDA
}

# Nama file lokal untuk model
MODEL_FILENAMES = {
    "efficientnet": "efficientnetb0_fine_tuned_model.h5"
}

# Ukuran gambar input yang dibutuhkan oleh model EfficientNetB0
IMAGE_TARGET_SIZES = {
    "efficientnet": (224, 224) # EfficientNetB0 membutuhkan 224x224
}

# Nama kelas untuk hasil prediksi (pastikan urutannya sama dengan saat pelatihan)
CLASS_NAMES = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']

# --- Gemini API Configuration ---
# API Key akan disediakan oleh lingkungan Canvas saat runtime (__initial_auth_token)
# Di Streamlit Cloud, Anda perlu menambahkan ini sebagai Secret.
GEMINI_API_KEY = "" # Ini akan diisi otomatis oleh Canvas/Streamlit Secrets
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# --- Helper Function for Image Preprocessing ---
def preprocess_image_for_model(image_path, model_name):
    """
    Memuat dan pra-memproses gambar sesuai dengan persyaratan model EfficientNetB0.
    """
    target_size = IMAGE_TARGET_SIZES.get(model_name)
    if not target_size:
        raise ValueError(f"Ukuran gambar tidak dikenal untuk model: {model_name}")

    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0) # Tambahkan dimensi batch

    # Untuk EfficientNetB0 yang dilatih dengan ImageDataGenerator rescale=1./255,
    # cukup normalisasi 1./255.
    return img_array / 255.0

# --- Model Loading with Streamlit Caching ---
# @st.cache_resource sangat penting di sini: memastikan model diunduh dan dimuat
# hanya sekali di seluruh sesi pengguna, mencegah operasi berat berulang.
# hash_funcs ditambahkan untuk memaksa cache di-invalidate jika ada perubahan internal.
@st.cache_resource(show_spinner=False, hash_funcs={"_thread.RLock": lambda _: None})
def load_efficientnet_model_cached():
    """
    Mengunduh dan memuat model EfficientNetB0. Fungsi ini di-cache oleh Streamlit.
    """
    loaded_model = None
    
    # Buat direktori model jika belum ada
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    st.markdown("---")
    st.write("Memulai proses memuat model EfficientNetB0 di latar belakang...")
    st.write("Ini mungkin memakan waktu beberapa menit, terutama pada deployment pertama.")
    
    # Gunakan placeholder untuk pesan dinamis
    status_placeholder = st.empty()

    name = "efficientnet" # Nama model yang akan diunduh
    model_filepath = os.path.join(MODEL_DIR, MODEL_FILENAMES[name])
    
    # Unduh model jika belum ada
    if not os.path.exists(model_filepath):
        status_placeholder.info(f"⬇️ Mengunduh model {name}...")
        try:
            # Gunakan fuzzy=True untuk gdown agar lebih tangguh dalam menangani potensi redirect
            gdown.download(url=MODEL_URLS[name], output=model_filepath, quiet=True, fuzzy=True)
            status_placeholder.success(f"✅ Model {name} berhasil diunduh.")
        except Exception as e:
            status_placeholder.error(f"❌ Gagal mengunduh model {name}: {e}")
            st.exception(e) # Tampilkan pengecualian lengkap di Streamlit
            return None # Menunjukkan kegagalan
            
    # Muat model
    status_placeholder.info(f"🧠 Memuat model {name}...")
    try:
        model = load_model(model_filepath)
        loaded_model = model
        status_placeholder.success(f"✅ Model {name} berhasil dimuat.")
    except Exception as e:
        status_placeholder.error(f"❌ Gagal memuat model {name}: {e}")
        st.exception(e) # Tampilkan pengecualian lengkap di Streamlit
        return None # Menunjukkan kegagalan
            
    status_placeholder.success("🎉 Model EfficientNetB0 berhasil dimuat dan siap digunakan!")
    st.markdown("---")
    return loaded_model

# --- Function to get treatment suggestions from Gemini API ---
def get_treatment_suggestions(plant_disease):
    """
    Calls the Gemini API to get treatment suggestions for a given plant disease.
    """
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    if not api_key:
        st.error("API Key untuk Gemini tidak ditemukan. Harap pastikan GEMINI_API_KEY diatur di Streamlit Secrets.")
        return "Tidak dapat memberikan saran perawatan tanpa API Key."

    prompt = f"Berikan saran perawatan singkat dan praktis untuk penyakit daun jagung: {plant_disease}. Fokus pada langkah-langkah yang bisa dilakukan petani. Berikan dalam bentuk poin-poin singkat."
    
    chat_history = []
    chat_history.append({"role": "user", "parts": [{"text": prompt}]})
    
    payload = {"contents": chat_history}
    
    try:
        with st.spinner(f"Mencari saran perawatan untuk {plant_disease} dari AI..."):
            api_url_with_key = f"{GEMINI_API_URL}?key={api_key}"
            
            response = requests.post(api_url_with_key, headers={'Content-Type': 'application/json'}, data=json.dumps(payload))
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
            result = response.json()

            if result and result.get("candidates") and result["candidates"][0].get("content") and result["candidates"][0]["content"].get("parts"):
                return result["candidates"][0]["content"]["parts"][0]["text"]
            else:
                st.warning("Tidak dapat memperoleh saran perawatan dari AI. Struktur respons tidak sesuai.")
                st.json(result) # Display full response for debugging
                return "Tidak ada saran perawatan yang ditemukan."
    except requests.exceptions.RequestException as e:
        st.error(f"Terjadi kesalahan saat memanggil API Gemini: {e}")
        return "Gagal mendapatkan saran perawatan karena masalah koneksi atau API."
    except json.JSONDecodeError as e:
        st.error(f"Terjadi kesalahan saat mengurai respons JSON dari Gemini: {e}")
        return "Gagal mendapatkan saran perawatan karena masalah format data."
    except Exception as e:
        st.error(f"Terjadi kesalahan tidak terduga: {e}")
        return "Terjadi kesalahan saat mendapatkan saran perawatan."


# --- Streamlit Application UI ---

# Set basic page configuration
st.set_page_config(
    page_title="Deteksi Penyakit Daun Jagung (EfficientNetB0 + Gemini AI)",
    page_icon="🌽",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Application Title and Description
st.title("🌽 Deteksi Penyakit Daun Jagung")
st.markdown("""
Aplikasi ini menggunakan model Convolutional Neural Network (CNN) **EfficientNetB0**
untuk mendeteksi penyakit pada daun jagung. Setelah deteksi, aplikasi akan memberikan saran perawatan
menggunakan **Google Gemini AI**.
""")

# Call the cached model loading function
efficientnet_model = load_efficientnet_model_cached()

# Check if model was loaded successfully
if efficientnet_model is None:
    st.error("Aplikasi tidak dapat berfungsi penuh karena ada masalah dalam memuat model. Silakan periksa log deployment.")
    st.stop() # Stop further execution if model is not loaded

# File Uploader for user input
uploaded_file = st.file_uploader(
    "Unggah gambar daun jagung Anda di sini:",
    type=["jpg", "jpeg", "png"],
    help="Hanya file gambar JPG, JPEG, atau PNG yang diizinkan."
)

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Gambar yang Diunggah', use_column_width=True)
    st.write("")
    st.write("---")
    st.write("Menganalisis gambar...")

    # Create temporary file path
    temp_image_path = os.path.join(MODEL_DIR, "uploaded_temp_image.jpg")
    with open(temp_image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Placeholder for prediction status
    prediction_status_placeholder = st.empty()
    prediction_status_placeholder.info("Melakukan prediksi...")

    try:
        # Preprocess image using the helper function for EfficientNetB0
        processed_image = preprocess_image_for_model(temp_image_path, "efficientnet")
        
        # Debugging: Display processed image shape
        st.write(f"DEBUG: Memproses EfficientNetB0 dengan shape: {processed_image.shape}") 
        
        # Predict with the EfficientNetB0 model
        prediction = efficientnet_model.predict(processed_image)
        
        # Get predicted class index and confidence level
        predicted_class_index = np.argmax(prediction)
        result = CLASS_NAMES[predicted_class_index]
        confidence = np.max(prediction) * 100 # Convert to percentage

        prediction_status_placeholder.success("✅ Prediksi Selesai!")
        st.write("---")
        st.subheader(f"Hasil Deteksi: **{result}**")
        st.write(f"Tingkat Keyakinan: **{confidence:.2f}%**")
        st.markdown("---")

        # --- Get and display treatment suggestions from Gemini ---
        st.subheader("Saran Perawatan dari AI:")
        if result == "Healthy":
            st.info("Daun jagung terlihat sehat. Pertahankan praktik perawatan yang baik!")
        else:
            treatment_suggestions = get_treatment_suggestions(result)
            st.write(treatment_suggestions)
        st.markdown("---")
        
    except Exception as e:
        prediction_status_placeholder.error("❌ Terjadi kesalahan saat memproses gambar atau melakukan prediksi.")
        st.exception(e) # Display full exception for debugging
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
