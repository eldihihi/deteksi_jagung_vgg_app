# main_streamlit_app.py (Untuk Model VGG16 Saja dengan Gemini API)

import streamlit as st
import os
import numpy as np
import gdown # Digunakan untuk mengunduh model dari Google Drive
from tensorflow.keras.models import load_model # Untuk memuat model Keras/TensorFlow
from PIL import Image # Digunakan untuk manipulasi gambar (misalnya, mengubah ukuran)
import requests # Digunakan untuk membuat permintaan HTTP ke Gemini API
import json # Digunakan untuk mengurai respons JSON dari Gemini API

# Impor fungsi pra-pemrosesan spesifik untuk VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess

# --- Konfigurasi Model ---
# Direktori untuk menyimpan model yang diunduh
MODEL_DIR = "models"

# Google Drive URL untuk model VGG16 Anda (menggunakan 'uc?id=' format untuk gdown)
# PENTING: Pastikan URL ini benar dan file dapat diakses publik
MODEL_URLS = {
    "vgg": "https://drive.google.com/uc?id=1kKUN75slUQtEsqv8tULqAC-HIVy_OBU8" # URL VGG16
}

# Nama file lokal untuk model
MODEL_FILENAMES = {
    "vgg": "vgg_best_model.h5"
}

# Ukuran gambar input yang dibutuhkan oleh model VGG16
IMAGE_TARGET_SIZES = {
    "vgg": (224, 224) # VGG16 biasanya membutuhkan 224x224
}

# Nama kelas untuk hasil prediksi
CLASS_NAMES = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']

# --- Gemini API Configuration ---
# API Key akan disediakan oleh lingkungan Canvas saat runtime (__initial_auth_token)
# Di Streamlit Cloud, Anda perlu menambahkan ini sebagai Secret.
# Untuk pengujian lokal, Anda bisa masukkan API Key Gemini Anda di sini,
# atau lebih baik, gunakan st.secrets jika sudah di-deploy.
# Contoh: GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
# Untuk deployment di Canvas, biarkan kosong, Canvas akan mengisinya.
GEMINI_API_KEY = "" # Ini akan diisi otomatis oleh Canvas
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# --- Helper Function for Image Preprocessing ---
def preprocess_image_for_model(image_path, model_name):
    """
    Memuat dan pra-memproses gambar sesuai dengan persyaratan model VGG16.
    """
    target_size = IMAGE_TARGET_SIZES.get(model_name)
    if not target_size:
        raise ValueError(f"Ukuran gambar tidak dikenal untuk model: {model_name}")

    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0) # Tambahkan dimensi batch

    # Terapkan pra-pemrosesan spesifik VGG16
    if model_name == "vgg":
        return vgg_preprocess(img_array)
    else:
        # Fallback (seharusnya tidak terpakai jika hanya VGG)
        return img_array / 255.0

# --- Model Loading with Streamlit Caching ---
# @st.cache_resource sangat penting di sini: memastikan model diunduh dan dimuat
# hanya sekali di seluruh sesi pengguna, mencegah operasi berat berulang.
# hash_funcs ditambahkan untuk memaksa cache di-invalidate jika ada perubahan internal.
@st.cache_resource(show_spinner=False, hash_funcs={"_thread.RLock": lambda _: None})
def load_vgg_model_cached():
    """
    Mengunduh dan memuat model VGG16. Fungsi ini di-cache oleh Streamlit.
    """
    loaded_model = None
    
    # Buat direktori model jika belum ada
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    st.markdown("---")
    st.write("Memulai proses memuat model VGG16 di latar belakang...")
    st.write("Ini mungkin memakan waktu beberapa menit, terutama pada deployment pertama.")
    
    # Gunakan placeholder untuk pesan dinamis
    status_placeholder = st.empty()

    name = "vgg"
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
            
    status_placeholder.success("🎉 Model VGG16 berhasil dimuat dan siap digunakan!")
    st.markdown("---")
    return loaded_model

# --- Function to get treatment suggestions from Gemini API ---
def get_treatment_suggestions(plant_disease):
    """
    Calls the Gemini API to get treatment suggestions for a given plant disease.
    """
    # In Streamlit Cloud, the API key is typically managed via st.secrets
    # For local testing, you might need to set an environment variable or hardcode it (not recommended for production)
    # For Canvas environment, __initial_auth_token will be used to get the API key.
    # We will assume the API key is available via a global variable or st.secrets
    
    # Check if API key is available from st.secrets (typical for Streamlit Cloud deployment)
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    if not api_key:
        # Fallback for Canvas environment where __initial_auth_token is provided
        # This part is specific to Canvas and might not work in generic Streamlit Cloud
        if 'initial_auth_token' in st.session_state and st.session_state.initial_auth_token:
            # In a real Canvas environment, this token would be used to authenticate
            # and then fetch the actual API key. For simplicity here, we'll just
            # use a placeholder or assume it's handled by the environment.
            # For this specific scenario (calling generateContent), the API key
            # is typically passed directly as a query parameter.
            # The Canvas environment will inject the API key for the fetch call.
            pass # API key will be injected by Canvas for the fetch call
        else:
            st.error("API Key untuk Gemini tidak ditemukan. Harap pastikan GEMINI_API_KEY diatur di Streamlit Secrets atau __initial_auth_token tersedia.")
            return "Tidak dapat memberikan saran perawatan tanpa API Key."

    prompt = f"Berikan saran perawatan singkat dan praktis untuk penyakit daun jagung: {plant_disease}. Fokus pada langkah-langkah yang bisa dilakukan petani. Berikan dalam bentuk poin-poin singkat."
    
    chat_history = []
    chat_history.append({"role": "user", "parts": [{"text": prompt}]})
    
    payload = {"contents": chat_history}
    
    try:
        with st.spinner(f"Mencari saran perawatan untuk {plant_disease} dari AI..."):
            # The API key is added as a query parameter.
            # In Canvas, the API key will be automatically provided if GEMINI_API_KEY is empty.
            api_url_with_key = f"{GEMINI_API_URL}?key={api_key}" if api_key else GEMINI_API_URL
            
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
    page_title="Deteksi Penyakit Daun Jagung (VGG16 + Gemini AI)",
    page_icon="🌽",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Application Title and Description
st.title("🌽 Deteksi Penyakit Daun Jagung")
st.markdown("""
Aplikasi ini menggunakan model Convolutional Neural Network (CNN) **VGG16**
untuk mendeteksi penyakit pada daun jagung. Setelah deteksi, aplikasi akan memberikan saran perawatan
menggunakan **Google Gemini AI**.
""")

# Call the cached model loading function
vgg_model = load_vgg_model_cached()

# Check if model was loaded successfully
if vgg_model is None:
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
        # Preprocess image using the helper function for VGG
        processed_image = preprocess_image_for_model(temp_image_path, "vgg")
        
        # Debugging: Display processed image shape
        st.write(f"DEBUG: Memproses VGG dengan shape: {processed_image.shape}") 
        
        # Predict with the VGG model
        prediction = vgg_model.predict(processed_image)
        
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
