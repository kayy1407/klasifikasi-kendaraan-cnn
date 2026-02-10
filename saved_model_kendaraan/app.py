import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

st.set_page_config(
    page_title="Klasifikasi Jenis Kendaraan",
    page_icon="🚗",
    layout="centered"
)

CLASS_NAMES = [
    'Auto Rickshaws',
    'Bikes',
    'Cars',
    'Motorcycles',
    'Planes',
    'Ships',
    'Trains'
]

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("saved_model_kendaraan")
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None


def import_and_predict(image_data, model):

    size = (128, 128)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)

    img = np.asarray(image)

    if img.shape[-1] == 4:
        img = img[..., :3]

    img = img / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)

    # ===== SAVEDMODEL FIX =====
    infer = model.signatures["serve"]
    prediction = infer(tf.constant(img))

    prediction = list(prediction.values())[0].numpy()

    return prediction


def main():
    st.title("🚗 Klasifikasi Jenis Kendaraan")
    st.write("Upload gambar kendaraan untuk mengetahui jenisnya.")
    st.write(f"Model dapat mengenali: {', '.join(CLASS_NAMES)}")

    with st.spinner("Memuat model..."):
        model = load_model()

    if model is None:
        return

    file = st.file_uploader("Pilih gambar", type=["jpg","png","jpeg"])

    if file:
        image = Image.open(file)
        st.image(image, caption="Gambar yang diupload", width="stretch")

        if st.button("Prediksi"):
            with st.spinner("Memprediksi..."):
                prediction = import_and_predict(image, model)
                class_index = np.argmax(prediction)
                st.success(f"Hasil Prediksi: **{CLASS_NAMES[class_index]}**")


if __name__ == "__main__":
    main()
