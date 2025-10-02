import streamlit as st
import pandas as pd
import joblib

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Kualitas Jeruk",
    page_icon="🍊",
    layout="centered"
)

# Load model
@st.cache_resource
def load_model():
    return joblib.load('model_klasifikasi_jeruk.joblib')

model = load_model()

# Header
st.title("🍊 Prediksi Kualitas Jeruk")
st.markdown("Aplikasi untuk memprediksi kualitas jeruk menjadi **Bagus, Sedang, atau Jelek**")

# Input form dalam columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("Karakteristik Fisik")
    diameter = st.slider("Diameter (cm)", 5.0, 10.0, 7.5, 0.1)
    berat = st.slider("Berat (gram)", 80.0, 250.0, 180.0, 1.0)
    tebal_kulit = st.slider("Tebal Kulit (cm)", 0.1, 1.5, 0.5, 0.01)
    kadar_gula = st.slider("Kadar Gula (%)", 5.0, 20.0, 12.0, 0.1)

with col2:
    st.subheader("Karakteristik Lainnya")
    
    # Menggunakan st.pills untuk single selection
    asal_daerah = st.pills(
        "Pilih Asal Daerah:",
        ["Jawa Barat", "Jawa Tengah", "Kalimantan"],
        selection_mode="single"
    )
    
    warna = st.pills(
        "Pilih Warna Kulit:",
        ["hijau", "kuning", "oranye"],
        selection_mode="single"
    )
    
    musim_panen = st.pills(
        "Pilih Musim Panen:",
        ["kemarau", "hujan"],
        selection_mode="single"
    )

# Tombol prediksi
if st.button("Prediksi Kualitas", type="primary", use_container_width=True):
    # Validasi input
    if not asal_daerah or not warna or not musim_panen:
        st.error("⚠️ Harap pilih semua opsi yang tersedia!")
    else:
        # Buat dataframe input
        input_data = pd.DataFrame({
            'diameter': [diameter],
            'berat': [berat],
            'tebal_kulit': [tebal_kulit],
            'kadar_gula': [kadar_gula],
            'asal_daerah': [asal_daerah],
            'warna': [warna],
            'musim_panen': [musim_panen]
        })
        
        # Prediksi
        try:
            prediction = model.predict(input_data)[0]
            probabilities = model.predict_proba(input_data)[0]
            
            # Tampilkan hasil utama
            if prediction == "Bagus":
                st.success(f"## Hasil Prediksi: **{prediction}** 🟢")
            elif prediction == "Sedang":
                st.warning(f"## Hasil Prediksi: **{prediction}** 🟡")
            else:
                st.error(f"## Hasil Prediksi: **{prediction}** 🔴")
            
            # Probabilitas
            st.subheader("Probabilitas Kualitas:")
            quality_labels = model.named_steps['model'].classes_
            
            for quality, prob in zip(quality_labels, probabilities):
                col_proba, col_bar = st.columns([1, 3])
                with col_proba:
                    # Beri warna berdasarkan kualitas
                    if quality == "Bagus":
                        st.markdown(f"🟢 **{quality}**: {prob*100:.2f}%")
                    elif quality == "Sedang":
                        st.markdown(f"🟡 **{quality}**: {prob*100:.2f}%")
                    else:
                        st.markdown(f"🔴 **{quality}**: {prob*100:.2f}%")
                with col_bar:
                    st.progress(float(prob))
            
            # Rekomendasi
            st.subheader("💡 Rekomendasi:")
            if prediction == "Bagus":
                st.success("**Kualitas Ekspor** - Harga premium, cocok untuk pasar internasional")
            elif prediction == "Sedang":
                st.warning("**Kualitas Lokal** - Harga standar, cocok untuk pasar domestik")
            else:
                st.error("**Kualitas Industri** - Disarankan untuk olahan seperti jus atau selai")
                
            st.balloons()
            
        except Exception as e:
            st.error(f"Terjadi error dalam prediksi: {e}")

# Footer
st.divider()
st.caption("Dibuat dengan 🍊 oleh Al Zaki Ibra Ramadani | Model Accuracy: 98.6%")
