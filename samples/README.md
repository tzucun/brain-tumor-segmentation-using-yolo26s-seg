# Sampel Gambar Uji

Folder ini berisi citra MRI contoh dari dataset test untuk mencoba aplikasi Streamlit.

## Citra untuk Uji Fungsi

Gunakan salah satu file di `samples/test_images/`:

- `glioma_sample.jpg`
- `meningioma_sample.jpg`
- `pituitary_sample.jpg`
- `no_tumor_sample.jpg`

File `no_tumor_sample.jpg` digunakan untuk menguji kasus negative sample. `no_tumor` bukan kelas prediksi model.

Unggah salah satu berkas tersebut melalui aplikasi. Kedua model menerima citra
dan confidence threshold yang sama. Tidak ada pilihan sampel langsung di antarmuka.

Keempat citra ini hanya untuk memeriksa fungsi program, bukan pengganti evaluasi
pada dataset penelitian lengkap. Tidak ada arsip dataset atau label segmentasi
yang disertakan dalam folder ini di repositori.
