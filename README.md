# 📊Laporan Proyek Machine Learning Terapan - Maylina Nur'aini

## Domain Proyek

Kemajuan pendidikan menuntut strategi pengajaran yang berbasis data. Salah satu tantangan utamanya adalah memprediksi performa akademik siswa berdasarkan faktor-faktor latar belakang seperti jenis kelamin, etnis, pendidikan orang tua, jenis makan siang, dan kursus persiapan ujian. Proyek ini menggunakan dataset dari Kaggle berisi data 1000 siswa untuk membangun model prediktif yang mampu memperkirakan hasil ujian matematika secara akurat. Hasil prediksi ini dapat dimanfaatkan untuk mendukung strategi pembelajaran dan kebijakan pendidikan yang lebih efektif. 🎓

---

## Business Understanding

### Problem Statements

Masalah pada proyek ini :

1. Bagaimana data akademik dan latar belakang siswa dapat dimanfaatkan untuk mengembangkan sistem prediktif skor matematika sebagai dasar pengambilan keputusan strategis di lingkungan pendidikan?
2. Bagaimana memilih model prediksi terbaik untuk memperkirakan nilai matematika siswa berdasarkan perbandingan performa beberapa algoritma?

### Goal

Tujuan pada proyek ini :

1. Merancang sebuah model prediksi yang mampu mengestimasi nilai matematika siswa secara akurat, sehingga sekolah dapat memperoleh gambaran potensi siswa sejak awal dan melakukan tindak lanjut yang sesuai.
2. Menganalisis dan membandingkan kinerja dari model KNN, Boosting, dan Random Forest untuk menentukan model dengan akurasi tertinggi yang paling sesuai diterapkan di bidang pendidikan.

### Solution statements

1. Mengembangkan model regresi prediktif yang mampu memperkirakan skor ujian matematika siswa berdasarkan fitur-fitur latar belakang seperti jenis kelamin, etnis, tingkat pendidikan orang tua, jenis makan siang, dan kursus persiapan ujian.

2. Menerapkan teknik preprocessing data termasuk penanganan data duplikat, outlier, dan encoding variabel kategorikal untuk memastikan data berkualitas tinggi sebelum pelatihan model.

3. Membandingkan performa beberapa algoritma machine learning (seperti KNN, Random Forest, dan Boosting) berdasarkan metrik Mean Squared Error (MSE) untuk menentukan model terbaik.

---

## Data Understanding

- **Sumber Data:** [Kaggle - Students Performance in Exams] (https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)
- **Jumlah Data:** 1000 baris, 7 kolom
- **Kondisi Data:**
  - Tidak ada missing value
  - Tidak terdapat data duplikat
  - Terdeteksi outlier pada kolom 'math score', 'reading score', 'writing score'.
 
### Variabel

- **Variabel Numerik:**
  - Math score: Skor ujian matematika siswa (0–100).
  - Reading score: Skor ujian membaca siswa (0–100).
  - Writing score: Skor ujian menulis siswa (0–100).
    
- **Variabel Kategorikal:**
  - Gender: Jenis kelamin siswa (male atau female).
  - Race/ethnicity: Kelompok etnis siswa (Group A, B, C, D, atau E).
  - Parental level of education: Tingkat pendidikan tertinggi orang tua (misalnya bachelor's degree, some college, master's degree, dll.).
  - Lunch: Jenis makan siang yang diterima siswa (standard atau free/reduced).
  - Test preparation course: Status partisipasi dalam kursus persiapan ujian (none atau completed).

### Exploratory Data Analysis (EDA):

#### 1. Univariate Analysis:

- **Fitur Kategorikal**:
  - Distribusi fitur kategorikal ini menunjukkan female (51,8%) dan male (48,2)
  - Female lebih dominan dibandingkan dengan male.

- **Fitur Numerik**:
  - Math score: Distribusi cenderung normal, puncaknya di sekitar nilai tengah. Ada beberapa siswa dengan skor rendah.
  - Reading score: Distribusi juga cenderung normal, dengan puncak di sekitar nilai tengah. Terlihat lebih sedikit siswa dengan skor sangat rendah dibandingkan math score.
  - Writing score: Distribusi mirip dengan reading score, cenderung normal dengan puncak di sekitar nilai tengah.

#### 2. Multivariate Analysis:

- **Fitur Kategorikal**:
  - Gender: Rata-rata math score untuk female sekitar 63-64, sedangkan untuk male sekitar 68-69. Jadi, laki-laki memang memiliki rata-rata skor matematika yang sedikit lebih tinggi.
  - Race/ethnicity: Visualisasi menunjukkan rata-rata skor bervariasi antar grup. Untuk math score, Grup E tampaknya memiliki rata-rata tertinggi, diikuti oleh Grup D, C, B, dan A. Jadi, pernyataan "Grup C adalah yang paling banyak (~33%)" mengacu pada jumlah sampel, bukan rata-rata skor. Untuk rata-rata skor matematika, Grup E yang tertinggi.
  - Parental level of education: Rata-rata skor (math, reading, writing) cenderung meningkat seiring dengan tingkat pendidikan orang tua yang lebih tinggi, dengan "master's degree" dan "bachelor's degree" memiliki rata-rata skor yang lebih tinggi daripada "some high school" atau "high school". Jadi, "Mayoritas orang tua memiliki pendidikan hingga "some college"" adalah tentang jumlah sampel, bukan rata-rata skor.
  - Lunch: Siswa yang mendapatkan "standard lunch" memiliki rata-rata skor (math, reading, writing) yang lebih tinggi dibandingkan dengan siswa yang mendapatkan "free/reduced lunch". Pernyataan "Sebagian besar siswa mendapatkan standard lunch (~65%)" mengacu pada jumlah sampel.
  - Test preparation course: Siswa yang menyelesaikan "test preparation course" memiliki rata-rata skor (math, reading, writing) yang lebih tinggi dibandingkan dengan yang tidak. Pernyataan "Sebagian besar siswa tidak mengikuti kursus persiapan (~65%)" mengacu pada jumlah sampel.

- **Fitur Numerik**:
  - Pairplot dan Heatmap menunjukkan korelasi positif yang kuat antara math score, reading score, dan writing score. Ini berarti siswa yang berkinerja baik di satu mata pelajaran cenderung berkinerja baik di mata pelajaran lain.
 
---

## Data Preparation

- Data diperiksa untuk menghindari adanya duplikat, meskipun dataset asli tidak mengandung duplikasi.
- Outlier pada fitur numerik seperti math score, reading score, dan writing score ditangani menggunakan metode IQR (Interquartile Range).
- Fitur kategorikal seperti gender, race/ethnicity, parental level of education, lunch, dan test preparation course dikonversi menjadi fitur numerik menggunakan one-hot encoding.
- Dataset dibagi menjadi data latih dan data uji dengan proporsi 80%:20% menggunakan fungsi train_test_split.
- Fitur numerik (reading score, writing score) distandarisasi menggunakan StandardScaler, terutama karena target prediksi adalah math score.

## Modeling

Pada bagian ini, mengeksplorasi berbagai model regresi untuk memprediksi kinerja siswa. Untuk melatih dan mengevaluasi tiga model: K-Nearest Neighbors Regressor, Random Forest Regressor, dan AdaBoost Regressor.

### K-Nearest Neighbors Regressor (KNN)

K-Nearest Neighbors (KNN) Regressor adalah metode non-parametrik yang digunakan untuk regresi. Model ini memprediksi nilai untuk titik data baru dengan merata-ratakan nilai dari 'k' tetangga terdekatnya di dataset pelatihan. Model ini relatif mudah dipahami dan diimplementasikan, tetapi bisa memakan banyak komputasi untuk dataset yang besar.

### Random Forest Regressor

Random Forest Regressor adalah metode pembelajaran ansambel yang membangun banyak pohon keputusan selama pelatihan dan menghasilkan prediksi rata-rata dari masing-masing pohon. Pendekatan ini membantu mengurangi overfitting dan meningkatkan ketahanan model. Random Forest dikenal karena performanya yang baik dan kemampuannya menangani hubungan kompleks dalam data.

### AdaBoost Regressor

AdaBoost (Adaptive Boosting) Regressor adalah teknik ansambel lainnya. Ini bekerja dengan secara berurutan membangun serangkaian regressor yang lemah (seringkali pohon keputusan) dan memberikan bobot lebih pada titik data yang salah diprediksi pada langkah-langkah sebelumnya. Proses iteratif ini berfokus pada sampel yang sulit diprediksi, menghasilkan model keseluruhan yang lebih kuat. AdaBoost efektif dalam mengurangi bias dan meningkatkan akurasi dari pembelajar yang lemah (weak learners).

---

## Evaluation

### Metrik Evaluasi

Model dievaluasi menggunakan Mean Squared Error (MSE), metrik regresi yang mengukur rata-rata kuadrat dari selisih antara nilai aktual dan nilai prediksi. Semakin kecil nilai MSE, semakin baik performa model dalam melakukan prediksi.

### Hasil Evaluasi Berdasarkan MSE:

| Model         | MSE (Train) | MSE (Test) |
| ------------- | ----------- | ---------- |
| KNN           | 280.5       | 420.7      |
| Random Forest | 180.2       | 210.4      |
| Boosting      | 190.8       | 230.6      |

📌 Random Forest memberikan performa terbaik pada data uji dengan MSE terkecil, menandakan kemampuan generalisasi yang baik.

### Hasil Evaluasi Berdasarkan Data Aktual:

| y\_true | Prediksi KNN | Prediksi RF | Prediksi Boosting |
| ------- | ------------ | ----------- | ----------------- |
| 66      | 70.7         | 69.6        | 74.9              |

💡 Dari sampel ini terlihat bahwa semua model memberikan prediksi yang cukup dekat dengan nilai sebenarnya, namun Random Forest cenderung menghasilkan prediksi yang paling stabil dan mendekati y_true.
