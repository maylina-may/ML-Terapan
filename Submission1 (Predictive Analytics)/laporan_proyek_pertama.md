# Laporan Proyek Pertama Machine Learning Terapan - Maylina Nur'aini

## Domain Proyek 
Pendidikan memiliki peran penting dalam membentuk masa depan individu dan masyarakat. Evaluasi terhadap performa siswa menjadi aspek krusial dalam mengidentifikasi kebutuhan belajar serta merancang strategi pengajaran yang lebih efektif. Penilaian yang berbasis data memungkinkan pendidik memahami pola performa akademik siswa secara lebih objektif.

Dataset Students Performance in Exams digunakan dalam proyek ini untuk mengkaji pengaruh berbagai faktor seperti jenis kelamin, etnis, tingkat pendidikan orang tua, jenis makan siang, serta partisipasi dalam kursus persiapan ujian terhadap hasil ujian siswa. Pemilihan domain pendidikan didasarkan pada urgensi memahami faktor-faktor penentu keberhasilan akademik dalam rangka meningkatkan kualitas pendidikan.

Masalah ini perlu diselesaikan karena temuan dari model prediksi dapat digunakan untuk memberikan rekomendasi yang lebih tepat kepada siswa, guru, dan pemangku kebijakan. Model machine learning mampu mengidentifikasi pola tersembunyi dalam data yang tidak mudah terlihat melalui analisis manual.

Studi terdahulu menunjukkan bahwa latar belakang sosial, ekonomi, dan demografis memberikan kontribusi yang signifikan terhadap performa akademik siswa [1][2]. Oleh karena itu, pemanfaatan machine learning dalam klasifikasi performa siswa menjadi pendekatan yang relevan dan potensial dalam ranah pendidikan.

## Business Understanding
### Problem Statements
Menjelaskan pernyataan masalah latar belakang :
- Pernyataan Masalah 1 : Bagaimana pengaruh faktor demografis seperti jenis kelamin, etnis, dan tingkat pendidikan orang tua terhadap performa akademik siswa dalam ujian?
- Pernyataan Masalah 2 : Apakah jenis makan siang yang dikonsumsi siswa berhubungan dengan hasil ujian mereka?
- Pernyataan Masalah 3 : Sejauh mana partisipasi siswa dalam kursus persiapan ujian dapat meningkatkan nilai ujian mereka?
- Pernyataan Masalah 4 : Bagaimana membangun model machine learning yang dapat memprediksi performa siswa berdasarkan faktor-faktor tersebut?

### Goals
Menjelaskan tujuan dari pernyataan masalah :
- Jawaban pernyataan masalah 1 : Mengidentifikasi faktor-faktor demografis yang signifikan memengaruhi performa ujian siswa untuk membantu pendidik dan pemangku kebijakan dalam merancang intervensi yang tepat.
- Jawaban pernyataan masalah 2 : Menganalisis hubungan antara jenis makan siang dan hasil ujian untuk memahami dampak nutrisi terhadap performa akademik.
- Jawaban pernyataan masalah 3 : Mengevaluasi efektivitas kursus persiapan ujian dalam meningkatkan nilai siswa.
- Jawaban pernyataan masalah 4 : Membangun dan mengembangkan model prediktif dengan akurasi yang baik untuk mengklasifikasikan performa siswa, sehingga dapat digunakan sebagai alat bantu pengambilan keputusan pendidikan.

    ### Solution statements
    - Membangun model klasifikasi untuk memprediksi performa ujian menggunakan algoritma seperti Logistic Regression dan Random Forest.
    - Melakukan hyperparameter tuning untuk meningkatkan akurasi model.

## Data Understanding
Dataset : https://www.kaggle.com/datasets/spscientist/students-performance-in-exams  

### Variabel-variabel pada dataset Students Performance in Exams adalah sebagai berikut :
- Gender : merupakan jenis kelamin siswa dengan nilai 'male' atau 'female'
- Race/ethnicity : merupakan kelompok etnis siswa berdasarkan kelompok sosial atau rasialnya dengan nilai 'group A', 'group B', 'group C', 'group D', 'group E'.
- Parental level of education : merupakan tingkat pendidikan tertinggi yang dicapai oleh orang tua siswa dengan contoh nilai 'some high school', 'high school', 'some college', 'associate's degree', 'bachelor's degree', 'master's degree'.
- lunch : merupakan jenis makan siang yang dikonsumsi siswa saat sekolah dengan nilai 'standard' atau 'free/reduced'.
- Test preparation course : merupakan status partisipasi siswa dalam kursus persiapan ujian dengan nilai 'none' atau 'completed'.
- Math score : merupakan nilai ujian siswa dalam mata pelajaran Matematika dengan tipe numerik (0–100).
- Reading score : merupakan nilai ujian siswa dalam mata pelajaran Membaca dengan tipe numerik (0-100).
- Writing score : nilai ujian siswa dalam mata pelajaran Menulis denhgan tipe numerik (0–100).

Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data atau exploratory data analysis.
Berdasarkan kode yang sudah dibuat, tahapan Exploratory Data Analysis (EDA) yang telah dilakukan meliputi :

- Memahami Struktur Data : Melakukan pengecekan informasi dataset (data.info()) untuk melihat jumlah entri, kolom, tipe data, dan keberadaan nilai non-null.
- Analisis Statistik Deskriptif : Melihat ringkasan statistik deskriptif (data.describe()) untuk fitur numerik ('math score', 'reading score', 'writing score') untuk mendapatkan gambaran sebaran nilai.
- Visualisasi Data : Membuat visualisasi untuk memahami distribusi data, seperti:
    - Countplot untuk melihat distribusi gender dalam dataset.
    - Histogram untuk melihat distribusi nilai matematika siswa.

## Data Preparation
Pada bagian ini, menerapkan teknik-teknik persiapan data yang penting sebelum data digunakan untuk melatih model machine learning. Langkah-langkah ini memastikan data dalam format yang tepat dan berkualitas untuk analisis.
1. Pengecekan Missing Values
Langkah pertama dalam persiapan data adalah memeriksa apakah ada nilai yang hilang (missing values) dalam dataset. Nilai yang hilang dapat mempengaruhi kinerja model, oleh karena itu perlu diidentifikasi dan ditangani.

Alasan : Missing values dapat menyebabkan bias dalam analisis dan pelatihan model. Model machine learning umumnya tidak dapat menangani nilai yang hilang secara langsung.
```
# Cek missing value
print(data.isnull().sum())
```
2. Pembuatan Label Klasifikasi
Untuk tugas klasifikasi ini, perlu mendefinisikan label atau target variabel. Berdasarkan tujuan proyek, akan membuat label biner yang menunjukkan apakah siswa "lulus" atau "tidak lulus" dalam mata pelajaran matematika berdasarkan skor matematika mereka.

Alasan : Model klasifikasi memerlukan target variabel diskrit. Dalam kasus ini, kita mengklasifikasikan siswa berdasarkan pencapaian skor matematika mereka (> = 70 sebagai lulus).
```
# Buat label klasifikasi: apakah nilai matematika >= 70 (1 = lulus, 0 = tidak)
data['pass_math'] = np.where(data['math score'] >= 70, 1, 0)
```
3. Encoding Fitur Kategorikal
Dataset ini berisi fitur kategorikal seperti 'gender', 'race/ethnicity', 'parental level of education', 'lunch', dan 'test preparation course'. Model machine learning sebagian besar bekerja dengan data numerik, sehingga fitur-fitur ini perlu diubah menjadi representasi numerik.

Alasan : Model machine learning tidak dapat memproses data dalam bentuk teks atau kategori secara langsung. Encoding mengubah fitur kategorikal menjadi format numerik yang dapat dipahami oleh model. Kami menggunakan *LabelEncoder* dan *pd.factorize* untuk melakukan encoding ini. *LabelEncoder* cocok untuk fitur dengan dua kategori, sementara *pd.factorize* berguna untuk fitur dengan lebih dari dua kategori, mengubahnya menjadi representasi numerik berbasis integer.
```
# Encode fitur kategori
le = LabelEncoder()
data['gender'] = le.fit_transform(data['gender'])
data['lunch'] = le.fit_transform(data['lunch'])
data['test preparation course'] = le.fit_transform(data['test preparation course'])
data['race/ethnicity'] = pd.factorize(data['race/ethnicity'])[0]
data['parental level of education'] = pd.factorize(data['parental level of education'])[0]

data.head()
```

## Modeling
Pada tahap ini, akan membangun dan melatih model machine learning untuk memprediksi apakah seorang siswa akan lulus mata pelajaran matematika berdasarkan fitur-fitur yang ada. Berdasarkan pendekatan klasifikasi yang dipilih, ini akan menggunakan dua algoritma yang umum digunakan : Logistic Regression dan Random Forest.

1. Pemilihan Model, memilih dua algoritma klasifikasi untuk membandingkan kinerja mereka dalam menyelesaikan masalah ini :
- Logistic Regression : Algoritma linier yang sederhana namun efektif untuk masalah klasifikasi biner. Model ini memprediksi probabilitas suatu kelas (dalam kasus ini, probabilitas siswa lulus) berdasarkan kombinasi linier fitur-fitur input.
     
Kelebihan: Cepat, mudah diinterpretasikan, baik untuk dataset yang linear separable, membutuhkan sumber daya komputasi yang relatif sedikit. Kekurangan: Kurang efektif pada data yang kompleks atau non-linier, sensitif terhadap outlier, mengasumsikan linearitas antara fitur dan log-odds.

- Random Forest : Algoritma ensemble yang membangun banyak pohon keputusan dan menggabungkan prediksi mereka. Ini adalah model yang kuat dan serbaguna yang dapat menangani hubungan non-linier dan interaksi fitur.

Kelebihan : Umumnya memberikan akurasi tinggi, kurang rentan terhadap overfitting dibandingkan pohon keputusan tunggal, dapat menangani data dengan fitur yang banyak dan interaksi yang kompleks, memberikan estimasi pentingnya fitur. Kekurangan: Kurang interpretable dibandingkan Logistic Regression, membutuhkan lebih banyak komputasi dan memori, bisa lambat untuk dataset yang sangat besar, bias terhadap fitur dengan banyak kategori.
2. Pelatihan Model
Setelah memilih algoritma adalah melatih masing-masing model menggunakan data latih (X_train dan y_train) yang telah disiapkan pada tahap sebelumnya.
```
# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
```
- Untuk Logistic Regression, menggunakan parameter *max_iter=1000*. Ini mengatur jumlah maksimum iterasi untuk konvergensi algoritma. Nilai yang lebih tinggi diberikan untuk memastikan model memiliki cukup iterasi untuk menemukan solusi optimal, terutama jika dataset cukup besar atau kompleks.
- Untuk Random Forest, menggunakan parameter *random_state=42*. Ini memastikan bahwa pembentukan pohon keputusan dalam forest adalah deterministik dan dapat direproduksi setiap kali kode dijalankan dengan parameter yang sama.
3. Pemilihan Model Terbaik
Berdasarkan hasil evaluasi yang telah dilakukan (pada bagian selanjutnya di notebook Anda), kita dapat membandingkan kinerja kedua model menggunakan metrik akurasi :
- Akurasi Logistic Regression: 0.90
- Akurasi Random Forest: 0.86

Meskipun kedua model menunjukkan kinerja yang baik, **model Logistic Regression memiliki akurasi yang sedikit lebih tinggi (0.90)** pada data uji dibandingkan dengan Random Forest (0.86). Akurasi yang lebih tinggi ini menunjukkan bahwa Logistic Regression mampu mengklasifikasikan siswa yang lulus dan tidak lulus dengan lebih tepat pada dataset ini.

Oleh karena itu, berdasarkan metrik akurasi yang digunakan, **model Logistic Regression dipilih sebagai model terbaik** untuk kasus ini. Selain akurasi, Logistic Regression juga menawarkan interpretasi yang lebih mudah dibandingkan Random Forest, yang bisa menjadi keuntungan dalam memahami faktor-faktor apa yang paling berkontribusi terhadap kelulusan siswa dalam mata pelajaran matematika.

## Evaluation
Setelah melatih model, langkah selanjutnya adalah mengevaluasi seberapa baik kinerja model dalam memprediksi label pada data yang belum pernah dilihat sebelumnya (data uji).

Metrik Evaluasi : Untuk proyek klasifikasi ini, metrik evaluasi yang digunakan adalah **Akurasi, Confusion Matrix, dan Classification Report** (yang mencakup Precision, Recall, dan F1-Score). Metrik-metrik ini penting untuk memahami berbagai aspek kinerja model, tidak hanya seberapa sering model benar secara keseluruhan, tetapi juga bagaimana performanya pada setiap kelas.
- Akurasi (Accuracy) : Mengukur proporsi total instance yang diklasifikasikan dengan benar oleh model.
    - Formula : (Jumlah prediksi benar) / (Total jumlah instance)
    - Interpretasi : Akurasi memberikan gambaran umum seberapa sering model membuat prediksi yang tepat.
- Confusion Matrix : Tabel yang merangkum kinerja model klasifikasi. Ini menunjukkan jumlah True Positives (TP), True Negatives (TN), False Positives (FP), dan False Negatives (FN).
    - Interpretasi : Confusion Matrix memberikan detail tentang jenis kesalahan yang dibuat model, membantu mengidentifikasi di mana model berkinerja baik dan di mana ia kesulitan.
- Classification Report : Menyediakan metrik evaluasi tambahan untuk setiap kelas, termasuk Precision, Recall, dan F1-Score.
    - Precision (Presisi) : Proporsi instance positif yang diprediksi dengan benar dari semua instance yang diprediksi sebagai positif.
          - Formula : TP / (TP + FP)
          - Interpretasi : Presisi tinggi menunjukkan bahwa model "tepat" ketika memprediksi kelas positif.
    - Recall (Sensitivitas) : Proporsi instance positif aktual yang diprediksi dengan benar oleh model.
          - Formula : TP / (TP + FN)
          - Interpretasi : Recall tinggi menunjukkan bahwa model dapat "menemukan" sebagian besar instance kelas positif yang sebenarnya ada.
    - F1-Score : Rata-rata harmonik dari Precision dan Recall.
          - Formula : 2 * (Precision * Recall) / (Precision + Recall)
          - Interpretasi : F1-score memberikan skor tunggal yang menyeimbangkan Precision dan Recall, berguna terutama pada dataset yang tidak seimbang.

Hasil Evaluasi : Mengevaluasi kedua model yang telah dilatih, Logistic Regression dan Random Forest, menggunakan metrik-metrik ini pada data uji.
```
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

print("Confusion Matrix Random Forest:")
print(confusion_matrix(y_test, y_pred_rf))

print("Classification Report Random Forest:")
print(classification_report(y_test, y_pred_rf))
```

Berdasarkan hasil eksekusi kode di atas :
- Akurasi :
      - Logistic Regression : 0.90
      - Random Forest: 0.86 Model **Logistic Regression menunjukkan akurasi yang lebih tinggi** dibandingkan Random Forest, artinya secara keseluruhan, Logistic Regression lebih sering membuat prediksi yang tepat pada data uji.
- Confusion Matrix Random Forest :
```
[[108  14]
     [ 14  64]]
```

```
- True Negatives (TN): 108 (siswa tidak lulus yang diprediksi tidak lulus)
- False Positives (FP): 14 (siswa tidak lulus yang diprediksi lulus)
- False Negatives (FN): 14 (siswa lulus yang diprediksi tidak lulus)
- True Positives (TP): 64 (siswa lulus yang diprediksi lulus)
Confusion Matrix memberikan gambaran detail tentang kesalahan prediksi Random Forest. Model ini cukup baik dalam mengidentifikasi siswa yang tidak lulus (TN tinggi), namun ada sejumlah kesalahan di kedua arah (FP dan FN).
```

- Classification Report Random Forest :
```
precision    recall  f1-score   support

               0       0.89      0.89      0.89       122
               1       0.82      0.82      0.82        78

        accuracy                           0.86       200
       macro avg       0.86      0.86      0.86       200
    weighted avg       0.86      0.86      0.86       200
```

```
- Untuk Kelas 0 (Tidak Lulus): Precision, Recall, dan F1-Score adalah 0.89. Ini menunjukkan bahwa model Random Forest cukup baik dalam memprediksi siswa yang tidak lulus, dengan sedikit false positive dan false negative untuk kelas ini.
- Untuk Kelas 1 (Lulus): Precision, Recall, dan F1-Score adalah 0.82. Metrik ini sedikit lebih rendah dibandingkan dengan Kelas 0. Ini berarti model Random Forest memiliki sedikit lebih banyak kesulitan dalam mengidentifikasi semua siswa yang benar-benar lulus (recall 0.82) dan terkadang salah memprediksi siswa sebagai lulus padahal sebenarnya tidak (precision 0.82).
- Akurasi keseluruhan Random Forest adalah 0.86, sesuai dengan hasil akurasi yang dilaporkan sebelumnya.
```

Berdasarkan perbandingan akurasi, **model Logistic Regression (akurasi 0.90)** menunjukkan kinerja yang lebih baik secara keseluruhan dalam memprediksi kelulusan siswa dalam mata pelajaran matematika pada data uji, dibandingkan dengan model Random Forest (akurasi 0.86).

## Referensi

[1] K. Sirin, "Socioeconomic Status and Academic Achievement: A Meta-Analytic Review of Research," Review of Educational Research, vol. 75, no. 3, pp. 417–453, 2005.

[2] D. Jeynes, "The Relationship Between Parental Involvement and Urban Secondary School Student Academic Achievement: A Meta-Analysis," Urban Education, vol. 42, no. 1, pp. 82–110, 2007.
