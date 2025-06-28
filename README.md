# Laporan Proyek Machine Learning Classification: Satria Dirgantara Nuryaman

## Domain Proyek 🌾

Pertanian merupakan sektor vital yang berperan besar dalam ketahanan pangan dan perekonomian, khususnya di negara agraris seperti Indonesia. Menurut [BPS 2023](https://www.bps.go.id/id/statistics-table/2/NjMjMg==/produk-domestik-bruto-menurut-lapangan-usaha.html), sektor pertanian menyumbang lebih dari 13% terhadap PDB nasional dan menjadi sumber penghidupan bagi jutaan keluarga.

Namun, petani sering menghadapi tantangan dalam menentukan jenis tanaman yang paling sesuai untuk ditanam pada lahan tertentu, mengingat banyaknya faktor yang memengaruhi seperti kandungan unsur hara tanah, suhu, kelembapan, pH tanah, dan curah hujan. Kesalahan dalam pemilihan jenis tanaman dapat menyebabkan penurunan produktivitas, kerugian ekonomi, dan ketidakseimbangan ekosistem ([FAO, 2021](https://www.fao.org/3/cb4476en/cb4476en.pdf)).

Seiring berkembangnya teknologi, pemanfaatan data dan kecerdasan buatan (Artificial Intelligence) menjadi solusi potensial untuk mengoptimalkan proses pengambilan keputusan di bidang pertanian. Dengan memanfaatkan data lingkungan dan karakteristik tanah, sistem rekomendasi berbasis machine learning dapat membantu petani memilih jenis tanaman yang paling optimal, sehingga meningkatkan hasil panen, efisiensi penggunaan lahan, dan keberlanjutan pertanian ([Sharma et al., 2020](https://ieeexplore.ieee.org/document/9121234)).

## Business Understanding 🗒️

### Problem Statements 🤖

Dalam dunia pertanian modern, petani dihadapkan pada tantangan untuk menentukan jenis tanaman yang paling sesuai dengan kondisi lahan dan lingkungan yang dimiliki. Berbagai faktor seperti kandungan Nitrogen, Phosphorous, Potassium, suhu, kelembapan, pH tanah, dan curah hujan, semuanya saling memengaruhi dan menentukan keberhasilan panen. Namun, kompleksitas hubungan antar faktor tersebut seringkali menyulitkan pengambilan keputusan secara manual.

Oleh karena itu, muncul pertanyaan pertanyaan penting yang ingin dijawab melalui proyek ini:
- Bagaimana pengaruh faktor lingkungan dan tanah terhadap pemilihan jenis tanaman yang optimal?
- Fitur apa yang paling dominan dalam menentukan rekomendasi tanaman untuk lahan tertentu?
- Jenis tanaman apa yang paling tepat direkomendasikan untuk kondisi lahan dan lingkungan tertentu?

### Goals 🏆

Untuk menjawab pertanyaan pertanyaan tersebut, proyek ini bertujuan membangun sebuah sistem rekomendasi berbasis machine learning yang mampu:
- Memberikan rekomendasi jenis tanaman secara akurat berdasarkan data lingkungan dan tanah yang tersedia.
- Mengidentifikasi fitur fitur kunci yang paling berpengaruh dalam proses pengambilan keputusan, sehingga dapat menjadi acuan bagi petani dalam mengelola lahan secara lebih efektif dan berkelanjutan.

### Solution Statements 💡

- Melakukan analisis statistik dan visualisasi data untuk menemukan pola, outlier, serta korelasi antara variabel lingkungan/tanah dengan jenis tanaman.
- Mengimplementasikan beberapa algoritma klasifikasi seperti KNeighborsClassifier, RandomForestClassifier, dan XGBoost untuk membangun model prediktif yang andal.
- Mengevaluasi performa model menggunakan metrik akurasi, precision, recall, F1-score, dan confusion matrix, guna memastikan rekomendasi yang dihasilkan benar benar dapat diandalkan.
- Melakukan hyperparameter tuning dengan GridSearchCV untuk mengoptimalkan performa model, sehingga solusi yang dihasilkan benar-benar siap diterapkan di lapangan.

## Data Understanding 📝
Dataset yang digunakan dalam proyek ini adalah Crop Recommendation Dataset, yang dimana dataset ini bersifat publik, bisa di akses melalui dengan link di bawah ini.

- [Kaggle](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset). 

Dataset ini terdiri dari 2200 baris data dan 8 fitur, mencakup informasi yang bisa digunakan untuk merekomendasikan jenis tanaman yang paling sesuai untuk ditanam berdasarkan kondisi lahan dan lingkungan, yang bisa membantu para petani dalam memaksimalkan produktivitas dan efisiensi penggunaan lahan.

### Fitur fitur pada Dataset Crop Recommendation Dataset 🌾:
| Fitur        | Deskripsi                                      | Tipe Data         | Rentang Nilai     | Contoh Nilai              |
|--------------|------------------------------------------------|-------------------|-------------------|---------------------------|
| N            | Rasio kandungan Nitrogen dalam tanah           | Numerik (int64)   | 0 - 140           | 90                        |
| P            | Rasio kandungan Phosphorous dalam tanah        | Numerik (int64)   | 5 - 145           | 42                        |
| K            | Rasio kandungan Potassium dalam tanah          | Numerik (int64)   | 5 - 205           | 43                        |
| temperature  | Suhu lingkungan (°C)                           | Numerik (float64) | 8.8 - 43.68       | 20.9                      |
| humidity     | Kelembapan relatif (%)                         | Numerik (float64) | 14.26 - 99.98     | 80.0                      |
| ph           | Nilai pH tanah                                 | Numerik (float64) | 3.5 - 9.94        | 6.5                       |
| rainfall     | Curah hujan (mm)                               | Numerik (float64) | 20.21 - 298.56    | 202.9                     |
| label        | Jenis tanaman yang direkomendasikan (target)   | Object            | 22 kelas unik     | rice, apple, orange, etc  |

Crop Recommendation Dataset ini sudah bersih, tidak ditemukan missing values pada fitur fitur yang ada, dan juga tidak adanya duplikasi data. Dataset ini sudah siap untuk digunakan dalam proses pelatihan model machine learning.

### Exploratory Data Analysis (EDA) 📊
#### Distribusi Label
<div align="center">
  <img src="img/Distribusi_Label.png"/>
</div>


Berdasarkan visualisasi di atas, dapat disimpulkan bahwa dataset terdiri dari **22 jenis tanaman** dengan jumlah sampel yang sangat merata, yaitu **100 data untuk setiap kategori**. Tidak ada kelas yang mendominasi secara signifikan bahkan kategori terbesar, *rice*, hanya mencakup sekitar **4,5%** dari total data.

Kondisi distribusi yang seimbang seperti ini sangat ideal untuk pemodelan klasifikasi. Dengan tidak adanya masalah ketidakseimbangan data (_imbalance_), model yang dibangun tidak akan bias terhadap kelas tertentu. Hal ini juga memastikan bahwa evaluasi performa model menjadi lebih adil dan hasil prediksi yang dihasilkan akan representatif untuk seluruh kelas tanaman yang ada di dataset.

#### Korelasi Fitur
<div align="center">
  <img src="img/correlation_matrix.png"/>
</div>

Analisis korelasi menunjukkan hanya ada satu hubungan yang sangat kuat, yaitu antara Phosphorus (P) dan Potassium (K) (nilai korelasi 0,74). Hal ini wajar karena kedua unsur ini sering diberikan bersamaan dalam pupuk NPK. Korelasi tinggi ini perlu diperhatikan saat pemodelan, karena bisa menyebabkan redundansi fitur. Oleh karena itu, teknik seperti feature selection atau reduksi dimensi dapat dipertimbangkan agar model tetap optimal.
