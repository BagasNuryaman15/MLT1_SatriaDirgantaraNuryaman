# Laporan Proyek Machine Learning Classification : Crop Recommendation System
# Satria Dirgantara Nuryaman

# Domain Proyek
Pertanian merupakan sektor vital yang berperan besar dalam ketahanan pangan dan perekonomian, khususnya di negara agraris seperti Indonesia. Namun, petani sering menghadapi tantangan dalam menentukan jenis tanaman yang paling sesuai untuk ditanam pada lahan tertentu, mengingat banyaknya faktor yang memengaruhi seperti kandungan unsur hara tanah, suhu, kelembapan, pH tanah, dan curah hujan. Kesalahan dalam pemilihan jenis tanaman dapat menyebabkan penurunan produktivitas, kerugian ekonomi, dan ketidakseimbangan ekosistem.

Seiring berkembangnya teknologi, pemanfaatan data dan kecerdasan buatan (Artificial Intelligence) menjadi solusi potensial untuk mengoptimalkan proses pengambilan keputusan di bidang pertanian. Dengan memanfaatkan data lingkungan dan karakteristik tanah, sistem rekomendasi berbasis machine learning dapat membantu petani memilih jenis tanaman yang paling optimal, sehingga meningkatkan hasil panen, efisiensi penggunaan lahan, dan keberlanjutan pertanian.

## Business Understanding

### Problem Statements

Dalam dunia pertanian modern, petani dihadapkan pada tantangan untuk menentukan jenis tanaman yang paling sesuai dengan kondisi lahan dan lingkungan yang dimiliki. Berbagai faktor seperti kandungan Nitrogen, Phosphorous, Potassium, suhu, kelembapan, pH tanah, dan curah hujan, semuanya saling memengaruhi dan menentukan keberhasilan panen. Namun, kompleksitas hubungan antar faktor tersebut seringkali menyulitkan pengambilan keputusan secara manual.

Oleh karena itu, muncul pertanyaan pertanyaan penting yang ingin dijawab melalui proyek ini:
- Bagaimana pengaruh faktor lingkungan dan tanah terhadap pemilihan jenis tanaman yang optimal?
- Fitur apa yang paling dominan dalam menentukan rekomendasi tanaman untuk lahan tertentu?
- Jenis tanaman apa yang paling tepat direkomendasikan untuk kondisi lahan dan lingkungan tertentu?

### Goals

Untuk menjawab pertanyaan pertanyaan tersebut, proyek ini bertujuan membangun sebuah sistem rekomendasi berbasis machine learning yang mampu:
- Memberikan rekomendasi jenis tanaman secara akurat berdasarkan data lingkungan dan tanah yang tersedia.
- Mengidentifikasi fitur fitur kunci yang paling berpengaruh dalam proses pengambilan keputusan, sehingga dapat menjadi acuan bagi petani dalam mengelola lahan secara lebih efektif dan berkelanjutan.

### Solution Statements

Alur solusi yang diusulkan dalam proyek ini dimulai dari hulu, yaitu pengumpulan dan pemahaman data, hingga ke hilir, yaitu implementasi dan evaluasi model:
- Melakukan analisis statistik dan visualisasi data untuk menemukan pola, outlier, serta korelasi antara variabel lingkungan/tanah dengan jenis tanaman.
- Mengimplementasikan beberapa algoritma klasifikasi seperti KNeighborsClassifier, RandomForestClassifier, dan XGBoost untuk membangun model prediktif yang andal.
- Mengevaluasi performa model menggunakan metrik akurasi, precision, recall, F1-score, dan confusion matrix, guna memastikan rekomendasi yang dihasilkan benar-benar dapat diandalkan.
- Melakukan hyperparameter tuning dengan GridSearchCV untuk mengoptimalkan performa model, sehingga solusi yang dihasilkan benar-benar siap diterapkan di lapangan.
