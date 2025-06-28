# -*- coding: utf-8 -*-
"""
**Proyek Predictive Analytics: [Crop Recommendation Dataset]**

- **Nama** : Satria Dirgantara Nuryaman
- **Email** : satriadirgantaranuryaman15@gmail.com
- **ID Dicoding** : Satria Dirgantara Nuryaman 
"""

# ## **Import Library**

# Data Manipulation
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Data Processing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import time

# Model Machine Learning
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# Model Evaluation
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score

# Model Tuning
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Warnings
import warnings 

# Configureations
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

# ## **Data Loading**

# Dataset ini 

FILE_PATH = 'data/Crop_recommendation.csv'
df = pd.read_csv(FILE_PATH)
display(df.head(), df.tail())

# Dataset Crop Recommendation ini memiliki:
# - 2200 baris.
# - 8 fitur yang bernama **N (Nitrogen)**, **P (Phosphorous)**, **K (Potassium)**, **temperature**, **humidity**, **ph**, **rainfall**, dan **label**. 

# ## **Exploratory Data Analysis (EDA)**

# ### **Deskripsi Variable Dataset**

# | Fitur        | Deskripsi                                      |
# |--------------|------------------------------------------------|
# | N            | Rasio kandungan Nitrogen dalam tanah           |
# | P            | Rasio kandungan Phosphorous dalam tanah        |
# | K            | Rasio kandungan Potassium dalam tanah          |
# | temperature  | Suhu lingkungan (°C)                           |
# | humidity     | Kelembapan relatif (%)                         |
# | ph           | Nilai pH tanah                                 |
# | rainfall     | Curah hujan (mm)                               |
# | label        | Jenis tanaman yang direkomendasikan (target)   |

df.info()

# Seperti yang sudah di jelaskan dataset ini memiliki 2200 baris yang terdiri dari 8 Fitur yang berbeda beda tipe datanya:
# - **temperature**, **humidity**, **ph**, dan **rainfall** bertipe data float64.
# - **N (Nitrogen)**, **P (Phosphorous)**, dan **K (Potassium)** bertipe data float64.
# - **label** bertipe data object.
# Bisa kita lihat juga ternyata tidak ada missing values di dataset kita ini, inkonsistensi data juga tidak ada, semuanya sudah clean, tapi kita harus memastikan nya lagi.

df.describe()

# Berdasarkan analisis deskriptif terhadap 2.200 sampel, ditemukan bahwa fitur fitur lingkungan dan tanah memiliki sebaran nilai yang cukup luas. 
# - Unsur hara tanah seperti **nitrogen (N: 0 – 140 ppm)**, **fosfor (P: 5 – 145 ppm)**, dan **potassium (K: 5 – 205 ppm)** menunjukkan variasi yang signifikan, menandakan perbedaan kondisi tanah yang cukup ekstrem antar wilayah. 
# - Kandungan kalium secara khusus dan ajaib memiliki standar deviasi tertinggi (**50.6**), menunjukkan keragaman yang besar dan kemungkinan peran pentingnya dalam penentuan jenis tanaman yang sesuai.
# - Rata rata suhu sebesar **25.6 C** dengan rentang **8.8** – **43.7 C**, serta kelembapan **71.5%** (**14.3 – 99.9%**), mengindikasikan bahwa dataset ini mencakup berbagai kondisi iklim. pH tanah umumnya netral hingga sedikit asam/basa (rata rata **6.47**), dan curah hujan berkisar antara **20.2** – **298.6** mm, mendukung analisis untuk berbagai jenis tanaman yang tumbuh di lingkungan berbeda.
# Dengan kondisi lingkungan yang seberagam ini, penting untuk tahu fitur mana saja yang benar benar berpengaruh terhadap pilihan tanaman. Ini bukan cuma penting untuk akurasi model, tapi juga bisa jadi pegangan nyata bagi petani dalam mengelola lahannya dengan lebih tepat.

# ### **Identifikasi Missing Values**

missing_values = df.isnull().sum()
missing_values_percentage = (missing_values / len(df)) * 100

missing_df = pd.DataFrame({
    'Missing Values' : missing_values,
    'Percentage' : missing_values_percentage
}) 

display(missing_df[missing_df['Missing Values'] > 0])

# Ternyata benar informasi kita diatas, tidak ada nya missing values disini.

# ### **Mengecek Duplicated Data**

print(f'Jumlah data yang terduplikasi adalah {df.duplicated().sum()}')

# Ternyata dataset Crop Recommendation sudah bersih, di buktikan dengan ketidakadaan nilai yang NaN dan juga duplikasi.

# ### **Identifikasi Outliers**

num_features = df.select_dtypes(include=[np.number]).columns.tolist()
fig, ax = plt.subplots(2, len(num_features), figsize=(4 * len(num_features), 8))

for i, feature in enumerate(num_features):
    # Boxplot dengan seaborn
    sns.boxplot(y=df[feature], ax=ax[0, i], color='skyblue', fliersize=5)
    ax[0, i].set_title(f'Boxplot {feature}')
    ax[0, i].set_ylabel(feature)
    ax[0, i].grid(True, axis='y', linestyle='--', alpha=0.7)

    # Histogram dengan seaborn
    sns.histplot(df[feature], bins=30, ax=ax[1, i], color='skyblue', edgecolor='black')
    ax[1, i].set_title(f'Histogram {feature}')
    ax[1, i].set_xlabel(feature)
    ax[1, i].set_ylabel('Frequency')
    ax[1, i].grid(True, axis='y', linestyle='--', alpha=0.7)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# Outlier summary
print(' Jumlah outlier pada setiap fitur '.center(50 , '+'))
outlier_summary = []
for feature in num_features:
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
    outlier_summary.extend(outliers)

    print(f'{feature}: {len(outliers)} outliers detected')

# Visualisasi boxplot memperlihatkan banyak outlier pada fitur seperti **P (Phosphorous)**, **K (Potassium)**, **temperature**, **ph**, dan **rainfall**. Namun, ketika kita lihat bersama dengan visualisasi histogramnya, kita mendapatkan gambaran yang lebih utuh tentang sifat distribusinya.
# - **P** dan **K** memang memiliki outlier di ujung atas, tapi histogram **P (Fosfor)** menunjukkan distribusi yang agak miring ke kanan (**right-skewed**) dan histogram **K (Potassium)** sama seperti **P** tapi terlihat lebih parah yang sangat (**right-skewed**). Kalau positifnya, nilai nilai tinggi itu bukan anomali acak bisa jadi mewakili kelompok wilayah yang secara alami memiliki kandungan hara tinggi dan sangat cocok dijadikan lahan pertanian.
# - **temperature** dan **ph** tampak hampir normal di histogram, tapi tetap punya outlier di boxplot. Ini mengindikasikan beberapa kondisi ekstrem yang jarang tapi masih terstruktur, seperti daerah dataran tinggi atau sangat basa/asam.
# - **rainfall** punya outlier yang cukup banyak, dan histogramnya mendukung distribusinya jelas skewed ke kanan. Ini menunjukkan beberapa wilayah memang memiliki curah hujan jauh di atas rata rata dan ini bukan kesalahan data, tapi refleksi dari kondisi dunia nyata.
# - **nitrogen** dan