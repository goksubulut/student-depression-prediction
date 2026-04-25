# Analysis Log — Student Depression Prediction

Tüm veri bilimi ve analiz adımlarının bulguları, kararları ve sonuçları.

---

## STEP 1 — Data Quality Assessment (DQA)

**Tarih:** 2026-04-25
**Dosya:** `src/dqa.py`, `notebooks/01_dqa.ipynb`
**Girdi:** `data/Student Depression Dataset.csv`
**Çıktı:** `data/cleaned.csv`

### Ham Veri
| Özellik | Değer |
|---|---|
| Satır sayısı | 27.901 |
| Sütun sayısı | 18 |
| Bellek kullanımı | 13.89 MB |

### Bulunan Sorunlar ve Alınan Kararlar

| # | Kontrol | Bulgu | Karar |
|---|---|---|---|
| 1 | Missing values | `Financial Stress`: 3 satır null | Silindi |
| 2 | Duplicate rows | 0 duplicate | İşlem yapılmadı |
| 3 | CGPA = 0 | 9 satır | Data entry hatası → silindi |
| 4 | Age outliers (IQR) | IQR sınırları: [7.5, 43.5] — 12 outlier; Age > 60: 0 satır | Age > 60 drop kuralı uygulandı, etkilenen satır yok |
| 5 | Work/Study Hours = 0 | 1.699 satır | Geçerli değer olabilir → flaglendi, silinmedi |
| 6 | Sleep Duration "Others" | 18 satır | Silindi (küçük grup, encode edilemiyor) |
| 7 | Dietary Habits "Others" | 12 satır | Silindi (küçük grup, encode edilemiyor) |
| 8 | Work Pressure (NZV) | %100 tek değer (0.0) | Near-zero variance → `features.py`'de drop edilecek |
| 9 | Job Satisfaction (NZV) | %99.99 tek değer (0.0) | Near-zero variance → `features.py`'de drop edilecek |

### Sınıf Dağılımı
| Sınıf | Satır | Oran |
|---|---|---|
| 0 — Depresyon Yok | 11.545 | %41.4 |
| 1 — Depresyon Var | 16.314 | %58.6 |

> Hafif imbalance (~1.4:1). `class_weight='balanced'` ve `scale_pos_weight` ile yönetilecek.

### Temizlik Özeti
- **Ham:** 27.901 satır
- **Temiz:** 27.859 satır (42 satır silindi)
- **Kaydedildi:** `data/cleaned.csv`

---

## STEP 2 — Exploratory Data Analysis (EDA)

**Tarih:** 2026-04-25
**Dosya:** `src/eda.py`, `notebooks/02_eda.ipynb`
**Girdi:** `data/cleaned.csv`
**Çıktı:** `reports/figures/` altında 8 grafik

### Kaydedilen Grafikler
| Dosya | İçerik |
|---|---|
| `univariate_numeric.png` | Age, CGPA, Academic Pressure, Work/Study Hours, Financial Stress histogram + KDE |
| `univariate_categorical.png` | Gender, Sleep Duration, Dietary Habits, Degree, Suicidal Thoughts, Family History countplot |
| `bivariate_numeric_boxplots.png` | Numeric özellikler × Depression boxplot |
| `bivariate_categorical_countplots.png` | Kategorik özellikler × Depression countplot |
| `mean_features_by_depression.png` | Sınıfa göre ortalama numeric değerler |
| `correlation_heatmap.png` | Tüm encode edilmiş özellikler arası korelasyon ısı haritası |
| `class_distribution_pie.png` | Sınıf dağılımı pasta grafik |
| `pairplot_top5.png` | En yüksek korelasyonlu 5 özellik pair plot |

### Point-Biserial Korelasyon (Target ile)
| Sıra | Özellik | \|r\| |
|---|---|---|
| 1 | Have you ever had suicidal thoughts ? | 0.546 |
| 2 | Academic Pressure | 0.475 |
| 3 | Financial Stress | 0.364 |
| 4 | Age | 0.227 |
| 5 | Work/Study Hours | 0.209 |

### Öne Çıkan Bulgular
- **Suicidal thoughts** tek başına en güçlü sinyal (r=0.55) — beklenebilir, ancak model için kritik özellik.
- **Academic Pressure** güçlü pozitif ilişki: depresyonlu grubun baskı skoru belirgin şekilde yüksek.
- **Financial Stress** da anlamlı pozitif korelasyon.
- **Age**: genç öğrencilerde depresyon oranı daha yüksek (negatif değil, pozitif ilişki — veri seti ağırlıklı olarak genç bireylerden oluşuyor).
- **Work Pressure & Job Satisfaction** %100 / %99.99 tek değer → bilgi taşımıyor, drop edilecek.

---

## STEP 3 — Feature Engineering

**Tarih:** 2026-04-25
**Dosya:** `src/features.py`
**Girdi:** `data/cleaned.csv`

### Uygulanan Adımlar
| Adım | Detay |
|---|---|
| Drop | `id`, `Profession`, `Work Pressure`, `Job Satisfaction`, `City` |
| Binary encode | `Have you ever had suicidal thoughts ?`, `Family History of Mental Illness` → 0/1 |
| Ordinal encode | `Sleep Duration` → 1–5 |
| OHE | `Gender`, `Dietary Habits`, `Degree` |
| MinMax scale | `Age`, `CGPA`, `Academic Pressure`, `Work/Study Hours`, `Financial Stress`, `AP_x_Sleep` |
| Interaction | `AP_x_Sleep` = Academic Pressure × Sleep Duration encoded |
| Split | %80 train / %20 test, stratified, seed=42 |
| RFE | LogisticRegression ile top 15 özellik seçimi (43 → 15) |

### Train / Test Boyutları
| Set | Satır | 0 | 1 |
|---|---|---|---|
| Train | 22.287 | 9.236 | 13.051 |
| Test | 5.572 | 2.309 | 3.263 |

### RFE — Seçilen 15 Özellik
`Age`, `CGPA`, `Academic Pressure`, `Work/Study Hours`, `Financial Stress`,
`Gender_Female`, `Gender_Male`, `Dietary Habits_Healthy`, `Dietary Habits_Moderate`,
`Degree_BA`, `Degree_LLM`, `Degree_ME`,
`Study Satisfaction`, `Have you ever had suicidal thoughts ?`, `Family History of Mental Illness`

### RFE — Drop Edilen 28 Özellik
`AP_x_Sleep` (interaction feature bilgi katmadı), `Sleep Duration` (ordinal encode sonrası RFE tarafından elendi),
`Dietary Habits_Unhealthy` ve çok sayıda seyrek `Degree_*` OHE sütunu.

---

## STEP 4 — Modeling

**Tarih:** 2026-04-25
**Dosya:** `src/train.py`, `notebooks/03_modeling.ipynb`
**MLflow Experiment:** `student-depression-prediction`
**CV:** Stratified 5-Fold, seed=42

### Model Karşılaştırması (CV sonuçları, F1-macro'ya göre sıralı)

| Model | F1-macro | ±std | ROC-AUC | Accuracy | MLflow Run ID |
|---|---|---|---|---|---|
| LogisticRegression_Tuned | 0.8397 | 0.0046 | 0.9192 | 0.8435 | `bf1830701cf44f51a1eb807b32e8f830` |
| LogisticRegression | 0.8388 | 0.0038 | 0.9195 | 0.8427 | `754ea1b039f14f35a8e39629fcf29cac` |
| LightGBM | 0.8369 | 0.0022 | 0.9140 | 0.8409 | `33ef7f30bff34785a957e055ad57f5f4` |
| RandomForest | 0.8304 | 0.0039 | 0.9102 | 0.8367 | `af88e81613fd4e5b8f38f0b4a229cffa` |
| XGBoost | 0.8233 | 0.0042 | 0.9010 | 0.8281 | `df72da2a012a48fca6642d4ec2d3fe25` |
| DecisionTree | 0.8156 | 0.0036 | 0.8946 | 0.8197 | `bc230a959fa34bf5b6092f3a8189f737` |

### Bulgular
- **En iyi model: LogisticRegression_Tuned** (C=0.1) — F1-macro: 0.8397, ROC-AUC: 0.9192
- Logistic Regression, tree-based modellerin tamamını geride bıraktı. Bu veri seti için doğrusal bir sınır yeterli.
- LightGBM en düşük varyansa sahip (std=0.0022) — tutarlı ama LR kadar iyi değil.
- GridSearchCV ile en iyi C=0.1 bulundu (default 1.0'dan hafif regularization artışı fayda sağladı).
- **Best model kaydedildi:** `models/best_model.pkl`

---

## STEP 5 — Evaluation & Error Analysis

*Henüz tamamlanmadı.*
