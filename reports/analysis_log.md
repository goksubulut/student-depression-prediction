# Analysis Log — Student Depression Prediction

Tüm veri bilimi ve analiz adımlarının bulguları, kararları ve sonuçları.

---

## STEP 1 — Data Quality Assessment (DQA)

**Tarih:** 2026-04-25
**Dosya:** `src/dqa.py`, `notebooks/student_depression_colab.ipynb`
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
**Dosya:** `src/eda.py`, `notebooks/student_depression_colab.ipynb`
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

### Missingness Pattern Görselleştirmesi
| Dosya | İçerik |
|---|---|
| `missingness_matrix.png` | Raw datasette null konumları (missingno matrix) |

- Raw datasette yalnızca **1 sütunda null** var: `Financial Stress` (3 satır)
- Nullity korelasyon heatmap'i atlandı — en az 2 null sütun gerekiyor; veri seti bu açıdan temiz.

### Train vs Test Dağılım Karşılaştırması
| Dosya | İçerik |
|---|---|
| `train_test_distribution.png` | Numeric özellikler için KDE — Train (mavi) vs Test (kırmızı) |

- Tüm numeric özelliklerde train ve test dağılımları birbiriyle örtüşüyor.
- **Data drift yok** — stratified split başarılı çalışmış.

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
**Dosya:** `src/train.py`, `notebooks/student_depression_colab.ipynb`
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

### SVM Sonuçları (Ek Model)
| Parametre | Değer |
|---|---|
| Kernel | RBF |
| Best C | 1.0 |
| Best gamma | auto |
| CV F1-macro | 0.8397 ± 0.0040 |
| CV ROC-AUC | 0.9190 |
| CV Accuracy | 0.8436 |
| MLflow Run ID | `c4cc4ad037604fd099a107b03adca38f` |
| Model dosyası | `models/svm_model.pkl` |

SVM (RBF, C=1.0, gamma=auto), LR_Tuned ile aynı F1-macro skorunu elde etti (0.8397). Bu veri setinde doğrusal ve RBF kernel karşılaştırılabilir sınırlar çiziyor — veriyi zaten doğrusal olarak ayrıştırılabilir kılıyor.

---

## STEP 2b — Embedding-Space Visualization (UMAP & t-SNE)

**Tarih:** 2026-04-27
**Dosya:** `src/embedding_viz.py`

| Grafik | Yöntem | Örnek sayısı |
|---|---|---|
| `umap_embedding.png` | UMAP (n_neighbors=30, min_dist=0.1) | 3.000 |
| `tsne_embedding.png` | t-SNE (perplexity=40, max_iter=1000) | 2.000 |

Her iki projeksionda da iki sınıf (depresyonlu/depresyonsuz) belirgin şekilde ayrışıyor — veri setinin doğrusal olarak iyi ayrıştırılabilir yapısını görsel olarak teyit ediyor.

---

## STEP 3b — Seed Sensitivity Analysis

**Tarih:** 2026-04-27
**Dosya:** `src/seed_sensitivity.py`

10 farklı seed (0, 7, 21, 42, 99, 123, 256, 512, 1337, 2024) üzerinde LogisticRegression (C=0.1) CV F1-macro:

| Metrik | Değer |
|---|---|
| Ortalama F1 (tüm seedler) | 0.8397 |
| Std (seed'ler arası) | **0.0005** |
| Min | 0.8391 (seed=0) |
| Max | 0.8403 (seed=123, 2024) |
| seed=42 F1 | 0.8397 |

**Bulgu:** Seed değişikliği modeli neredeyse hiç etkilemiyor (std=0.0005). seed=42 özellikle avantajlı değil — model stabil.

---

## STEP 4f — Bayesian Hyperparameter Optimization

**Tarih:** 2026-04-27
**Dosya:** `src/bayesian_opt.py`

- **Yöntem:** BayesSearchCV (scikit-optimize), n_iter=30
- **Arama uzayı:** C ∈ [0.001, 100] log-uniform, solver ∈ {lbfgs, saga}, max_iter ∈ [500, 2000]

| Parametre | GridSearch sonucu | Bayes sonucu |
|---|---|---|
| C | 0.1 | **0.0976** |
| solver | lbfgs | saga |
| max_iter | 1000 | 500 |
| CV F1-macro | 0.8397 | **0.8399** |

Bayesian opt GridSearch'ten +0.0002 kazanım sağladı. MLflow run: `f957fc98f6374da399f7111cdf15b6dc`

---

## STEP 4g — Artificial Neural Network (ANN / MLP)

**Tarih:** 2026-04-27
**Dosya:** `src/ann_model.py`

- **Mimari:** Input(15) → Dense(128) → Dense(64) → Dense(32) → Output(2)
- **Aktivasyon:** ReLU, **Optimizer:** Adam (lr=0.001), **Regularization:** L2 (α=0.001)
- **Early stopping:** validation_fraction=0.1, n_iter_no_change=15

| Metrik | ANN | LR_Tuned (Best) |
|---|---|---|
| CV F1-macro | 0.8388 ± 0.0030 | **0.8397 ± 0.0046** |
| CV ROC-AUC | 0.9183 | **0.9192** |
| CV Accuracy | 0.8450 | 0.8435 |

ANN, LR_Tuned'ı geçemedi. Tabular ve nispeten küçük boyutlu veri setinde doğrusal model daha iyi. Model: `models/ann_model.pkl`

---

## STEP 5c — LIME Explainability

**Tarih:** 2026-04-27
**Dosya:** `src/lime_analysis.py`

- **Yöntem:** LimeTabularExplainer, num_samples=1000, num_features=10
- 3 True Positive + 2 False Negative örnek açıklandı

| Grafik | İçerik |
|---|---|
| `lime_tp_sample*.png` | Doğru tahmin edilen depresyon vakaları |
| `lime_fn_sample*.png` | Kaçırılan depresyon vakaları |

LIME sonuçları SHAP ile tutarlı: `Have you ever had suicidal thoughts?` ve `Academic Pressure` her iki yöntemde de en güçlü faktörler.

---

## STEP 5d — Counterfactual Explanations

**Tarih:** 2026-04-27
**Dosya:** `src/counterfactual.py`

- **Yöntem:** Nearest-neighbour counterfactual — depresyonlu örnekler için en yakın depresyonsuz eğitim örneği bulunup delta hesaplandı
- 4 örnek açıklandı

| Grafik | İçerik |
|---|---|
| `counterfactual_explanations.png` | Tahmini çevirmek için gereken özellik değişimleri |

**Bulgu:** Çoğu vakada `Academic Pressure` ve `Have you ever had suicidal thoughts?` değerlerinin düşürülmesi/değiştirilmesi tahmini Depression=0'a çevirmek için yeterli — klinisyenler için müdahale noktası öneriyor.

---

## STEP 4d — Subgroup / Fairness Evaluation

**Tarih:** 2026-04-27
**Dosya:** `src/fairness.py`

### Analiz Edilen Subgrouplar
Gender, Age Group, Academic Pressure, Financial Stress, Family History of Mental Illness

### Sonuçlar (F1-macro | Recall-Depression)

| Subgroup | Group | n | F1-macro | Recall (Dep) |
|---|---|---|---|---|
| Overall | All | 5.572 | 0.835 | 0.837 |
| Gender | Female | 2.413 | 0.832 | 0.846 |
| Gender | Male | 3.159 | 0.838 | 0.831 |
| Age Group | <=20 | 1.105 | 0.804 | **0.886** |
| Age Group | 21–25 | 1.711 | 0.839 | 0.864 |
| Age Group | 26–30 | 1.558 | 0.821 | 0.803 |
| Age Group | 31+ | 1.198 | 0.833 | **0.759** ⚠️ |
| Academic Pressure | Low (1–2) | 1.857 | 0.774 | **0.586** ⚠️ |
| Academic Pressure | Medium (3) | 1.480 | 0.799 | 0.804 |
| Academic Pressure | High (4–5) | 2.235 | 0.768 | **0.925** |
| Financial Stress | Low (1–2) | 2.041 | 0.816 | **0.692** ⚠️ |
| Financial Stress | Medium (3) | 1.015 | 0.786 | **0.784** ⚠️ |
| Financial Stress | High (4–5) | 2.516 | 0.818 | 0.915 |
| Family History | Yes | 2.738 | 0.834 | 0.852 |
| Family History | No | 2.834 | 0.837 | 0.822 |

### Fairness Gaps
| Subgroup | F1 Gap | Recall Gap | Flagged |
|---|---|---|---|
| Gender | 0.006 | 0.016 | — |
| Age Group | 0.036 | **0.126** | 31+ |
| Academic Pressure | 0.031 | **0.340** | Low (1–2) |
| Financial Stress | 0.033 | **0.223** | Low (1–2), Medium (3) |
| Family History | 0.003 | 0.030 | — |

### Kaydedilen Grafikler
| Dosya | İçerik |
|---|---|
| `subgroup_fairness.png` | Subgroup bazında F1 / Recall / Precision bar chart |
| `subgroup_heatmap.png` | Tüm subgrouplar için metrik ısı haritası |

### Kritik Bulgular
- **Gender & Family History**: Adil dağılım, anlamlı fark yok.
- **Academic Pressure LOW (1–2)** ⚠️: Recall sadece **%58.6** — düşük baskılı ama depresyonda olan öğrencilerin %41'i kaçırılıyor. En kritik fairness açığı.
- **Financial Stress LOW (1–2)** ⚠️: Recall **%69.2** — düşük finansal stres ile depresyon kombinasyonu modeli zorluyor.
- **Age 31+** ⚠️: Recall **%75.9** — yaşlı öğrencilerde depresyon daha az görünür kalıyor.
- Yüksek baskılı/stresli gruplarda recall çok yüksek (>%90) — model bu grupları kolayca yakalıyor.

---

## STEP 4e — Stacking Ensemble (BONUS)

**Tarih:** 2026-04-27
**Dosya:** `src/ensemble.py`

### Yapı
- **Base learners:** Logistic Regression, Random Forest, XGBoost, LightGBM
- **Meta-learner:** Logistic Regression (C=1.0, balanced)
- **Stack method:** predict_proba (OOF tahminleri ile eğitim)
- **passthrough=True:** Meta-learner orijinal özellikleri de görüyor

### Sonuçlar
| Metrik | Stacking | LR_Tuned (Best Single) | Delta |
|---|---|---|---|
| CV F1-macro | 0.8395 ± 0.0032 | 0.8397 ± 0.0046 | -0.0002 |
| CV ROC-AUC | 0.9197 | 0.9192 | +0.0005 |
| CV Accuracy | 0.8433 | 0.8435 | -0.0002 |

**MLflow Run ID:** `77b2af4c49074ac985e7c395d8e72c86`
**Model dosyası:** `models/stacking_model.pkl`

### Yorum
Stacking ensemble LR_Tuned'ı pratikte geçemiyor (ΔF1 = -0.0002). Bu veri setinde temel ilişkiler doğrusal ve basit — karmaşık ensemble ekstra varyans kazancı sağlamıyor. **Üretim için LR_Tuned tercih edilir:** 100x daha hızlı inference, daha az bellek, daha yorumlanabilir.

---

## STEP 4b — Statistical Significance Testing

**Tarih:** 2026-04-27
**Dosya:** `src/significance.py`
**Karşılaştırılan modeller:** LogisticRegression_Tuned vs SVM_RBF_Tuned

### Testler ve Sonuçlar

#### 1. Paired t-test (5-fold CV F1-macro skorları üzerinde)
| | LR_Tuned | SVM_RBF |
|---|---|---|
| Fold 1 | 0.8392 | 0.8382 |
| Fold 2 | 0.8384 | 0.8376 |
| Fold 3 | 0.8323 | 0.8345 |
| Fold 4 | 0.8425 | 0.8419 |
| Fold 5 | 0.8460 | 0.8461 |
| **Mean** | **0.8397** | **0.8397** |

- t-statistic = 0.0171 | **p = 0.9872**
- Wilcoxon signed-rank p = 0.8125
- **Sonuç:** İstatistiksel olarak anlamlı fark yok — iki model CV skorları açısından birbirinden ayırt edilemiyor.

#### 2. McNemar's Test (test seti üzerinde hata örüntüsü karşılaştırması)
| | SVM doğru | SVM yanlış |
|---|---|---|
| **LR doğru** | 4.637 | 36 |
| **LR yanlış** | 53 | 846 |

- Statistic = 2.8764 | **p = 0.0899**
- **Sonuç:** p > 0.05 — iki modelin hata örüntüleri istatistiksel olarak farklı değil.

#### 3. Bootstrap %95 Güven Aralığı (LR_Tuned test F1-macro, n=1000)
- **[0.8288, 0.8475]** (mean=0.8385)

### Kaydedilen Grafik
| Dosya | İçerik |
|---|---|
| `cv_score_boxplot.png` | 5-fold CV F1 dağılımı — model başına boxplot |

### Genel Yorum
LR_Tuned ve SVM_RBF **istatistiksel olarak eşdeğer** performans gösteriyor. Hiçbir test (paired t-test, Wilcoxon, McNemar) p < 0.05 vermedi. Dağıtım için **Logistic Regression tercih edilir** — daha hızlı inference, daha yorumlanabilir katsayılar, daha düşük bellek kullanımı. SVM'in ek maliyeti (eğitim süresi, kernel hesabı) bu veri setinde anlamlı kazanım sağlamıyor.

---

## STEP 5b — SHAP Explainability

**Tarih:** 2026-04-26
**Dosya:** `src/shap_analysis.py`
**Explainer:** `shap.LinearExplainer` (LogisticRegression için)

### Kaydedilen Grafikler
| Dosya | İçerik |
|---|---|
| `shap_summary_beeswarm.png` | Her örnek için feature impact dağılımı |
| `shap_bar_importance.png` | Global önem — mean \|SHAP\| |
| `shap_waterfall_true_positive.png` | Doğru depresyon tahmini için bireysel açıklama |
| `shap_waterfall_false_negative.png` | Kaçırılan vaka için bireysel açıklama |
| `shap_dependence_Have_you_ever...png` | Suicidal thoughts SHAP dependence |
| `shap_dependence_Academic_Pressure.png` | Academic Pressure SHAP dependence |

### Top 10 Özellik — Mean |SHAP|
| Sıra | Özellik | Mean \|SHAP\| |
|---|---|---|
| 1 | Have you ever had suicidal thoughts ? | 1.079 |
| 2 | Academic Pressure | 0.893 |
| 3 | Financial Stress | 0.659 |
| 4 | Dietary Habits_Healthy | 0.435 |
| 5 | Work/Study Hours | 0.355 |
| 6 | Age | 0.352 |
| 7 | Study Satisfaction | 0.279 |
| 8 | Dietary Habits_Moderate | 0.243 |
| 9 | Family History of Mental Illness | 0.134 |
| 10 | CGPA | 0.078 |

### Bulgular
- **Suicidal thoughts** (SHAP=1.08) modelin en baskın sinyali — varlığı tek başına depresyon tahminini güçlü şekilde yukarı çekiyor.
- **Academic Pressure** (0.89) — yüksek baskı skoru SHAP değerini pozitif yönde sürüklüyor; düşük değerler negatif katkı yapıyor.
- **Financial Stress** (0.66) — monoton pozitif ilişki: stres arttıkça SHAP artar.
- **Dietary Habits_Healthy** (0.44) — sağlıklı beslenme depresyon olasılığını düşürüyor (negatif SHAP).
- SHAP sıralaması point-biserial korelasyon sıralamasıyla büyük ölçüde örtüşüyor — model yorumlanabilir ve tutarlı.

---

## STEP 4c — Model Card

**Tarih:** 2026-04-27
**Dosya:** `reports/model_card.md`

Model card şu bölümleri kapsamaktadır:
- Model detayları (hiperparametreler, MLflow run ID, dosya)
- Amaçlanan ve önerilmeyen kullanımlar
- Eğitim verisi ve ön işleme adımları
- Test/CV performans metrikleri + bootstrap CI
- Karşılaştırmalı model tablosu
- SHAP özellik önemi
- Hata analizi (FP/FN profilleri + eşik önerisi)
- Etik değerlendirme (kısıtlamalar, adalet, gizlilik)
- Yeniden üretilebilirlik bilgileri

---

## STEP 5 — Evaluation & Error Analysis

**Tarih:** 2026-04-25
**Dosya:** `src/evaluate.py`, `notebooks/student_depression_colab.ipynb`
**Model:** LogisticRegression_Tuned (C=0.1)
**Test seti:** 5.572 satır

### Test Seti Metrikleri
| Metrik | Değer |
|---|---|
| F1-macro | 0.8353 |
| ROC-AUC | 0.9208 |
| Average Precision | 0.9401 |
| Accuracy | 0.84 |

### Sınıf Bazında Sonuçlar
| Sınıf | Precision | Recall | F1 |
|---|---|---|---|
| No Depression (0) | 0.79 | 0.84 | 0.81 |
| Depression (1) | 0.88 | 0.84 | 0.86 |

### Kaydedilen Grafikler
| Dosya | İçerik |
|---|---|
| `confusion_matrix.png` | Confusion matrix ısı haritası |
| `roc_curve.png` | ROC eğrisi (AUC=0.9208) |
| `precision_recall_curve.png` | Precision-Recall eğrisi (AP=0.9401) |
| `feature_importance.png` | Top 15 permutation importance |

### Hata Analizi
| | Sayı |
|---|---|
| Toplam False Positive (FP) | 368 |
| Toplam False Negative (FN) | 531 |

**FP Profili** (depresyon yokken var dendi):
- Ortalama yaş: 28.7 | CGPA: 7.45 | Academic Pressure: 3.8 | Financial Stress: 3.1
- Yüksek akademik baskı ve intihar düşüncesi olan ama depresyon geliştirmemiş öğrenciler.

**FN Profili** (depresyon varken yok dendi):
- Ortalama yaş: 25.8 | CGPA: 7.32 | Academic Pressure: 2.7 | Financial Stress: 2.7
- Düşük akademik baskı skoru taşımasına rağmen depresyonda olan öğrenciler — model bu gizli vakaları kaçırıyor.
- FN grubunda `Have you ever had suicidal thoughts?` = No oranı yüksek: model bu sinyali bulamadan depresyonu tahmin edemiyor.

### Genel Değerlendirme
- Model **depresyon olan grubu** daha iyi yakalıyor (F1=0.86 vs 0.81).
- FN sayısı (531) FP'den (368) fazla — klinik açıdan FN'ler daha kritik (gözden kaçan vakalar).
- ROC-AUC 0.92 güçlü bir ayrıştırma gücüne işaret ediyor.
- Eşik değeri düşürülerek recall artırılabilir; trade-off kabul edilebilirse daha az FN elde edilir.
