# Model Card — Student Depression Prediction

> Follows the [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993) framework (Mitchell et al., 2019).

---

## Model Details

| Alan | Değer |
|---|---|
| **Model adı** | LogisticRegression_Tuned |
| **Model tipi** | Binary Logistic Regression (L2 regularization) |
| **Versiyon** | 1.0 |
| **Tarih** | 2026-04-27 |
| **Dosya** | `models/best_model.pkl` |
| **Framework** | scikit-learn 1.3+ |
| **Hiperparametreler** | C=0.1, class_weight='balanced', max_iter=1000, solver='lbfgs' |
| **MLflow Run ID** | `bf1830701cf44f51a1eb807b32e8f830` |

---

## Amaçlanan Kullanım

### Birincil Kullanım
Öğrencilerde depresyon riskini **erken tespit etmek** amacıyla akademik, yaşam tarzı ve demografik özelliklerden ikili sınıflandırma (Depresyonlu / Depresyonsuz) yapmak.

### Kullanım Senaryoları
- Üniversite psikolojik danışmanlık birimlerinde yüksek riskli öğrencilerin önceliklendirilmesi
- Öğrenci refahı araştırmalarında keşifsel analiz aracı
- Akademik stres ile ruh sağlığı ilişkisini inceleyen çalışmalarda destek modeli

### Önerilmeyen Kullanımlar
- **Klinik teşhis aracı olarak kullanılmamalıdır.** Model bir psikiyatrist veya klinisyenin yerini alamaz.
- Bireysel öğrenciler hakkında disiplin veya cezai kararlar almak için kullanılmamalıdır.
- Eğitim verisi dışındaki popülasyonlara (farklı kültür, yaş grubu, eğitim sistemi) doğrudan uygulanmamalıdır.

---

## Eğitim Verisi

| Alan | Detay |
|---|---|
| **Kaynak** | [hopesb/student-depression-dataset](https://www.kaggle.com/datasets/hopesb/student-depression-dataset) |
| **Ham boyut** | 27.901 satır × 18 sütun |
| **Temizlenmiş boyut** | 27.859 satır × 18 sütun |
| **Eğitim seti** | 22.287 satır (80%) |
| **Test seti** | 5.572 satır (20%) |
| **Split stratejisi** | Stratified, random_state=42 |
| **Hedef değişken** | `Depression` (0=Yok, 1=Var) |
| **Sınıf dağılımı** | 0: %41.4 (11.545) / 1: %58.6 (16.314) |

### Uygulanan Ön İşleme
- 3 satır (Financial Stress null) silindi
- 9 satır (CGPA=0, veri girişi hatası) silindi
- 30 satır ("Others" kategorisi) silindi
- `Work Pressure` ve `Job Satisfaction` (%100 / %99.99 tek değer) drop edildi
- Sleep Duration → ordinal encode (1–5)
- Gender, Dietary Habits, Degree → One-Hot Encode
- Suicidal Thoughts, Family History → binary encode (0/1)
- Age, CGPA, Academic Pressure, Work/Study Hours, Financial Stress → MinMax normalize
- RFE ile 43 özellikten 15 özellik seçildi

---

## Model Performansı

### Test Seti Sonuçları (n=5.572)

| Metrik | Değer |
|---|---|
| **F1-macro** | **0.8353** |
| **ROC-AUC** | **0.9208** |
| Average Precision | 0.9401 |
| Accuracy | 0.84 |

### Sınıf Bazında

| Sınıf | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| No Depression (0) | 0.79 | 0.84 | 0.81 | 2.309 |
| Depression (1) | 0.88 | 0.84 | 0.86 | 3.263 |

### Cross-Validation (Stratified 5-Fold)

| Metrik | Mean | Std |
|---|---|---|
| F1-macro | 0.8397 | 0.0046 |
| ROC-AUC | 0.9192 | — |
| Accuracy | 0.8435 | — |

### Bootstrap %95 Güven Aralığı (F1-macro)
**[0.8288, 0.8475]** — 1000 resample, random_state=42

---

## Karşılaştırmalı Model Sonuçları

| Model | CV F1-macro | CV ROC-AUC |
|---|---|---|
| **LogisticRegression_Tuned** ✅ | **0.8397** | **0.9192** |
| SVM_RBF_Tuned | 0.8397 | 0.9190 |
| LightGBM | 0.8369 | 0.9140 |
| RandomForest | 0.8304 | 0.9102 |
| XGBoost | 0.8233 | 0.9010 |
| DecisionTree | 0.8156 | 0.8946 |

> LR_Tuned ve SVM istatistiksel olarak eşdeğer (McNemar p=0.09, paired t-test p=0.99).
> Logistic Regression seçildi: daha hızlı inference, yorumlanabilir katsayılar, düşük bellek.

---

## Özellik Önemi (SHAP — Mean |SHAP Value|)

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

---

## Hata Analizi

| | Sayı | Oran |
|---|---|---|
| False Positives (FP) | 368 | %6.6 (test seti) |
| False Negatives (FN) | 531 | %9.5 (test seti) |

**FP Profili:** Yüksek akademik baskı + intihar düşüncesi geçmişi olan ama depresyon geliştirmemiş öğrenciler (ort. yaş 28.7, Academic Pressure 3.8).

**FN Profili:** Düşük görünen baskı skorlarına rağmen depresyonda olan öğrenciler (ort. yaş 25.8, Academic Pressure 2.7). Bu grup klinik açıdan en riskli segment.

> **Öneri:** Klinik uygulamada FN maliyeti FP'den yüksektir. Karar eşiği 0.5'ten 0.4'e düşürülerek recall artırılabilir (daha az FN, daha fazla FP kabul edilir).

---

## Etik Değerlendirme

### Bilinen Kısıtlamalar
- Veri seti Hindistan'daki üniversite öğrencilerinden toplanmıştır; farklı coğrafya ve kültürlere genelleme yapılmamalıdır.
- `Have you ever had suicidal thoughts?` sorusu güçlü bir sinyal olmakla birlikte, bu soruyu yanıtlamayan veya yanıtı gizleyen bireyler için model zayıflayabilir.
- Modelin yanlış negatif üretmesi (FN=531) göz ardı edilmiş vakalar anlamına gelir — yüksek riskli ortamlarda tek başına kullanılmamalıdır.

### Adalet ve Önyargı
- Cinsiyet (Gender) modelde kullanılmaktadır; gruplar arası performans farkı kontrol edilmelidir.
- Sosyoekonomik faktörler (Financial Stress) dahil edilmiş olmakla birlikte ölçüm kalitesi sınırlıdır.

### Veri Gizliliği
- Eğitim verisindeki bireyler anonim olmalıdır; gerçek uygulamada KVKK/GDPR uyumu zorunludur.

---

## Yeniden Üretilebilirlik

| Alan | Değer |
|---|---|
| Random seed | 42 (tüm adımlarda) |
| Python | 3.10+ |
| Bağımlılıklar | `requirements.txt` |
| Deney takibi | MLflow (local), experiment: `student-depression-prediction` |
| Kod | `src/` klasörü |
| Uçtan uca çalıştırma | `python main.py` |

---

## İletişim

Proje: **SENG 352 — Student Depression Prediction**
Repo: [github.com/goksubulut/student-depression-prediction](https://github.com/goksubulut/student-depression-prediction)
