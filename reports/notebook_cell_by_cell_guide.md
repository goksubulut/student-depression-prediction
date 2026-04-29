# Notebook Sunum Rehberi (Hücre Hücre)

Bu dosya `notebooks/student_depression_colab.ipynb` için hazırlanmıştır.  
Amaç: Sunum sırasında her hücrede ne yaptığınızı, bunun veri bilimi açısından anlamını, kullanılan teknolojileri/algoritmaları ve hesaplamaları hızlıca anlatabilmek.

---

## Cell 00 (Markdown) — Proje başlığı ve yol haritası
- **Ne yapar:** Projenin başlığını, kapsamını ve 11 bölümlük akışını verir.
- **DS anlamı:** Problem framing + analiz planı; değerlendirmede “metodolojik düzen” gösterir.
- **Teknoloji/Algoritma:** Yok (dokümantasyon).
- **Hesaplama:** Yok.

## Cell 01 (Code) — Paket kurulumları
- **Ne yapar:** Colab ortamına gerekli kütüphaneleri kurar.
- **DS anlamı:** Reproducibility (aynı ortamda tekrar çalıştırılabilirlik).
- **Teknoloji/Algoritma:** `pip`, `missingno`, `shap`, `plotly`, `xgboost`, `lightgbm`, `imbalanced-learn`, `gradio`.
- **Hesaplama:** Yok (ortam hazırlığı).

## Cell 02 (Code) — Import + tema + global ayarlar
- **Ne yapar:** Tüm temel kütüphaneleri import eder, görsel tema/renk paletini ve yardımcı fonksiyonları tanımlar.
- **DS anlamı:** Analiz standardizasyonu + tutarlı görselleştirme + deney düzeni.
- **Teknoloji/Algoritma:** `pandas`, `numpy`, `sklearn`, `plotly`, `matplotlib`, `seaborn`, `missingno`, opsiyonel `xgboost/lightgbm/shap`.
- **Hesaplama:** Yok; klasör oluşturma (`reports/figures`, `models`) ve plotting config.

## Cell 03 (Code) — Veri yükleme (Colab/local fallback)
- **Ne yapar:** Colab’da dosya upload ile, lokalde path ile veri setini okur.
- **DS anlamı:** Dağıtık çalışma senaryosu (colab/lokal) + veri giriş esnekliği.
- **Teknoloji/Algoritma:** `google.colab.files`, `pandas.read_csv`.
- **Hesaplama:** Veri boyutu ve ilk satırların gösterimi.

## Cell 04 (Markdown) — DQA bölüm başlığı
- **Ne yapar:** Data Quality Assessment bölümünü açar.
- **DS anlamı:** Temiz veri olmadan model güvenilir olmaz.
- **Teknoloji/Algoritma:** Yok.
- **Hesaplama:** Yok.

## Cell 05 (Code) — DQA temizleme adımları
- **Ne yapar:** Missing, duplicate, `CGPA=0`, yaş filtresi, “Others” kategorileri gibi kalite kontrollerini uygular.
- **DS anlamı:** Noise azaltma, encoding uyumluluğu, veri güvenilirliği.
- **Teknoloji/Algoritma:** `pandas` filtreleme/temizlik.
- **Hesaplama:** Kaç satır silindi, temiz veri boyutu.

## Cell 06 (Code) — DQA görsel özeti
- **Ne yapar:** Missing bar, sınıf dağılımı, yaş dağılımı (hist + KDE) tek figürde verir.
- **DS anlamı:** Veri kalitesi + target balance + dağılım davranışı.
- **Teknoloji/Algoritma:** `matplotlib`, `scipy.gaussian_kde`.
- **Hesaplama:** Missing yüzdeleri, class count/ratio, yaş yoğunluk tahmini.

## Cell 07 (Markdown) — DQA insight metni
- **Ne yapar:** DQA bulgularını kısa yorumlar.
- **DS anlamı:** Ham çıktıdan karar çıkarımı.
- **Teknoloji/Algoritma:** Yok.
- **Hesaplama:** Yok.

## Cell 08 (Markdown) — EDA bölüm başlığı
- **Ne yapar:** Exploratory Data Analysis bölümünü başlatır.
- **DS anlamı:** Model öncesi veri davranışını anlama.
- **Teknoloji/Algoritma:** Yok.
- **Hesaplama:** Yok.

## Cell 09 (Code) — Sayısal özellik dağılım analizi
- **Ne yapar:** Numeric değişkenleri sınıfa göre hist+KDE karşılaştırır.
- **DS anlamı:** Class separation sinyali (hangi özellik ayrıştırıcı).
- **Teknoloji/Algoritma:** `matplotlib`, `scipy.gaussian_kde`.
- **Hesaplama:** Sınıf bazlı yoğunluk karşılaştırmaları.

## Cell 10 (Markdown) — EDA insight
- **Ne yapar:** Academic/financial pressure ve CGPA bulgularını yorumlar.
- **DS anlamı:** Özelliklerin hedefle ilişkisini anlatır.
- **Teknoloji/Algoritma:** Yok.
- **Hesaplama:** Yok.

## Cell 11 (Code) — Violin plotlar
- **Ne yapar:** Numeric değişkenlerin sınıf bazlı dağılım şekillerini violin plotla verir.
- **DS anlamı:** Medyan/çeyreklik/yoğunluk farkı analizi.
- **Teknoloji/Algoritma:** `seaborn.violinplot`.
- **Hesaplama:** Sınıf dağılımlarının görsel karşılaştırması.

## Cell 12 (Code) — Sankey akışı
- **Ne yapar:** Suicidal thoughts -> family history -> depression akışını görselleştirir.
- **DS anlamı:** Çok aşamalı etki ilişkisini anlatır.
- **Teknoloji/Algoritma:** `plotly.graph_objects.Sankey`.
- **Hesaplama:** `groupby(...).size()` ile akış bağlantı ağırlıkları.

## Cell 13 (Markdown) — Sankey insight
- **Ne yapar:** Sankey akışındaki kritik yolu metinle açıklar.
- **DS anlamı:** Görselden klinik/işsel yorum üretme.
- **Teknoloji/Algoritma:** Yok.
- **Hesaplama:** Yok.

## Cell 14 (Code) — Korelasyon için encoding + heatmap
- **Ne yapar:** Kategorikleri encode edip korelasyon matrisi/heatmap üretir.
- **DS anlamı:** Doğrusal ilişkilerin global görünümü.
- **Teknoloji/Algoritma:** `pd.get_dummies`, `corr()`, `seaborn.heatmap`.
- **Hesaplama:** Pearson korelasyonları, target ile top korelasyonlar.

## Cell 15 (Code) — Point-biserial analiz
- **Ne yapar:** Binary target ile numeric özellik ilişkisini `pointbiserialr` ile skorlar.
- **DS anlamı:** Predictive power sıralaması.
- **Teknoloji/Algoritma:** `scipy.stats.pointbiserialr`.
- **Hesaplama:** Özellik başına |r|, top feature listesi.

## Cell 16 (Markdown) — Predictive power insight
- **Ne yapar:** En güçlü sürücüleri özetler.
- **DS anlamı:** Model öncesi özellik önem hipotezi.
- **Teknoloji/Algoritma:** Yok.
- **Hesaplama:** Yok.

## Cell 17 (Code) — Kategorik değişken depresyon oranları
- **Ne yapar:** Kategorik gruplarda depresyon oranı barh grafikleri üretir.
- **DS anlamı:** Segment bazlı risk farklılıkları.
- **Teknoloji/Algoritma:** `groupby().mean()`, `matplotlib`.
- **Hesaplama:** Category-wise depression rate.

## Cell 18 (Code) — Donut target distribution
- **Ne yapar:** Hedef sınıf dağılımını donut chart ile sunar.
- **DS anlamı:** Sınıf dengesini sunuma uygun net göstermek.
- **Teknoloji/Algoritma:** `plotly.pie`.
- **Hesaplama:** Sınıf payları.

## Cell 19 (Markdown) — Feature engineering başlığı
- **Ne yapar:** Feature pipeline adımlarını listeler.
- **DS anlamı:** Modelleme hazırlık tasarımını açıklar.
- **Teknoloji/Algoritma:** Yok.
- **Hesaplama:** Yok.

## Cell 20 (Code) — Feature engineering pipeline
- **Ne yapar:** Drop, encoding, scaling, interaction, split, RFE ile model girdisini üretir.
- **DS anlamı:** Leakage-safe, üretime uygun feature workflow.
- **Teknoloji/Algoritma:** `ColumnTransformer`, `OneHotEncoder`, `MinMaxScaler`, `RFE`, `LogisticRegression`.
- **Hesaplama:** Train/test ayrımı, feature sayısı (43->15), seçilen/düşen özellikler.

## Cell 21 (Code) — RFE seçilen özellik görselleştirmesi
- **Ne yapar:** Seçilen feature listesini görsel olarak sunar.
- **DS anlamı:** Model sadeleştirme ve explainability.
- **Teknoloji/Algoritma:** `matplotlib`.
- **Hesaplama:** Seçilen özelliklerin sunumu.

## Cell 22 (Markdown) — Model training başlığı
- **Ne yapar:** Eğitim stratejisini açar.
- **DS anlamı:** Baseline->tuned model yaklaşımı.
- **Teknoloji/Algoritma:** Yok.
- **Hesaplama:** Yok.

## Cell 23 (Code) — CV model karşılaştırma
- **Ne yapar:** Çoklu modeli stratified 5-fold CV ile değerlendirir.
- **DS anlamı:** Tek split’e bağımlılığı azaltır, sağlam model seçimi.
- **Teknoloji/Algoritma:** LR, DT, RF, XGB, LGB; `cross_validate`.
- **Hesaplama:** F1-macro, ROC-AUC, accuracy, precision, recall mean/std.

## Cell 24 (Code) — GridSearch tuning + model kaydetme
- **Ne yapar:** En iyi baz modeli hyperparameter tune eder, final modeli kaydeder.
- **DS anlamı:** Performans optimizasyonu + deployment hazırlığı.
- **Teknoloji/Algoritma:** `GridSearchCV`, `pickle`.
- **Hesaplama:** Best params, tuned CV score.

## Cell 25 (Code) — Learning curve
- **Ne yapar:** Train size arttıkça train/CV skorlarını çizer.
- **DS anlamı:** Bias-variance ve veri ölçeği etkisi analizi.
- **Teknoloji/Algoritma:** `sklearn.learning_curve`.
- **Hesaplama:** Train/CV mean/std, generalization gap.

## Cell 26 (Markdown) — Model comparison başlığı
- **Ne yapar:** Karşılaştırma bölümünü açar.
- **DS anlamı:** Model seçimi için çok-metrikli bakış.
- **Teknoloji/Algoritma:** Yok.
- **Hesaplama:** Yok.

## Cell 27 (Code) — Radar chart karşılaştırması
- **Ne yapar:** Modelleri 5 metrikte radar üzerinde karşılaştırır.
- **DS anlamı:** Çok boyutlu performans profilini tek görselde sunar.
- **Teknoloji/Algoritma:** `plotly.scatterpolar`.
- **Hesaplama:** Normalize metrik eksenlerinde model izleri.

## Cell 28 (Code) — Stilize sonuç tablosu
- **Ne yapar:** CV sonuçlarını sıralı/stilize tablo ile gösterir.
- **DS anlamı:** Final modeli şeffaf şekilde gerekçelendirme.
- **Teknoloji/Algoritma:** `pandas.style`.
- **Hesaplama:** En iyi model vurgusu, metrik formatlama.

## Cell 29 (Markdown) — Evaluation başlığı
- **Ne yapar:** Held-out test değerlendirmesini başlatır.
- **DS anlamı:** Gerçek genelleme performansı.
- **Teknoloji/Algoritma:** Yok.
- **Hesaplama:** Yok.

## Cell 30 (Code) — Tahmin + sınıflandırma raporu
- **Ne yapar:** Test seti prediction/probability ve classification report üretir.
- **DS anlamı:** Sınıf bazlı precision/recall/f1 analizi.
- **Teknoloji/Algoritma:** `classification_report`, `predict_proba`.
- **Hesaplama:** Sınıf metrikleri.

## Cell 31 (Code) — Confusion matrix
- **Ne yapar:** Confusion matrix’i görselleştirir ve sensitivity/specificity annot eder.
- **DS anlamı:** FP/FN maliyetini doğrudan görmek.
- **Teknoloji/Algoritma:** `confusion_matrix`, `seaborn.heatmap`.
- **Hesaplama:** TN, FP, FN, TP; recall/specificity.

## Cell 32 (Code) — ROC + PR eğrileri
- **Ne yapar:** ROC ve Precision-Recall eğrilerini aynı figürde verir.
- **DS anlamı:** Sınıf ayrıştırma + pozitif sınıf odaklı performans.
- **Teknoloji/Algoritma:** `roc_curve`, `roc_auc_score`, `precision_recall_curve`, `average_precision_score`.
- **Hesaplama:** AUC ve AP skorları.

## Cell 33 (Code) — Calibration curve
- **Ne yapar:** Olasılık kalibrasyonunu test eder.
- **DS anlamı:** Tahmin olasılığı güvenilir mi sorusunu cevaplar.
- **Teknoloji/Algoritma:** `sklearn.calibration_curve`.
- **Hesaplama:** Mean predicted prob vs observed fraction.

## Cell 34 (Markdown) — Calibration insight
- **Ne yapar:** Kalibrasyon eğrisinin anlamını açıklar.
- **DS anlamı:** Risk skorunun karar destek değerini anlatır.
- **Teknoloji/Algoritma:** Yok.
- **Hesaplama:** Yok.

## Cell 35 (Markdown) — SHAP başlığı
- **Ne yapar:** Explainability bölümünü açar.
- **DS anlamı:** Model kararlarının şeffaflaştırılması.
- **Teknoloji/Algoritma:** Yok.
- **Hesaplama:** Yok.

## Cell 36 (Code) — SHAP analizleri
- **Ne yapar:** SHAP summary, bar, top feature driverlarını üretir.
- **DS anlamı:** Global + lokal açıklanabilirlik.
- **Teknoloji/Algoritma:** `shap.LinearExplainer`/`TreeExplainer`.
- **Hesaplama:** SHAP değerleri, mean |SHAP| sıralaması.

## Cell 37 (Markdown) — SHAP yorum metni
- **Ne yapar:** SHAP bulgularını hikayeleştirir.
- **DS anlamı:** “Model ne öğrendi?” sorusunu cevaplar.
- **Teknoloji/Algoritma:** Yok.
- **Hesaplama:** Yok.

## Cell 38 (Markdown) — Error analysis başlığı
- **Ne yapar:** Hata analizi bölümünü başlatır.
- **DS anlamı:** Modelin zayıf kaldığı örüntüleri bulma.
- **Teknoloji/Algoritma:** Yok.
- **Hesaplama:** Yok.

## Cell 39 (Code) — FP/FN/TP/TN segment analizi
- **Ne yapar:** Hata tiplerini ayırır, profil ortalamaları çıkarır.
- **DS anlamı:** Nerede yanlış yaptığını profile bağlama.
- **Teknoloji/Algoritma:** Boolean masks + `pandas` profiling.
- **Hesaplama:** Outcome grupları için mean feature tablosu.

## Cell 40 (Code) — Outcome violin karşılaştırması
- **Ne yapar:** FP/FN/TP/TN için numeric dağılımları violin ile karşılaştırır.
- **DS anlamı:** Hata sınıflarının dağılımsal farkı.
- **Teknoloji/Algoritma:** `matplotlib.violinplot`.
- **Hesaplama:** Outcome-wise dağılım görselleştirmesi.

## Cell 41 (Code) — Threshold analizi
- **Ne yapar:** Eşik değeri değişince precision/recall/F1 nasıl değişiyor analiz eder.
- **DS anlamı:** Operasyonel karar eşik optimizasyonu.
- **Teknoloji/Algoritma:** `precision_score`, `recall_score`, `f1_score`, Plotly line chart.
- **Hesaplama:** Threshold sweep + best F1 threshold.

## Cell 42 (Markdown) — Final summary başlığı
- **Ne yapar:** Final sonuç bölümüne geçiş yapar.
- **DS anlamı:** Sonuçların yönetici özeti.
- **Teknoloji/Algoritma:** Yok.
- **Hesaplama:** Yok.

## Cell 43 (Code) — Sonuç dashboard + yazılı özet
- **Ne yapar:** KPI kartları ve text summary üretir.
- **DS anlamı:** Teknik metrikleri sunum diline çevirir.
- **Teknoloji/Algoritma:** `plotly.Indicator`, summary print.
- **Hesaplama:** Test/CV ana metriklerin raporlanması.

## Cell 44 (Markdown) — Fairness/Significance add-on başlığı
- **Ne yapar:** Gelişmiş metodoloji eklentilerini açar.
- **DS anlamı:** Checklist derinliği + akademik tamamlayıcılık.
- **Teknoloji/Algoritma:** Yok.
- **Hesaplama:** Yok.

## Cell 45 (Code) — PCA, Chi2, progressive sampling
- **Ne yapar:** PCA varyans, Chi-square feature ranking, progressive training-size kontrolü yapar.
- **DS anlamı:** Boyut indirgeme/feature relevance/doğrulama.
- **Teknoloji/Algoritma:** `PCA`, `SelectKBest(chi2)`, model clone ile fraksiyonel eğitim.
- **Hesaplama:** %95 varyans bileşen sayısı, chi2 top features, train fraction vs F1.

## Cell 46 (Code) — Imbalanced strateji karşılaştırması
- **Ne yapar:** Baseline, under, over, SMOTE senaryolarını test eder.
- **DS anlamı:** Dengesiz veri yönetim tekniklerinin etkisi.
- **Teknoloji/Algoritma:** `RandomUnderSampler`, `RandomOverSampler`, `SMOTE`.
- **Hesaplama:** Yöntem bazlı test F1-macro ve positive recall.

## Cell 47 (Code) — Fairness + significance (hızlı/güvenli)
- **Ne yapar:** Subgroup fairness snapshot, LR vs SVM significance (paired t-test + McNemar) yapar; eksik bağımlılıklar için fallback içerir.
- **DS anlamı:** Model adaleti + istatistiksel eşdeğerlik kontrolü.
- **Teknoloji/Algoritma:** `ttest_rel`, `mcnemar`, `SVC`, fallback feature prep.
- **Hesaplama:** subgroup F1/recall, p-değerleri.

## Cell 48 (Markdown) — Model card / etik / limitler
- **Ne yapar:** Amaçlanan kullanım, etik sınırlar, yanlış kullanım riskini yazar.
- **DS anlamı:** Responsible AI çerçevesi.
- **Teknoloji/Algoritma:** Yok.
- **Hesaplama:** Yok.

## Cell 49 (Markdown) — Gradio kullanım yönergesi
- **Ne yapar:** Demo arayüzünün nasıl çalıştırılacağını adım adım açıklar.
- **DS anlamı:** Teknik çıktının ürünleştirilmiş sunumu.
- **Teknoloji/Algoritma:** Yok.
- **Hesaplama:** Yok.

## Cell 50 (Code) — Gradio interaktif demo
- **Ne yapar:** Gerçek `preprocessor + rfe + best_model` ile kullanıcı profiline risk tahmini yapan profesyonel UI oluşturur.
- **DS anlamı:** Modeli karar destek aracına dönüştürme.
- **Teknoloji/Algoritma:** `gradio.Blocks`, custom CSS, preset senaryolar, gerçek inference pipeline.
- **Hesaplama:** Input -> feature transform -> model probability -> risk band + açıklama.

## Cell 51 (Markdown) — 10b başlığı
- **Ne yapar:** Checklist kapanış deneylerini açar.
- **DS anlamı:** Eksik checklist maddelerini tamamlayıcı analizlerle kapatma.
- **Teknoloji/Algoritma:** Yok.
- **Hesaplama:** Yok.

## Cell 52 (Code) — Aggregation + StandardScaler + SVD kıyas
- **Ne yapar:** Groupby aggregation, standardizasyon ek deneyi, TruncatedSVD deneyi ve karşılaştırma grafiği üretir.
- **DS anlamı:** Alternatif preprocessing/reduction kararlarının etkisini gösterir.
- **Teknoloji/Algoritma:** `groupby`, `StandardScaler`, `TruncatedSVD`, LR kıyas grafiği.
- **Hesaplama:** F1-macro karşılaştırmaları, explained variance.

## Cell 53 (Markdown) — Final checklist evidence başlığı
- **Ne yapar:** Checklist eşleme tablosuna giriş yapar.
- **DS anlamı:** Değerlendirme kriteri -> kanıt izlenebilirliği.
- **Teknoloji/Algoritma:** Yok.
- **Hesaplama:** Yok.

## Cell 54 (Code) — Evidence matrix
- **Ne yapar:** Hangi checklist maddesi kullanıldı/supplementary/N-A gerekçesiyle tablo halinde sunar.
- **DS anlamı:** Akademik savunmayı sistematikleştirir.
- **Teknoloji/Algoritma:** `pandas.DataFrame` + style.
- **Hesaplama:** Yok (kanıt eşleme).

---

## Sunumda kısa akış önerisi (2-3 dk teknik özet)
1. **DQA (Cell 5-7):** veri güvenilirliği ve temizlik kararları.  
2. **EDA (Cell 9-18):** hedefle en ilişkili sinyallerin keşfi.  
3. **Feature pipeline (Cell 20):** leakage-safe preprocessing + RFE.  
4. **Modelleme (Cell 23-25):** CV kıyas + tuning + learning curve.  
5. **Evaluation (Cell 30-41):** ROC/PR, calibration, threshold, error analysis.  
6. **Explainability/Fairness (Cell 36, 45-47):** SHAP + subgroup/istatistiksel test.  
7. **Productization (Cell 50):** Gradio ile gerçek inference demo.  

