
# Time Series Analysis

| Metric Sayısı | 🧭 Prophet | 🔍 Granger | Açıklama (Özet)                                                  | 🇬🇧 Example                                                  | 🇹🇷 Örnek                                               |
| ------------- | ---------- | ---------- | ---------------------------------------------------------------- | ------------------------------------------------------------- | -------------------------------------------------------- |
| 0             | ✅          | ❌          | Metrik belirtilmez → tüm metrikler için Prophet tahmini yapılır. | _Forecast overall campaign performance for the next 30 days._ | _Genel kampanya performansının 30 günlük tahminini yap._ |
| 1             | ✅          | ✅          | Tek metrik (örn. CTR) → Prophet + Granger birlikte çalışır.      | _Forecast CTR for the next 15 days._                          | _CTR metriğinin önümüzdeki 15 günlük tahminini yap._     |
| 2             | ❌          | ✅          | İki metrik varsa yalnızca Granger nedensellik testi yapılır.     | _Does spend Granger-cause CTR?_                               | _Spend metriği CTR’ı Granger anlamında etkiliyor mu?_    |
| 3+            | ❌          | ❌          | Fazla metrik varsa Prophet/Granger çalışmaz → uyarı döner.       | _Compare CTR, spend and revenue trends together._             | _CTR, spend ve revenue metriklerini birlikte analiz et._ |

---
# 🎯 **DoWhy / EconML Agent (dowEcon)**

**Amaç:** Statik veya panel veri üzerinde “treatment → outcome” nedensel etki analizi.  
**Zorunlu:** `campaign_name`, `ad_name`  
**Opsiyonel:** `treatment_time`  
**Anahtar kelimeler:** `effect of ... on ...`, `treatment`, `outcome`, `confounders`, `nedensel analiz`, `causal effect`

### 🟢 1️⃣ Basic English – classic DoWhy pattern

`Estimate the causal effect of SPEND on CTR  for campaign 'TR_purchase_meta_sales_maxconv_retargeting_apr2025_aicreatives'  and ad 'kubradeny' using [CPC, IMPRESSIONS] as confounders.`

➡️ Türkçe:

> Kampanya 'TR_purchase_meta_sales_maxconv_retargeting_apr2025_aicreatives' ve reklam 'kubradeny' için harcamanın (SPEND) CTR üzerindeki nedensel etkisini [CPC, IMPRESSIONS] değişkenlerini kontrol ederek analiz et.

---

### 🟢 2️⃣ English – explicit `at time`

`Analyze the effect of SPEND on CONVERSIONS  for campaign 'TR_purchase_meta_sales_maxconv_retargeting_apr2025_aicreatives'  and ad 'kubradeny' using [CTR, CPC] as confounders at time 12.`

➡️ Türkçe:

> Kampanya 'TR_purchase_meta_sales_maxconv_retargeting_apr2025_aicreatives' ve reklam 'kubradeny' için SPEND değişkeninin CONVERSIONS üzerindeki etkisini [CTR, CPC] değişkenlerini kontrol ederek zaman 12’de analiz et.

---

### 🟢 3️⃣ English – “using … as treatment … as outcome”

`Using SPEND as treatment and CTR as outcome,  estimate the causal effect with confounders [CPC, IMPRESSIONS]  for campaign 'TR_purchase_meta_sales_maxconv_retargeting_apr2025_aicreatives' and ad 'kubradeny'.`

➡️ Türkçe:

> SPEND değişkenini treatment, CTR değişkenini outcome olarak kullanarak; kampanya 'TR_purchase_meta_sales_maxconv_retargeting_apr2025_aicreatives' ve reklam 'kubradeny' için [CPC, IMPRESSIONS] kontrol değişkenleriyle nedensel etkiyi hesapla.

---

### 🟢 4️⃣ Turkish – doğal cümle yapısı

`Kampanya 'TR_purchase_meta_sales_maxconv_retargeting_apr2025_aicreatives'  ve reklam 'kubradeny' için harcamanın (SPEND) CTR üzerindeki nedensel etkisini  [CPC, IMPRESSIONS] değişkenlerini kontrol ederek incele.`

---

### 🟢 5️⃣ Turkish – farklı outcome

`Nedensel analiz ile kampanya 'TR_purchase_meta_sales_maxconv_retargeting_apr2025_aicreatives'  ve reklam 'kubradeny' için harcamanın (SPEND) satışlar (REVENUE) üzerindeki etkisini  [CPC, CTR] değişkenlerini kontrol ederek analiz et.`

---

✅ **Beklenen parse çıktısı (örnek)**

`{   "treatment": "spend",   "outcome": "ctr",   "confounders": ["cpc", "impressions"],   "campaign_name": "TR_purchase_meta_sales_maxconv_retargeting_apr2025_aicreatives",   "ad_name": "kubradeny",   "treatment_time": null }`


---
# 📊 **CausalPy Agent (Bayesian / Synthetic Control)**

**Amaç:** Zaman serisi tabanlı nedensel etki / uplift analizi.  
**Zorunlu:** `outcome`, `predictors`, `campaign_name`, `ad_name`, `treatment_time`  
**Anahtar kelimeler:** `causal impact`, `causal analysis on`, `using ... as outcome`, `with predictors`, `at time`, `Bayesian`, `synthetic control`

### 🔵 1️⃣ English – canonical pattern

`Using CTR as outcome,  analyze the causal impact with predictors [SPEND, IMPRESSIONS]  for campaign 'TR_purchase_meta_sales_maxconv_retargeting_apr2025_aicreatives'  and ad 'kubradeny' at time 20.`

➡️ Türkçe:

> Kampanya 'TR_purchase_meta_sales_maxconv_retargeting_apr2025_aicreatives' ve reklam 'kubradeny' için CTR metriğini outcome olarak al, [SPEND, IMPRESSIONS] predictor değişkenleriyle zaman 20’de nedensel etkiyi analiz et.

---

### 🔵 2️⃣ English – “causal analysis on … using predictors …”

`Causal analysis on REVENUE  using [SPEND, CTR, CPC] predictors  for campaign 'TR_purchase_meta_sales_maxconv_retargeting_apr2025_aicreatives'  and ad 'kubradeny' at time 30.`

➡️ Türkçe:

> Kampanya 'TR_purchase_meta_sales_maxconv_retargeting_apr2025_aicreatives' ve reklam 'kubradeny' için REVENUE metriği üzerinde, [SPEND, CTR, CPC] predictor değişkenlerini kullanarak zaman 30’da nedensel analiz yap.

---

### 🔵 3️⃣ English – “effect on … with predictors … time …”

`Effect on CONVERSIONS  with predictors [SPEND, CTR, CPC]  for campaign 'TR_purchase_meta_sales_maxconv_retargeting_apr2025_aicreatives'  and ad 'kubradeny' time 15.`

➡️ Türkçe:

> Kampanya 'TR_purchase_meta_sales_maxconv_retargeting_apr2025_aicreatives' ve reklam 'kubradeny' için CONVERSIONS üzerindeki etkiyi [SPEND, CTR, CPC] predictor’larıyla zaman 15’te analiz et.

---

### 🔵 4️⃣ Turkish – Bayesian vurgulu

`Bayes analizi ile kampanya 'TR_purchase_meta_sales_maxconv_retargeting_apr2025_aicreatives'  ve reklam 'kubradeny' için REVENUE metriğini bağımlı değişken olarak al,  [SPEND, CTR, CPC] predictor’larıyla zaman 25’te etkiyi ölç.`

---

### 🔵 5️⃣ Turkish – minimalist doğal ifade

`Kampanya 'TR_purchase_meta_sales_maxconv_retargeting_apr2025_aicreatives' ve reklam 'kubradeny' için  CTR metriğini outcome olarak al, [SPEND, IMPRESSIONS] predictor değişkenleriyle zaman 10’da Bayesian nedensel etki analizi yap.`

---

✅ **Beklenen parse çıktısı (örnek)**

`{   "outcome": "ctr",   "predictors": ["spend", "impressions"],   "campaign_name": "TR_purchase_meta_sales_maxconv_retargeting_apr2025_aicreatives",   "ad_name": "kubradeny",   "treatment_time": 20 }`

|Özellik|DoWhy (dowEcon)|CausalPy|
|---|---|---|
|🎯 Amaç|Statik / panel veri üzerinde nedensel ilişki|Zaman serisi üzerinde nedensel etki (öncesi-sonrası)|
|🔑 Anahtar kelime|“effect of … on …”, “treatment”, “outcome”, “confounders”|“causal impact”, “causal analysis on …”, “using … as outcome”, “with predictors”, “at time”|
|🕒 Zaman|Opsiyonel|🔥 Zorunlu|
|📦 Parser|`parse_dowhy_query()`|`parse_causalpy_query()`|
|🧩 Regex tetikleyicisi|`effect\s+of\s+(\w+)\s+on\s+(\w+)`|`causal\s+(impact|
|📊 Örnek|_Effect of SPEND on CTR using [CPC, IMPRESSIONS]_|_Using CTR as outcome with predictors [SPEND, IMPRESSIONS] at time 20_|

---

# 🧩 **Association Rules**

## 1️⃣ **GENEL ANALİZ (Campaign-level, entity yok)**

> 🔹 Tüm kampanyaları birlikte değerlendirir.  
> 🔹 `level = campaign_name`, `entity = None`

**🇹🇷 Türkçe**

- "Genel kampanya düzeyinde birliktelik kurallarını çıkar."
    
- "Kampanya bazında en güçlü 25 birliktelik kuralını göster."
    
- "Tüm kampanyalar için 0.05 destek ve 0.6 güven eşiğiyle birliktelik kuralları oluştur."
    
- "Kampanya düzeyinde lift 1.5’in üzerindeki kuralları çıkar."
    

**🇬🇧 English**

- "Generate association rules at the campaign level."
    
- "Show the top 25 association rules across all campaigns."
    
- "Mine rules for all campaigns with support 0.05 and confidence 0.6."
    
- "Find campaign-level rules with lift above 1.5."
    

---

## 2️⃣ **BELİRLİ ENTITY (örnek: Kampanya ismi belirtilmiş)**

> 🔹 Sadece belirtilen kampanya için kurallar çıkarılır.  
> 🔹 `entity = "Nike"`, `level = campaign_name`

**🇹🇷 Türkçe**

- "Nike kampanyası için birliktelik kurallarını göster."
    
- "Adidas kampanyasında lift değeri 1.3’ten yüksek kuralları çıkar."
    
- "‘Summer Sale’ kampanyası için 0.04 destek ve 0.7 güven ile birliktelik kurallarını üret."
    
- "‘Black Friday’ kampanyası için kampanya düzeyinde kuralları çıkar."
    

**🇬🇧 English**

- "Show association rules for the Nike campaign."
    
- "Generate association rules for the Adidas campaign with lift > 1.3."
    
- "For the 'Summer Sale' campaign, use min support 0.04 and confidence 0.7."
    
- "Extract campaign-level association rules for the 'Black Friday' campaign."
    

---

## 3️⃣ **ADSET DÜZEYİNDE ANALİZ**

> 🔹 Reklam seti (adset) seviyesinde kurallar çıkarılır.  
> 🔹 `level = adset_name`

**🇹🇷 Türkçe**

- "Ad set düzeyinde birliktelik kurallarını çıkar."
    
- "‘Promo Audience’ reklam seti için 0.03 destek, 0.6 güven ile kuralları bul."
    
- "Ad set bazında lift 1.4 üzerindeki ilişkileri göster."
    

**🇬🇧 English**

- "Generate association rules at the adset level."
    
- "Find rules for the 'Promo Audience' ad set with support 0.03 and confidence 0.6."
    
- "List adset-level rules with lift greater than 1.4."
    

---

## 4️⃣ **AD (Creative) DÜZEYİNDE ANALİZ**

> 🔹 Belirli bir reklam (creative/ad) için kurallar.  
> 🔹 `level = ad_name`

**🇹🇷 Türkçe**

- "Reklam düzeyinde birliktelik kurallarını çıkar."
    
- "‘Creative A’ reklamı için en yüksek lift değerli kuralları göster."
    
- "Reklam bazında 0.05 destek, 0.65 güven ile FP-Growth algoritması kullan."
    

**🇬🇧 English**

- "Generate association rules at the ad level."
    
- "Show rules with the highest lift for the 'Creative A' ad."
    
- "At the ad level, use FP-Growth algorithm with support 0.05 and confidence 0.65."
    

---

## 5️⃣ **PARAMETRELİ SORGULAR (support / confidence / lift / max_len)**

> 🔹 Sayısal parametreler belirlenir.  
> 🔹 `_find_float` ve `_find_int` ile parse edilir.

**🇹🇷 Türkçe**

- "Kampanya düzeyinde min destek 3%, güven 0.7, lift 1.2, max len 4 olsun."
    
- "0.04 destek, 0.75 güven eşiğiyle FP-Growth algoritmasını kullanarak kuralları çıkar."
    
- "Apriori algoritmasıyla lift 1.5 üzerinde ve maksimum itemset uzunluğu 3 olan kuralları üret."
    

**🇬🇧 English**

- "At the campaign level, use support 3%, confidence 0.7, lift 1.2, and max len 4."
    
- "Mine rules using FP-Growth with min support 0.04 and confidence 0.75."
    
- "Use Apriori algorithm, lift above 1.5, and maximum itemset length 3."
    

---

## 6️⃣ **GROUPWISE (Segment bazlı binning)**

> 🔹 Her kampanya/adset için ayrı kurallar oluşturur.  
> 🔹 `groupwise_bins=True`

**🇹🇷 Türkçe**

- "Her kampanya için ayrı (groupwise) birliktelik kurallarını üret."
    
- "Segment bazında (adset başına) 0.05 destek ve 0.6 güven ile kuralları bul."
    

**🇬🇧 English**

- "Generate groupwise association rules per campaign."
    
- "For each ad set (segment-wise), mine rules with support 0.05 and confidence 0.6."
    

---

## 7️⃣ **KARMA (Algoritma + Entity + Seviye + Parametre birlikte)**

> 🔹 Gerçek senaryo testleri için tam kombinasyon.  
> 🔹 `guess_level_from_text`, `guess_entity_from_text`, `_find_algo`, `_find_float` hepsi devrede.

**🇹🇷 Türkçe**

- "‘Holiday Promo’ kampanyası için adset düzeyinde FP-Growth algoritmasıyla, 0.04 destek, 0.7 güven, lift 1.3 parametreleriyle kuralları çıkar."
    
- "‘Creative A’ reklamı için Apriori algoritmasını kullan, destek %5, güven 0.6, max len 3 olsun."
    

**🇬🇧 English**

- "For the 'Holiday Promo' campaign at adset level, use FP-Growth algorithm with support 0.04, confidence 0.7, and lift 1.3."
    
- "Use Apriori algorithm for the 'Creative A' ad with support 5%, confidence 0.6, and max len 3."
    

---

## 8️⃣ **TÜRKÇE–İNGİLİZCE KARIŞIK (Robust Parsing Testi)**

> 🔹 Çok dilli sorguların parse dayanıklılığını test eder.  
> 🔹 LLM + regex parsing uyumluluğunu doğrular.

**Mixed Queries**

- "FP-Growth algoritmasıyla 0.05 support ve 0.7 confidence değerleriyle campaign level rules çıkar."
    
- "Find association rules for 'Promo Adset' adset düzeyinde, min_support=0.03, lift>1.4."

## 🧠 **Parsing Mantığı Özeti**

|Özellik|Fonksiyon|Algıladığı Şey|Örnek|
|---|---|---|---|
|**Level**|`guess_level_from_text`|campaign / adset / ad|“adset düzeyinde kurallar” → `adset_name`|
|**Entity**|`guess_entity_from_text`|spesifik kampanya/ad/adset|“Nike kampanyası için ...” → `Nike`|
|**Parametreler**|`_find_float`, `_find_int`|support, confidence, lift, max len|“0.04 destek, 0.7 güven”|
|**Algoritma**|`_find_algo`|apriori / fpgrowth / eclat|“FP-Growth algoritmasıyla”|
|**Groupwise**|`_find_groupwise`|True / False|“Her kampanya için ayrı (groupwise)”|

---

# 🧩 **SHAP Agent**

## 🔍 **Temel Mantık**

`extract_shap_target_column()` fonksiyonu, kullanıcı sorgusundaki metrik adını otomatik olarak SHAP hedefi (`target_var`) olarak seçer.

`for col in available_columns:     if col.lower() in query:         return col`

🔹 Yani query içinde `CTR`, `Revenue`, `Spend`, `Conversions`, `Clicks`, `CPC` gibi bir **kolon adı** geçerse o metrik SHAP analizi hedefi olur.  
🔹 Eğer hiçbir kolon adı geçmezse, fonksiyon `None` döner → analiz başlamaz.

---

## ⚙️ **Analiz Akışı**

1️⃣ **Query parsing:**  
→ “CTR” gibi bir metrik yakalanır → `target_var = "CTR"`

2️⃣ **Model eğitimi:**  
→ `getBestModel(cleaned_df, target_var)` çağrılır → en iyi tahmin modeli eğitilir

3️⃣ **SHAP analizi:**  
→ `shap_summary.png` grafiği + SHAP JSON çıktısı oluşturulur

4️⃣ **LLM yorumu:**  
→ `shap_agent()` çalışır → teknik + işsel yorumlar üretilir (dual-layer report)

---

## 🧠 **Nasıl Çalışır**

|Durum|Sistem davranışı|Kod karşılığı|
|---|---|---|
|Query bir metrik içeriyor|✅ Model eğitilir + SHAP analizi üretilir|`getBestModel()` + `shap_agent()`|
|Query’de hiçbir metrik yok|❌ Hedef kolon bulunamaz → hata mesajı döner|`extract_shap_target_column() → None`|

---

## 🧩 **Türkçe Query Örnekleri**

### 🎯 **Genel Performans Açıklamaları**

- "CTR değerini etkileyen en önemli faktörleri analiz et."
    
- "Revenue değişkenine en çok hangi değişkenler etki ediyor?"
    
- "Spend metriği için SHAP analizi yap, hangi özellikler öne çıkıyor?"
    
- "Conversions’ı etkileyen faktörleri SHAP değerlerine göre sırala."
    
- "CPC değişkeni üzerinde en büyük etkiye sahip değişkenleri bul."
    
- "Clicks değerine göre hangi değişkenler pozitif ya da negatif etki yapıyor?"
    
- "Revenue için SHAP feature importance analizi yap."
    
- "CTR tahmininde hangi değişkenler modelin kararını en çok etkiledi?"
    
- "Spend metriği için modelin açıklanabilirliğini SHAP ile incele."
    
- "Conversions metriğine göre SHAP grafiğini oluştur."
    

### 🔍 **Model Değerlendirme / Karşılaştırma Odaklı**

- "CTR tahmininde model performansını ve SHAP önem derecelerini raporla."
    
- "Revenue için SHAP özet grafiği çıkar, hangi değişkenler baskın?"
    
- "Spend tahmininde SHAP değerlerine göre pozitif/negatif etki yapan değişkenleri açıkla."
    
- "Conversions metriğini en çok etkileyen faktörleri görselleştir."
    

---

## 🧩 **English Query Examples**

### 🎯 **Standard SHAP Analysis**

- "Run SHAP analysis for CTR."
    
- "Which features most strongly affect Revenue?"
    
- "Analyze SHAP feature importance for Spend metric."
    
- "Show which variables drive Conversions up or down."
    
- "Perform SHAP analysis on CPC variable."
    
- "What are the top SHAP factors influencing Clicks?"
    
- "Explain feature impact on CTR predictions."
    
- "Generate SHAP summary for Spend."
    
- "Identify top positive and negative contributors for Revenue."
    
- "Run interpretability analysis for Conversions using SHAP."
    

### 🧮 **Model + SHAP Together**

- "Explain the SHAP results for CTR model, include performance metrics."
    
- "Which features have the strongest SHAP importance for Revenue?"
    
- "Show SHAP summary plot for CPC and discuss model reliability."
    
- "Analyze SHAP feature rankings for Spend prediction model."
    
- "Report top 10 SHAP important features for Conversions."
    

---

## 🧾 **Amaç Bazlı Tablo**

|🎯 Amaç|🇹🇷 Örnek Query|🇬🇧 English Equivalent|
|---|---|---|
|CTR etkileyen faktörleri bul|"CTR değerini etkileyen değişkenleri analiz et."|"Analyze which features impact CTR the most."|
|Revenue tahmini SHAP analizi|"Revenue metriği için SHAP analizi yap."|"Run SHAP analysis for Revenue."|
|Spend açıklanabilirliği|"Spend değerine göre SHAP önem derecelerini çıkar."|"Generate SHAP importance summary for Spend."|
|Conversions yorumu|"Conversions üzerindeki en önemli faktörleri sırala."|"Explain SHAP results for Conversions metric."|
|CPC içgörüleri|"CPC metriği için SHAP feature importance analiz et."|"Perform SHAP analysis on CPC."|

---

## 📊 **Özet Tablo**

|Durum|Hedef metrik|Sistem çıktısı|
|---|---|---|
|✅ Query “CTR”, “Revenue”, “Spend”, “CPC”, “Clicks”, “Conversions” içeriyor|`target_var` tespit edilir|Model eğitilir → `shap_summary.png` + JSON + LLM raporu|
|❌ Query’de hiçbir metrik geçmiyor|`target_var = None`|Hata mesajı: “No valid target column found for SHAP analysis.”|

---

## 💡 **Neden Bu Örnekler Uygun?**

- Her sorgu **available_columns** listesindeki metriklerden en az birini içeriyor.
    
- Böylece `extract_shap_target_column()` fonksiyonu doğrudan doğruya hedefi buluyor.
    
- `getBestModel()` + `shap_agent()` birlikte çalışarak hem teknik (SHAP values) hem işsel (LLM insight) yorum üretiyor.
    

---

# 🧮 **EDA Agent**


> 🎯 Amaç: İki değişken arasındaki **istatistiksel ilişkiyi veya farkı** test eder.  
> Türüne göre otomatik olarak uygun yöntem kullanılır:
> 
> - Numeric–Numeric → korelasyon testleri (Pearson, Spearman, Kendall)
>     
> - Numeric–Categorical → ANOVA, Kruskal–Wallis
>     
> - Categorical–Categorical → Khi-kare, Cramér’s V
>     

---

### 🇬🇧 **English EDA Queries**

- "Is there a relationship between CTR and Spend?"
    
- "Check the correlation between Revenue and CPC."
    
- "Compare Clicks and Impressions — are they statistically related?"
    
- "Do different campaign types affect CTR?"
    
- "Is there a significant difference in Revenue across Regions?"
    
- "Does CPC vary by campaign type?"
    
- "Are high CTR campaigns also high in Conversions?"
    
- "Is there any association between Device Type and Clicks?"
    
- "Does Spend correlate with Conversions?"
    
- "Are Revenue and Spend positively correlated?"
    

---

### 🇹🇷 **Türkçe EDA Sorguları**

- "CTR ile Spend arasında ilişki var mı?"
    
- "Revenue ile CPC arasındaki korelasyonu kontrol et."
    
- "Clicks ve Impressions arasında istatistiksel bir ilişki var mı?"
    
- "Kampanya tipi CTR üzerinde etkili mi?"
    
- "Farklı bölgeler arasında Revenue farkı anlamlı mı?"
    
- "CPC kampanya türüne göre değişiyor mu?"
    
- "Yüksek CTR kampanyaları daha fazla Conversions üretiyor mu?"
    
- "Cihaz tipi ile Clicks arasında anlamlı bir ilişki var mı?"
    
- "Spend ile Conversions arasında ilişki var mı?"
    
- "Revenue ve Spend pozitif korelasyon gösteriyor mu?"

---

# 📊 **Plot Agent**

> 🎨 Amaç: İki metrik arasındaki ilişkiyi **grafikle gösterir.**  
> Türüne göre otomatik grafik seçimi yapılır:
> 
> - Numeric–Numeric → Scatterplot, Regression Line
>     
> - Numeric–Categorical → Boxplot / Violin plot
>     
> - Categorical–Categorical → Mosaic / Count plot
>     

---

### 🇬🇧 **English Plot Queries**

- "Plot CTR and Spend."
    
- "Show the relationship between Revenue and CPC."
    
- "Visualize Clicks vs Impressions."
    
- "Display Spend vs Conversions chart."
    
- "Plot CTR against Campaign Type."
    
- "Show boxplot of Revenue by Region."
    
- "Create scatter plot for CTR and CPC."
    
- "Plot Conversions versus Spend."
    
- "Visualize how Revenue changes with Impressions."
    
- "Show chart comparing CTR and Device Type."
    

---

### 🇹🇷 **Türkçe Plot Sorguları**

- "CTR ve Spend grafiğini çiz."
    
- "Revenue ile CPC arasındaki ilişkiyi göster."
    
- "Clicks ve Impressions için bir görselleştirme oluştur."
    
- "Spend ile Conversions grafiğini göster."
    
- "CTR’yi kampanya türüne göre çiz."
    
- "Bölgelere göre Revenue boxplot’unu oluştur."
    
- "CTR ile CPC arasındaki scatter plot’u oluştur."
    
- "Conversions ve Spend arasındaki ilişkiyi görselleştir."
    
- "Revenue’nun Impressions’a göre nasıl değiştiğini göster."
    
- "CTR ve cihaz türü arasındaki farkları çiz."

---
# 🌍 **Wiki Agent**

## 🧩 **Türkçe Query Örnekleri**

### 🎓 Genel Kavramlar

- "CTR nedir?"
    
- "ROAS ne demektir?"
    
- "Varyans nasıl hesaplanır?"
    
- "Korelasyon katsayısı nedir?"
    
- "Entropi kavramını açıkla."
    
- "Makine öğrenmesi nedir?"
    
- "Lineer regresyon nasıl çalışır?"
    
- "Derin öğrenme ve sinir ağları arasındaki fark nedir?"
    
- "Bayes teoremi nedir?"
    
- "Standard sapma nasıl bulunur?"
    

### 🧮 Matematiksel / Bilimsel Sorgular

- "E=mc² formülü ne anlama geliyor?"
    
- "Pythagoras teoremi nedir?"
    
- "Normal dağılım eğrisi neyi ifade eder?"
    
- "T-testi nedir ve nasıl kullanılır?"
    
- "Gradient Descent nasıl çalışır?"
    
- "Z skorunun anlamı nedir?"
    
- "Monte Carlo simülasyonu nedir?"
    
- "R^2 istatistiği nasıl yorumlanır?"
    

### 🌐 Tarih / Genel Kültür

- "Aristoteles kimdir?"
    
- "Büyük Patlama teorisi nedir?"
    
- "İkinci Dünya Savaşı ne zaman başladı?"
    
- "Kuantum fiziği neyi inceler?"
    
- "DNA kim tarafından keşfedildi?"
    

---

## 🧩 **English Query Examples**

### 🎓 General Concepts

- "What is CTR?"
    
- "Define entropy."
    
- "What does ROAS mean?"
    
- "Explain variance and standard deviation."
    
- "What is correlation coefficient?"
    
- "What is machine learning?"
    
- "Describe linear regression."
    
- "What is the difference between deep learning and neural networks?"
    
- "What does Bayes’ theorem state?"
    
- "How is standard deviation calculated?"
    

### 🧮 Math / Science Queries

- "Explain the formula E = mc^2."
    
- "What is the Pythagorean theorem?"
    
- "Describe the normal distribution curve."
    
- "What is a t-test and when is it used?"
    
- "How does gradient descent work?"
    
- "Interpret the R-squared statistic."
    
- "What is Monte Carlo simulation?"
    

### 🌐 History / Culture

- "Who was Aristotle?"
    
- "What is the Big Bang theory?"
    
- "When did World War II begin?"
    
- "What does quantum physics study?"
    
- "Who discovered DNA?"

---
# 🧪 A/B TEST TOOLKIT 

_(Türkçe + İngilizce tam set — tüm 5 method tipi için)_

---

## 1️⃣ **basic_ab_test**

💡 _Anahtar kelimeler:_ `"top"`, `"best"`, `"rank"`, `"sorted"`, `"show"`, `"what is"`, `"value of"`

📘 _Amaç:_  
Belirli bir seviyede (genellikle `adset_name`) metriklerin sıralı özetini gösterir.  
Yani basit "kim daha iyi?" türü sorular.

---

### 🇹🇷 Türkçe

- "En yüksek CTR’a sahip kampanyalar hangileri?"
    
- "Reklam setlerini CTR değerine göre sırala."
    
- "En iyi performans gösteren reklam setlerini göster."
    
- "CTR değeri en yüksek 10 reklam setini listele."
    
- "Hangi kampanya en düşük CPC’ye sahip?"
    
- "Top 5 kampanyayı CTR açısından sırala."
    
- "Reklamları CTR’a göre sıralar mısın?"
    
- "CTR değerine göre sıralanmış kampanyaları göster."
    
- "En iyi performanslı creative’leri göster."
    

### 🇬🇧 English

- "Show the top campaigns by CTR."
    
- "Which ad sets have the highest CTR?"
    
- "Rank ads by CPC value."
    
- "List the best performing ad sets."
    
- "Show the top 10 campaigns sorted by CTR."
    
- "Which campaign has the lowest CPC?"
    
- "Rank all ads by performance metric CTR."
    
- "Show campaigns ranked by CTR value."
    

---

## 2️⃣ **specific_entity_ab_test**

💡 _Anahtar kelimeler:_ tek bir entity içeren sorgular (ör. “Nike campaign”), _karşılaştırma kelimesi yok_ (`compare`, `vs` geçmez).

📘 _Amaç:_  
Belirli bir **entity’yi (ör. tek kampanya/adset/ad)** diğer tüm sistemle karşılaştırır (“vs all others”).

---

### 🇹🇷 Türkçe

- "Nike kampanyasının CTR performansını test et."
    
- "Promo Set reklam setinin CTR’si diğerlerinden farklı mı?"
    
- "Creative A reklamının CPC değeri diğer reklamlardan anlamlı şekilde farklı mı?"
    
- "Campaign Alpha kampanyasının CTR’ı diğer kampanyalarla karşılaştır."
    
- "Kampanya X diğerlerinden daha iyi performans gösteriyor mu?"
    
- "Reklam seti Y’nin CTR ortalaması diğerlerinden yüksek mi?"
    

### 🇬🇧 English

- "Test CTR performance of Nike campaign against others."
    
- "Is the CTR of Promo Set adset significantly different from others?"
    
- "Compare CTR of Campaign Alpha with all other campaigns."
    
- "Is the CPC of Ad Creative A higher than the rest?"
    
- "Does Campaign X perform better than all others?"
    
- "Evaluate whether Adset Y’s CTR differs from the rest."
    

---

## 3️⃣ **adaptive_comparison_test**

💡 _Anahtar kelimeler:_ `"compare"`, `"versus"`, `"vs"`, `"with"`

📘 _Amaç:_  
İki grup (ör. iki kampanya, iki reklam seti, iki reklam) arasında karşılaştırmalı test yapar.

---

### 🇹🇷 Türkçe

- "Nike kampanyasını Adidas kampanyasıyla karşılaştır."
    
- "‘Promo Set’ reklam setini ‘Control Set’ ile karşılaştır."
    
- "Ad A ve Ad B’nin CTR farkı anlamlı mı?"
    
- "Creative 1 vs Creative 2 için CTR karşılaştırması yap."
    
- "Campaign Alpha’yı Campaign Beta ile karşılaştır, metrik CTR."
    
- "Kampanya X ve Kampanya Y arasında anlamlı fark var mı?"
    
- "Adset A ve Adset B’nin CPC farkını test et."
    
- "Compare Kampanya A vs Kampanya B performans."
    

### 🇬🇧 English

- "Compare Nike campaign with Adidas campaign by CTR."
    
- "Compare 'Promo Set' ad set vs 'Control Set'."
    
- "Is there a significant difference between Ad A and Ad B CTR?"
    
- "Compare Campaign Alpha versus Campaign Beta."
    
- "Test CTR difference between Adset A and Adset B."
    
- "Perform A/B test comparing two campaigns: Alpha vs Beta."
    
- "Compare CPC between Creative 1 and Creative 2."
    
- "Campaign X vs Campaign Y — which performs better?"
    

---

## 4️⃣ **chi_square_test**

💡 _Anahtar kelimeler:_ `"compare all"`, `"overall"`, `"distribution"`, `"across"`, `"differences between all"`

📘 _Amaç:_  
Tüm gruplar arasında (ör. tüm kampanyalar) oransal fark olup olmadığını test eder.  
Genellikle “hepsini karşılaştır” gibi sorgular tetikler.

---

### 🇹🇷 Türkçe

- "Tüm kampanyalar arasındaki CTR farkını test et."
    
- "Tüm reklam setleri arasında anlamlı fark var mı?"
    
- "Kampanyalar genelinde CTR dağılımı anlamlı şekilde farklı mı?"
    
- "CTR farklarını tüm reklamlar arasında test et."
    
- "Reklamlar arasında genel performans farkı var mı?"
    

### 🇬🇧 English

- "Compare all campaigns for CTR differences."
    
- "Test whether CTR differs significantly across all ad sets."
    
- "Perform chi-square test for CTR distribution across campaigns."
    
- "Check if overall ad performance differs among all campaigns."
    
- "Is there a significant difference in CTR across all creatives?"
    

---

## 5️⃣ **advanced_ab_test**

💡 _Anahtar kelimeler:_ `"does"`, `"impact"`, `"effect"`, `"affect"`  
📘 _Amaç:_  
Bir veya birden fazla faktörün metrik üzerindeki etkisini inceler (ANOVA).

---

### 🇹🇷 Türkçe

- "Adset tipi CTR üzerinde etkili mi?"
    
- "Campaign type CTR’ı etkiliyor mu?"
    
- "Creative türü CTR üzerinde anlamlı bir etki yaratıyor mu?"
    
- "Harcama miktarı CTR’ı etkiler mi?"
    
- "Kampanya türü ve hedef kitle etkisi CTR üzerinde anlamlı mı?"
    
- "Reklam formatı ve platform etkileri CTR’ı nasıl etkiliyor?"
    
- "Adset segmenti CTR değerini etkiler mi?"
    
- "CPC üzerinde adset tipi ve platformun etkisini test et."
    

### 🇬🇧 English

- "Does ad set type have an impact on CTR?"
    
- "Is campaign type affecting CTR?"
    
- "Test the effect of creative type on CTR."
    
- "Does spend amount impact CTR?"
    
- "Evaluate the effect of campaign type and audience on CTR."
    
- "How do ad format and platform affect CTR?"
    
- "Test whether targeting or placement has a significant effect on CPC."
    
- "What factors impact CTR the most?"
    

---

# 🔍 Özet Tablo

| Method                       | Anahtar kelimeler              | Türkçe örnek                                | İngilizce örnek                              |
| ---------------------------- | ------------------------------ | ------------------------------------------- | -------------------------------------------- |
| **basic_ab_test**            | top, best, rank, show, what is | “En iyi 10 kampanyayı göster.”              | “Show top 10 campaigns by CTR.”              |
| **specific_entity_ab_test**  | tek entity, no compare         | “Nike kampanyası diğerlerinden farklı mı?”  | “Compare CTR of Nike campaign vs others.”    |
| **adaptive_comparison_test** | compare, versus, vs            | “Kampanya A’yı Kampanya B ile karşılaştır.” | “Compare Campaign A vs Campaign B.”          |
| **chi_square_test**          | all, overall, across           | “Tüm kampanyalar arasındaki farkı test et.” | “Compare all campaigns for CTR differences.” |
| **advanced_ab_test**         | impact, effect, affect, does   | “Adset tipi CTR’ı etkiler mi?”              | “Does campaign type affect CTR?”             |

---

## `analyze` — Agent, Trends & Behavior
- “How did **revenue** evolve over time? Any notable trends or spikes?”
- “Analyze performance changes for **CTR** last month.”
- “What explains the drop in **conversions** last week?”
- “Investigate anomalies in **cost_per_acquisition**.”

### 🇬🇧 English Queries

| 🧠 Category                    | 💬 Example Query                                          | 🔍 What It Does                                 |
| ------------------------------ | --------------------------------------------------------- | ----------------------------------------------- |
| **Basic Summary**              | What are the columns in the dataset and their data types? | Returns `df.info()` summary.                    |
|                                | Show dataset summary statistics.                          | Runs `df.describe()`.                           |
|                                | List all available columns.                               | Lists `df.columns`.                             |
| **Averages & Totals**          | What is the average CTR?                                  | Computes mean of `CTR` column.                  |
|                                | Calculate total spend across all campaigns.               | Runs `df["spend"].sum()`.                       |
|                                | Find the median CPC.                                      | Uses `df["CPC"].median()`.                      |
| **Distributions**              | Describe the distribution of impressions.                 | Runs `df["impressions"].describe()`.            |
|                                | Show value counts for device_type.                        | Uses `df["device_type"].value_counts()`.        |
|                                | Plot a histogram of conversion rates.                     | Runs `df["conversion_rate"].plot(kind="hist")`. |
| **Trends & Evolution**         | Plot CTR trend over time.                                 | Uses line plot over time column.                |
|                                | Show how spend changed over the last 30 days.             | Calculates and visualizes daily change.         |
|                                | Detect which metric shows the strongest increasing trend. | Analyzes numeric column slopes.                 |
| **Comparisons & Correlations** | Compute correlation between spend and CTR.                | Uses `df[["spend","CTR"]].corr()`.              |
|                                | Which columns are most correlated with ROAS?              | Calculates correlation matrix.                  |
| **Missing & Data Quality**     | How many missing values are there per column?             | Uses `df.isna().sum()`.                         |
|                                | Show percentage of missing data per column.               | Computes `(df.isna().mean()*100)`.              |
|                                | Identify columns with constant values.                    | Detects columns where `df[col].nunique() == 1`. |

### 🇹🇷 Türkçe Sorgular

| 🧠 Kategori                       | 💬 Sorgu Örneği                                      | 🔍 Ne Yapar                                  |
| --------------------------------- | ---------------------------------------------------- | -------------------------------------------- |
| **Temel Özet**                    | Veri setindeki sütunlar ve veri tipleri nelerdir?    | `df.info()` çıktısını verir.                 |
|                                   | Veri setinin genel özet istatistiklerini göster.     | `df.describe()` çalıştırır.                  |
|                                   | Tüm sütun adlarını listele.                          | `df.columns` döndürür.                       |
| **Ortalama & Toplam**             | CTR ortalaması nedir?                                | `df["CTR"].mean()` hesaplar.                 |
|                                   | Kampanyalar arasındaki toplam harcamayı (spend) bul. | `df["spend"].sum()` döndürür.                |
|                                   | CPC değerinin medyanını göster.                      | `df["CPC"].median()` hesaplar.               |
| **Dağılımlar**                    | Impression sütununun dağılımını açıkla.              | `df["impressions"].describe()` çalıştırır.   |
|                                   | Cihaz türlerine göre kaç satır var?                  | `df["device_type"].value_counts()` döndürür. |
|                                   | Conversion rate histogramını çiz.                    | `df["conversion_rate"].plot(kind="hist")`.   |
| **Trendler & Değişimler**         | CTR zaman içindeki trendini çiz.                     | Zaman serisi grafiği oluşturur.              |
|                                   | Spend son 30 günde nasıl değişti?                    | Günlük fark ve trend hesaplar.               |
|                                   | Hangi metrik yükselen bir eğilim gösteriyor?         | Eğilim analizi yapar.                        |
| **Korelasyon & Karşılaştırmalar** | Spend ile CTR arasındaki korelasyonu hesapla.        | `df[["spend","CTR"]].corr()` çalıştırır.     |
|                                   | ROAS ile en yüksek korelasyona sahip sütun hangisi?  | Korelasyon matrisi hesaplar.                 |
| **Eksik Veri & Kalite**           | Her sütunda kaç eksik değer var?                     | `df.isna().sum()` döndürür.                  |
|                                   | Sütunlardaki eksik veri yüzdesini göster.            | `(df.isna().mean()*100)` hesaplar.           |
|                                   | Tüm satırlarda sabit değer taşıyan sütunları bul.    | `df[col].nunique() == 1` kontrol eder.       |
