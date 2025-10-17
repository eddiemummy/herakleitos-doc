# 🧭 AutoEDA-LangModel Yönlendirme Kılavuzu

Bu kılavuz, tüm analitik aracılarının (agent'ların), hedeflerinin, anahtar kelimelerinin ve örnek sorgularının (İngilizce ve Türkçe) tam dokümantasyonunu sunar.

---

## 📈 ZAMAN SERİSİ ANALİZİ (ts)

**Hedef:** Zamansal örüntüleri tahmin etmek veya ayrıştırmak.
**Yöntemler:** Prophet ve Granger nedensellik testleri.

| Metrik Sayısı | Prophet | Granger | Açıklama |
| :---: | :---: | :---: | :--- |
| 0 | ✅ | ❌ | Metrik belirtilmemiş $\to$ tüm metrikleri tahmin et (Yalnızca Prophet). |
| 1 | ✅ | ✅ | Tek metrik $\to$ hem Prophet tahmini hem de Granger testi. |
| 2 | ❌ | ✅ | İki metrik $\to$ yalnızca Granger nedensellik testi. |
| 3+ | ❌ | ❌ | Çoklu metrik sorguları desteklenmez (uyarı). |

**Örnekler**

| İngilizce | Türkçe |
| :--- | :--- |
| Forecast overall campaign performance for the next 30 days. | Genel kampanya performansının 30 günlük tahminini yap. |
| Does Spend Granger-cause CTR? | Spend metriği CTR’ı Granger anlamında etkiliyor mu? |

---

## 🎯 DOWECON ARACISI (DoWhy / EconML)

**Hedef:** Statik veya panel veri nedensel etki analizi (tedavi $\to$ sonuç).
**Gereklilikler:** `campaign_name`, `ad_name`
**Anahtar Kelimeler:** “causal effect”, “treatment”, “outcome”, “nedensel analiz ile”

**Beklenen JSON Çıktısı**
\`\`\`json
{
  "treatment": "spend",
  "outcome": "ctr",
  "confounders": ["cpc", "impressions"],
  "campaign_name": "...",
  "ad_name": "...",
  "treatment_time": null
}
\`\`\`

---

## 📊 CAUSALPY ARACISI (Bayesian / Synthetic Control)

**Hedef:** Zaman serisi tabanlı nedensel etki veya yükseliş (uplift) analizi.
**Gereklilikler:** `outcome`, `predictors`, `campaign_name`, `ad_name`, `treatment_time`
**Anahtar Kelimeler:** “causal impact”, “Bayesian”, “synthetic control”, “at time”

**Beklenen JSON Çıktısı**
\`\`\`json
{
  "outcome": "ctr",
  "predictors": ["spend", "impressions"],
  "campaign_name": "...",
  "ad_name": "...",
  "treatment_time": 20
}
\`\`\`

---

## 🧪 A/B TEST ARAÇ KİTİ (ab)

**Hedef:** Kampanyaların, reklam setlerinin veya yaratıcıların deneysel karşılaştırması ve sıralanması.

| Yöntem | Anahtar Kelimeler | Örnek Anahtar Kelimeler |
| :--- | :--- | :--- |
| `basic_ab_test` | Sıralama/Seçim | **top, best, rank, en iyi** |
| `adaptive_comparison_test` | İki Varlık Karşılaştırması | **compare, versus, vs, ile karşılaştır** |
| `advanced_ab_test` | Etki Analizi | **effect, impact, affect, etkiler mi** |

---

## 🛒 İLİŞKİLENDİRME KURALLARI (assoc)

**Hedef:** Sıkça görülen öğe birlikteliklerini keşfetmek.
**Parametreler:** `support`, `confidence`, `lift`, `max_len`
**Algoritmalar:** Apriori, FP-Growth, Eclat

**Örnekler (Türkçe)**
* Kampanya düzeyinde 0.05 destek, 0.6 güven eşiğiyle kuralları çıkar.
* Her reklam seti için (groupwise) FP-Growth kurallarını çıkar.

---

## 🔍 SHAP AÇIKLANABİLİRLİĞİ (shap)

**Hedef:** Model tahminlerini açıklamak ve özellik önemini belirlemek.
**Mekanizma:** Hedef metriği (CTR, CPC, vb.) otomatik bul $\to$ SHAP değerleri üret.

**Örnekler (Türkçe)**
* CTR değerini etkileyen değişkenleri analiz et.
* Revenue metriği için SHAP analizi yap.

---

## 🧮 EDA — Keşifçi Veri Analizi

**Hedef:** İki sütun arasındaki istatistiksel ilişkileri keşfetmek.

| İlişki | Yöntem Örnekleri |
| :--- | :--- |
| Sayısal–Sayısal | Korelasyon (Pearson, Spearman) |
| Sayısal–Kategorik | ANOVA, Kruskal–Wallis |
| Kategorik–Kategorik | Chi-Kare, Cramér’s V |

**Örnekler (Türkçe)**
* CTR ve Spend arasında ilişki var mı?

---

## 📈 ANALİZ — Metrik İçgörüleri

**Hedef:** Bir veya iki metrik için temel özetler ve trendler sağlamak.

**Örnekler (Türkçe)**
* Ortalama CTR nedir?
* Son 30 gün için harcama trendini analiz et.

---

## 📉 PLOT — Görselleştirme Aracısı

**Hedef:** Değişkenler arasındaki ilişkileri gösteren grafikler oluşturmak.

| İlişki | Grafik Türü Örnekleri |
| :--- | :--- |
| Sayısal–Sayısal | Dağılım / Regresyon |
| Sayısal–Kategorik | Boxplot / Violin |
| Kategorik–Kategorik | Mozaik / Sayım |

**Örnekler (Türkçe)**
* CTR ve Spend grafiğini çiz.
* Revenue ile CPC arasındaki grafiği çiz.

---

## 🌍 GEOLIFT — Coğrafi Tabanlı Etki

**Hedef:** Bölgeler arası yükselişi veya performans etkisini ölçmek.

**Örnekler (Türkçe)**
* EMEA bölgesinde uplift ne kadar?
* GeoLift ile kampanya etkisini bölgelere göre analiz et.

---

## 📚 WIKI — Kavram Tanımı

**Hedef:** Veri kümesiyle ilgili olmayan genel kavramları tanımlamak/açıklamak.

**Örnekler (Türkçe)**
* CTR nedir?
* Bayes teoremini açıkla.

---

## ⚙️ YÖNLENDİRME KURALLARI ÖZETİ

| Koşul | Yönlendirilen Araç (Agent) |
| :--- | :--- |
| **“compare”, “top”, “best”, “rank”, “en iyi”** | **ab** |
| **“forecast”, “tahmin”, “öngörü”, “Granger”** | **ts** |
| **“causal effect”, “treatment/outcome”** | **dowecon** |
| **“causal impact”, “Bayesian”, “at time”** | **causalpy** |
| **“ortalaması”, “trend”, “değişimi”** | **analyze** |
| **“arasındaki ilişki”, “korelasyon”** (2 değişken) | **eda** |
| **“grafik”, “plot”, “visualize”** | **plot** |
| **“nedir”, “tanım”, “kimdir”** | **wiki** |

***
*👤 Author: AutoEDA-LangModel v3.2*
*🕒 Version: Routing Schema 2025.10*
