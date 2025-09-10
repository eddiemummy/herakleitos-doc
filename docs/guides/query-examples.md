# Query Examples by Type

Use these examples as templates for forming queries in **Herakleitos**.  
Replace placeholder column names (e.g., `clicks`, `revenue`, `region`) with actual dataset columns.

---

## `eda` — Exploratory Data Analysis
- “Explore the relationship between **clicks** and **revenue** over the last 30 days.”
- “Show distributions and correlation between **impressions** and **CTR**.”
- “Compare **cost** vs **conversions** with summary stats and tests.”
- “Run EDA on **sessions** and **avg_order_value**.”

---

## `analyze` — Agent, Trends & Behavior
- “How did **revenue** evolve over time? Any notable trends or spikes?”
- “Analyze performance changes for **CTR** last month.”
- “What explains the drop in **conversions** last week?”
- “Investigate anomalies in **cost_per_acquisition**.”

---

## `shap` — Model Interpretability (Target Detection Required)
- “Explain SHAP for **CTR** and identify the most important drivers.”
- “Compute SHAP values to interpret **conversions**.”
- “Which features most influence **revenue** according to SHAP?”
- “Provide feature importance using SHAP for **retention_rate**.”

---

## `geolift` — Geo/Regional Uplift
```
Using spend as metric and region as location and date as date.
```
```
Using revenue as metric and geo as location and ds as date; budget 20000 cpic 3.5 alpha 0.1 treatment start 2024-09-01 treatment duration 21.
```
```
Using conversions as metric and city as location and date as date.
```
```
Using sales as metric and market as location and dt as date; alpha 0.05.
```

---

## `dowecon` — DoWhy / EconML Causal Inference
```
treatment=ad_spend; outcome=conversions; confounders=[seasonality, device, channel]; instrument=budget_cap; treatment_time=date; target_unit=campaign_id
```
```
treatment=discount; outcome=revenue; confounders=[weekday, traffic]
```
```
treatment=exposure; outcome=signup; confounders=[age, city, device]; instrument=assignment; target_unit=user_id
```
```
treatment=promo; outcome=ctr; confounders=[geo, hour, platform]; treatment_time=ds
```

> **Tip:** Keep names exactly as your columns; the pipeline will fuzzy-match but valid columns are required.

---

## `causalpy` — Synthetic Control (Bayesian & Sklearn)
```
Using revenue as outcome, analyze the causal effect with predictors [impressions, clicks, cost] at time 2024-06-15
```
```
Using sales as outcome, analyze the causal effect with predictors [footfall, ad_spend] at time 10
```
```
Using conversions as outcome, analyze the causal effect with predictors [sessions, ctr, cpc] at time 2023-12-01
```
```
Using signup as outcome, analyze the causal effect with predictors [traffic, device_share] at time 25
```

---

## `ts` — Time Series & Forecasting
- “Forecast **revenue** for the next **45 days**.”
- “Run Granger causality between **impressions** and **clicks**.”
- “Forecast **conversions** for **Acme_Campaign** for the next **30 days**.”
- “Decompose seasonality for **sessions** and predict 60 days ahead.”

> The horizon is parsed from phrases like *“next 30 days”*;  
> including a campaign name helps select a company/series.

---

## `wiki` — Knowledge / Definitions
- “What is **CTR**?”
- “Explain **Prophet** in time-series forecasting.”
- “What does **Granger causality** mean?”
- “Define **SHAP values**.”

---

## `plot` — Visualizations (Requires ≥2 Columns)
- “Plot **date** vs **revenue**.”
- “Visualize **clicks** against **conversions**.”
- “Create charts for **cost**, and **impressions**.”

> The plotting pipeline auto-casts object columns to datetime if ≥70% parse;  
> still, include a datetime column when possible.

---

## `ab` — A/B Testing
- “A/B test **Variant A** vs **Variant B** on **conversion_rate**; is the lift significant?”
- “Compare **control** vs **treatment** for **CTR**; report p-values and effect size.”
- “Evaluate significance for **signup_rate** between **A** and **B**.”
- “Which variant wins on **revenue_per_user**?”

---

## `assoc` — Association Rules / Market Basket
- “Mine association rules for **product_basket** (top 50 rules).”
- “Show frequent itemsets and rules with high **lift** and **confidence**.”
- “Find associations among **category** purchases.”
- “Market basket analysis on **items**; summarize strongest patterns.”

---

# Quick Guidance
- **Name columns explicitly** in your query (e.g., `date`, `revenue`, `region`) to help `extract_columns_from_query` and validators.  
- For **GeoLift**, always use:  
  `Using <metric> as metric and <location> as location and <date> as date (+ optional budget, cpic, alpha, treatment start, treatment duration)`.  
- For **DoWhy/EconML**, use key-value pairs exactly as shown; lists use `[a, b, c]`.  
- For **CausalPy**, stick to:  
  `Using <outcome> as outcome, ... predictors [a, b] at time <t>` (date or index).  
- For **TS**, include the horizon (“next 30 days”) and, if applicable, a campaign identifier present in your data.  
- For **Plot**, include two valid columns.
