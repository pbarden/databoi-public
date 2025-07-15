# 📊 databoi

**`databoi`** is a modular, Python-based analytics library purpose-built for experimentation and customer behavior modeling. It supports robust, scalable workflows for A/B/n testing, multivariate experiments, Bayesian analysis, regression, clustering, retention modeling, and segmentation.

`databoi` is designed to unify pre-test planning, in-test monitoring, and post-test evaluation using consistent inputs and composable statistical tools. It emphasizes operational analytics and is optimized for conversion-centric use cases.

---

## 🔧 What It Does

### Experiment Design & Power Planning
- Calculate minimum detectable effects (MDEs)
- Estimate duration needed for statistical confidence
- Model power curves and thresholds

### Statistical Testing
- Frequentist Z-tests & Chi-squared tests
- Bayesian posterior sampling & credible intervals
- Sample ratio mismatch detection

### Segment & Time-Based Modeling
- Weekly summaries and time-to-significance forecasting
- Segment-aware significance testing
- Multi-week retention and conversion tracking

### Behavioral & Customer Modeling
- Linear regression on engagement and revenue drivers
- K-Means clustering and customer segmentation
- Cohort-based retention analysis

### Attribution & Journey Analysis
- Markov chain attribution estimation
- Time-to-conversion distribution tracking
- Channel path impact modeling

### Predictive Targeting
- Optimal channel prediction via Random Forests
- High-value customer detection & similarity mapping

---

## 📁 Expected Input Format

All core modules expect a **normalized CSV** where each row represents a unique user per time period. The input data should match the following structure:

### Required Columns
- `user_id` – Unique ID per user  
- `converted` – Binary conversion flag (`1` or `0`)  
- `week` – Period identifier (e.g., `2024-W01`, `1`, `2`)

### Optional Columns
- `group` – A/B/n test label (`control`, `variant_a`, etc.)  
- `spend`, `visits`, `sessions`, `age`, `past_purchases`  
- `channel`, `device`, `region` — used for filtering or grouping

> Additional fields are supported and ignored unless specified in analysis calls.

---

## ⚙️ Usage Example

```python
import pandas as pd
import databoi

# Load your experiment data
df = pd.read_csv("experiment_data.csv")

# Generate weekly MDE timeline
summary = df.groupby("week").agg(
    users=('user_id', 'count'),
    conversions=('converted', 'sum')
).reset_index()

results = databoi.mde_time_table(summary, total_weeks=6)

print("Week-by-week MDE values:")
print(results["weekly_mdes"])

(C) Copyright 2025 Paul Barden