# 📊 databoi

**`databoi`** is a modular Python analytics toolkit for conversion rate optimization, behavioral modeling, and experiment analysis. It supports robust workflows for A/B/n testing, Bayesian and frequentist inference, segmentation, regression, and time-based customer analysis — all built for Python scripts, Jupyter notebooks, and CLI use without requiring a Django backend.

`databoi` brings structure to operational analytics by unifying:
- Pre-test planning (e.g., MDE and power calculations)
- In-test monitoring (e.g., SRM checks, path analysis)
- Post-test evaluation (e.g., Bayesian lifts, decision trees, cohort-based metrics)

---

## 🔧 Capabilities

### 📐 Experiment Design
- Minimum detectable effect (MDE) estimation
- Duration and sample size planning
- Power curves over time

### 🧪 Statistical Testing
- Z-tests and Chi-squared tests
- Bayesian posterior inference
- Credible intervals and sampling
- Sample ratio mismatch (SRM) checks

### 🧭 Attribution & Path Modeling
- First-order Markov attribution modeling
- Conversion path analysis
- Time-to-conversion estimation

### 👥 Customer Segmentation
- K-Means clustering of user cohorts
- High-value customer detection
- Similarity mapping via nearest neighbors

### ⏱ Retention & Time Modeling
- Weekly time-to-significance projections
- Cohort-based retention curves
- Time-to-conversion distributions

### 📊 Predictive Modeling
- Linear and logistic regression
- Decision trees and boosted tree classifiers
- Feature importance for targeting

---

## 📁 Data Format

Most modules accept either:
- A single **pandas DataFrame**
- Or multiple DataFrames representing `persons`, `sessions`, and `events`

Your data should be in a **normalized format**, where each row represents a user, session, or event depending on the analysis context.

### Common Columns
| Column         | Description                                  |
|----------------|----------------------------------------------|
| `user_id`      | Unique user identifier                       |
| `converted`    | Binary flag for conversion (0 or 1)          |
| `week`         | Time period indicator (e.g. `'2024-W03'`)    |
| `group`        | A/B test group (e.g., `'control'`, `'A'`)    |
| `channel`      | Acquisition or attribution channel           |
| `value`        | Revenue or outcome value                     |
| `timestamp`    | Event/session timestamp                      |

> Custom fields are supported and passed through unchanged.

---

## ⚙️ Example Usage

```python
import pandas as pd
import databoi

# Load experiment data
df = pd.read_csv("experiment_data.csv")

# Example: calculate MDE over a 6-week test
summary = df.groupby("week").agg(
    users=('user_id', 'count'),
    conversions=('converted', 'sum')
).reset_index()

mde_results = databoi.mde_time_table(summary, total_weeks=6)
print("Week-by-week MDEs:")
print(mde_results["weekly_mdes"])

(C) Copyright 2025 Paul Barden