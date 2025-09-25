# Churn Prediction — Spotify

---

## Executive Summary

โปรเจกต์นี้วิเคราะห์ข้อมูลผู้ใช้งาน Spotify เพื่อพยากรณ์ **churn (การเลิกใช้งาน)** โดยใช้ Machine Learning Pipeline ที่ประกอบด้วยการทำ **EDA, Feature Engineering, SMOTE balancing, Model Training, และ Hyperparameter Tuning**

* **Dataset:** ข้อมูลพฤติกรรมการฟังเพลง เช่น `listening_time`, `songs_played_per_day`, `skip_rate`, `ads_listened_per_week` และ demographic features (`age`, `gender`, `region`)
* **Feature Engineering:** สร้างตัวแปรใหม่ เช่น `song_skipped`, `high_skip`, `low_listening`, `ads_burden`, `engagement_score`
* **Models Tested:** Logistic Regression, Random Forest, XGBoost
* **Best Model:** Tuned XGBoost (`RandomizedSearchCV`) ให้ผลดีที่สุด (Accuracy ≈ 0.79, F1 ≈ 0.76, ROC-AUC ≈ 0.79)
* **Key Drivers of Churn:** engagement level, listening time, songs per day, skip rate, ads burden

✅ **ผลลัพธ์สรุป:** โมเดลสามารถทำนาย churn ได้อย่างแม่นยำพอสมควร โดยมี XGBoost เป็นตัวเลือกที่ดีที่สุดในงานนี้

---

## Appendix (Technical Details)

### Dataset & Preprocessing

* โหลดข้อมูล: `Spodify dataset 2025.csv`
* Data cleaning: drop missing, แปลง country → region, แบ่งช่วงอายุ, one-hot encoding
* Scaling: `MinMaxScaler` กับ features เชิงตัวเลข
* Handling imbalance: `SMOTE` oversampling

### Feature Engineering (ตัวอย่างโค้ด)

```python
Spodify_dataset['song_skipped'] = Spodify_dataset['songs_played_per_day'] * Spodify_dataset['skip_rate']
Spodify_dataset['high_skip'] = (Spodify_dataset['skip_rate'] > 0.5).astype(int)
Spodify_dataset['low_listening'] = (Spodify_dataset['listening_time'] < 30).astype(int)
Spodify_dataset['ads_burden'] = Spodify_dataset['ads_listened_per_week'] / (Spodify_dataset['listening_time'] + 1)
```

### Models & Evaluation

* Logistic Regression (`liblinear`, max_iter=1000)
* Random Forest (100 trees)
* XGBoost (default & tuned)
* Tuning: `RandomizedSearchCV` (cv=5, scoring=f1, n_iter=50)

**Performance Table:**

| Model              | Accuracy | Precision | Recall | F1   | ROC-AUC |
| ------------------ | -------- | --------- | ------ | ---- | ------- |
| LogisticRegression | 0.76     | 0.82      | 0.67   | 0.74 | 0.76    |
| Random Forest      | 0.78     | 0.84      | 0.68   | 0.75 | 0.78    |
| XGBoost            | 0.79     | 0.85      | 0.69   | 0.76 | 0.79    |
| Best XGBoost       | 0.79     | 0.84      | 0.70   | 0.76 | 0.79    |

### Visualization Highlights

* **Correlation Heatmap** (ตัวแปรเชิงตัวเลข)
* **Churn Distribution** (countplot)
* **Confusion Matrices** (per model)
* **Feature Importance** (XGBoost: top drivers = engagement, listening_time, songs_per_day, ads_burden)

---

## How to Run

```bash
git clone https://github.com/nawinsupasorn-commits/Churn-Prediction.git
cd Churn-Prediction
python -m venv venv
source venv/bin/activate  # หรือ venv\Scripts\activate บน Windows
pip install -r requirements.txt
jupyter notebook Spodify_EDA_ML.ipynb
```

---

## Author

**Nawin Supasorn**

---


