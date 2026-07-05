# 🫀 Heart Failure Prediction — Machine Learning

A clinical decision support tool that predicts heart disease risk using **Logistic Regression**. The model achieves **91.59% recall** (sensitivity) by using a custom **30% decision threshold** — prioritizing catching true positives over minimizing false alarms, which is the clinically safer approach for screening.

> **🧪 [Try the live app](https://thehadrix-heart-failure-prediction-ml.streamlit.app/)**

---

## 📊 Overview

- **📁 Dataset:** [Heart Failure Clinical Records](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) — 918 patients, 11 features (Kaggle)
- **⚙️ Pipeline:** EDA (Sweetviz) → preprocessing (imputation, one-hot encoding, scaling) → model training & evaluation → Streamlit deployment
- **🏆 Best Model:** Logistic Regression with a **0.3 threshold** (vs. default 0.5)
- **📈 Performance:** 87.50% accuracy, **91.59% recall**, 84.21% precision

## 🧪 Models Evaluated

| Model | Threshold | Accuracy | Recall |
|---|---|---|---|
| Logistic Regression | **0.3** ✅ | 87.50% | **91.59%** |
| Random Forest | 0.3 | 84.78% | 91.59% |
| Logistic Regression | 0.5 | 86.41% | 85.05% |
| Random Forest | 0.5 | 87.50% | 88.79% |
| XGBoost | 0.5 | 82.61% | 82.24% |

## 🧬 Features Used

`Age` · `Sex` · `ChestPainType` · `RestingBP` · `Cholesterol` · `FastingBS` · `RestingECG` · `MaxHR` · `ExerciseAngina` · `Oldpeak` · `ST_Slope`

## 🚀 Usage

```bash
pip install -r requirements.txt
streamlit run GUI.py
```

## 📁 Project Structure

```
├── GUI.py                 # Streamlit web app
├── Predictor.ipynb        # Training & evaluation notebook
├── heart.csv              # Dataset
├── medical_lr_model.pkl   # Trained model
├── medical_scaler.pkl     # Fitted scaler
├── requirements.txt       # Dependencies
└── README.md
```

## 📄 License

MIT
