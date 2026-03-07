# 🫀 CardioAI — Heart Disease Risk Predictor

A full-stack machine learning web application that predicts cardiovascular disease risk using a **K-Nearest Neighbours (KNN)** classifier trained on 918 patient records.

---

## 🚀 Live Demo

> **[https://cardioai-o6ug.onrender.com](https://cardioai-o6ug.onrender.com)**

---

## 🧠 How It Works

```
User fills form  →  JavaScript sends JSON  →  FastAPI receives request
                                                      ↓
                                            Pydantic validates input
                                                      ↓
                                            KNN model predicts risk
                                                      ↓
User sees result  ←  JavaScript updates UI  ←  JSON response returned
```

---

## 🗂️ Project Structure

```
CardioAI/
├── main.py                 ← FastAPI backend (API + serves frontend)
├── index.html              ← Frontend UI (served by FastAPI)
├── heart.csv               ← Training dataset (918 records)
│
├── knn_model.pkl           ← Trained KNN model (k=9)
├── scaler.pkl              ← Fitted StandardScaler
├── le_sex.pkl              ← LabelEncoder for Sex
├── le_angina.pkl           ← LabelEncoder for ExerciseAngina
├── feature_columns.pkl     ← Training feature column names
├── categorical_cols.pkl    ← Categorical column list
│
├── requirements.txt        ← Python dependencies
├── Procfile                ← Render deployment config
└── README.md               ← You are here
```

---

## ⚙️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | HTML · CSS · Vanilla JS |
| Backend | FastAPI + Uvicorn |
| ML Model | scikit-learn KNN (k=9) |
| Validation | Pydantic v2 |
| Hosting | Render |
| Dataset | [Heart Failure Prediction Dataset — Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) |

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Algorithm | K-Nearest Neighbours |
| K value | 9 |
| Train / Test split | 80% / 20% |
| Test Accuracy | **86.96%** |
| Dataset size | 918 patient records |
| Features | 11 clinical parameters |

---

## 🔬 Input Features

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| Age | int | 1–120 | Age in years |
| Sex | str | M / F | Biological sex |
| ChestPainType | str | ATA / NAP / ASY / TA | Type of chest pain |
| RestingBP | float | 80–220 mm Hg | Resting blood pressure |
| Cholesterol | float | 0–600 mg/dl | Serum cholesterol |
| FastingBS | int | 0 or 1 | Fasting blood sugar > 120 mg/dl |
| RestingECG | str | Normal / ST / LVH | Resting ECG result |
| MaxHR | int | 60–220 BPM | Maximum heart rate |
| ExerciseAngina | str | Y / N | Exercise-induced angina |
| Oldpeak | float | 0.0–7.0 | ST depression (Oldpeak) |
| ST_Slope | str | Up / Flat / Down | Slope of ST segment |

---

## 🛠️ Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/vansh7266/CardioAI.git
cd CardioAI
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Start the server
```bash
uvicorn main:app --reload --port 8000
```

### 4. Open in browser
```
http://localhost:8000        ← App
http://localhost:8000/docs   ← Swagger API docs
http://localhost:8000/health ← Health check
```

---

## 🧪 API Test Cases

**POST** `/predict_risk` — try it live at [/docs](https://cardioai-o6ug.onrender.com/docs)

**High Risk test:**
```json
{
  "Age": 55,
  "Sex": "M",
  "ChestPainType": "ASY",
  "RestingBP": 140,
  "Cholesterol": 289,
  "FastingBS": 0,
  "RestingECG": "Normal",
  "MaxHR": 132,
  "ExerciseAngina": "N",
  "Oldpeak": 1.5,
  "ST_Slope": "Flat"
}
```
Expected → `{ "prediction": 1, "probability": "77.8", "label": "High Risk" }`

**Low Risk test:**
```json
{
  "Age": 40,
  "Sex": "F",
  "ChestPainType": "ATA",
  "RestingBP": 110,
  "Cholesterol": 210,
  "FastingBS": 0,
  "RestingECG": "Normal",
  "MaxHR": 172,
  "ExerciseAngina": "N",
  "Oldpeak": 0.0,
  "ST_Slope": "Up"
}
```
Expected → `{ "prediction": 0, "probability": "88.9", "label": "Low Risk" }`

---

## ☁️ Deployed on Render

| Setting | Value |
|---|---|
| Platform | Render (Free tier) |
| Region | Singapore |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `uvicorn main:app --host 0.0.0.0 --port $PORT` |

> ⚠️ Free tier sleeps after 15 min of inactivity. First request after sleep takes ~30 seconds to wake up.

---

## ⚠️ Disclaimer

This tool is for **educational purposes only**. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider.

---

## 👨‍💻 Author

**Vansh Gupta** — [@vansh7266](https://github.com/vansh7266)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
