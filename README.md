# 🫀 CardioAI — Heart Disease Risk Prediction Platform

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-3.x-black?style=flat-square&logo=flask)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-orange?style=flat-square&logo=scikit-learn)
![SHAP](https://img.shields.io/badge/SHAP-Explainable_AI-purple?style=flat-square)
![Gemini](https://img.shields.io/badge/Gemini-AI_Chat-blue?style=flat-square&logo=google)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![CI](https://github.com/Astreonix/CardioAI/actions/workflows/ci.yml/badge.svg)
![Deploy](https://img.shields.io/badge/Deploy-Render-46E3B7?style=flat-square&logo=render)

An end-to-end AI-powered heart disease risk prediction platform built with Flask, scikit-learn, SHAP explainability, Gemini AI chat, SQLAlchemy, and ReportLab PDF generation.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎯 **ML Prediction** | Random Forest classifier — ROC-AUC 0.86 |
| 🧠 **SHAP Explainability** | Per-prediction feature impact chart |
| 💬 **Gemini AI Chat** | Context-aware cardiovascular health assistant |
| 📄 **PDF Reports** | Downloadable clinical report with SHAP chart |
| 🗄️ **Database** | SQLite via SQLAlchemy — predictions + chat history + reports |
| 🔒 **Rate Limiting** | Flask-Limiter protection on all POST routes |
| ✅ **Input Validation** | Server + client side validation on all 13 fields |
| 📊 **History Dashboard** | Filter, search, and review all past assessments |

---

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/mehrali-hub/CardioAI.git
cd CardioAI

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment variables
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# 4. Run
python app.py

# 5. Open browser
# http://localhost:5000
```

---

## 📁 Project Structure

```
cardioai_project/
├── app.py                          # Flask app — all routes + ML + PDF
├── requirements.txt
├── .env.example                    # Environment variable template
│
├── templates/
│   ├── index.html                  # Landing page
│   ├── predict.html                # Assessment form + SHAP results
│   ├── history.html                # Prediction history dashboard
│   ├── chat.html                   # Gemini AI chat assistant
│   └── 404.html                    # Error page
│
├── models/
│   ├── cardioai_model.pkl          # Trained Random Forest
│   ├── scaler.pkl                  # StandardScaler
│   ├── feature_columns.pkl         # Feature list
│   ├── model_metadata.pkl          # Metrics + params
│   └── shap_explainer.pkl          # TreeExplainer
│
├── data/
│   ├── CardioAI_Merged_Heart_Dataset.csv
│   └── processed/                  # Train/test splits
│
├── images/                         # EDA + model plots (9 files)
└── notebooks/
    ├── 01_EDA.ipynb
    ├── 02_Preprocessing.ipynb
    └── 03_Model_Training.ipynb
```

---

## 🤖 Model Performance

| Metric | Score |
|---|---|
| ROC-AUC | **0.86** |
| Accuracy | 76.7% |
| Precision | 76.7% |
| Recall | 76.7% |
| F1 Score | 76.7% |

Trained on the UCI Heart Disease dataset (303 patients, 13 features).

---

## 🔑 Environment Variables

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | Get free at [aistudio.google.com](https://aistudio.google.com) |
| `FLASK_SECRET_KEY` | Any random string |
| `FLASK_ENV` | `development` or `production` |

---

## 📍 Routes

| Route | Method | Description |
|---|---|---|
| `/` | GET | Landing page |
| `/predict` | GET/POST | Run ML prediction |
| `/history` | GET | View all assessments |
| `/history/data` | GET | JSON API for history |
| `/chat` | GET | AI chat assistant |
| `/chat/message` | POST | Send message to Gemini |
| `/report/<id>` | GET | Download PDF report |

---

## 🏗️ Milestones

- ✅ M1 — Documentation
- ✅ M2 — EDA & Visualization
- ✅ M3 — Model Training & Evaluation
- ✅ M4 — Flask Backend (all routes + DB)
- ✅ M5 — Full Frontend (4 pages)
- ✅ M6 — AI Integration, SHAP, PDF, Rate Limiting, Validation
- ✅ M7 — Docker + Render Deployment
- 🔲 M8 — Kaggle Notebook + Portfolio

---

## 🐳 Deployment (M7)

### Deploy to Render (Free)

1. Push to GitHub:
```bash
git init
git add .
git commit -m "🫀 CardioAI - Complete ML Platform"
git branch -M main
git remote add origin https://github.com/mehrali-hub/CardioAI.git
git push -u origin main
```

2. Go to **render.com** → New → Web Service → Connect your GitHub repo

3. Render auto-detects `render.yaml` and `Dockerfile`

4. Add your `GEMINI_API_KEY` in Render dashboard → Environment

5. Click **Deploy** → your app goes live at `https://cardioai.onrender.com`

### Run with Docker locally
```bash
# Build
docker build -t cardioai .

# Run
docker run -p 5000:5000 -e GEMINI_API_KEY=your_key cardioai

# Open http://localhost:5000
```


---

## 👨‍💻 Author

**Mehr Ali** — BS Computer Science, FAST NUCES  
[![GitHub](https://img.shields.io/badge/GitHub-Astreonix-black?style=flat-square&logo=github)](https://github.com/Astreonix)
[![Kaggle](https://img.shields.io/badge/Kaggle-mehralieng-blue?style=flat-square&logo=kaggle)](https://kaggle.com/mehralieng)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
