Perfect — I verified the repo you linked: **[https://github.com/kvesu/Integrated-Sentiment-Analyzer-With-Dashboard](https://github.com/kvesu/Integrated-Sentiment-Analyzer-With-Dashboard)**. This is the exact project we just formatted a README for. ([GitHub][1])

Here is the **full Markdown README you can paste directly into your repo’s `README.md`**, now updated with your GitHub link included and polished:

---

````markdown
# Integrated Financial Sentiment Analyzer & Dashboard

A professional-grade, full-stack financial analysis system that transforms raw news headlines into actionable trading signals. It leverages an ensemble of NLP models (FinBERT, VADER, Custom ML) and a robust **Gatekeeper (Binary Classifier)** to filter out market noise—ensuring you only see high-confidence opportunities.

[**View on GitHub**](https://github.com/kvesu/Integrated-Sentiment-Analyzer-With-Dashboard)  

![Status](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Python-3.12%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B)
![LightGBM](https://img.shields.io/badge/AI-LightGBM-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🚀 Project Overview

Most sentiment analyzers fail because they treat every news article as important. This system is different. It implements a strict **Signal vs. Noise** philosophy:

1. **Ingests** thousands of financial news headlines continuously  
2. **Scores** them using financial-specific NLP models (FinBERT, VADER, Custom ML)  
3. **Validates** sentiment against historical market behavior  
4. **Filters** headlines using a trained **Gatekeeper AI** that rejects ~90% of noise  
5. **Visualizes** only **high-confidence signals** in a real-time dashboard  

---

## ⭐ Key Features

- **Ensemble NLP Engine**  
  Combines **FinBERT** (financial LLM), **VADER**, and a **custom sentiment model** for nuanced scoring.

- **Gatekeeper Binary Classifier**  
  A LightGBM model trained to predict whether news will *actually* move price, achieving **~68% precision** on actionable signals.

- **Robust Data Engineering**  
  Includes a custom **Weekend Patch** that maps weekend news to Friday’s close to prevent missing market context.

- **Technical Market Context**  
  Automatically computes RSI (14), MACD, Bollinger Bands, SMA-50, and volatility context.

- **Live Analytics Dashboard**  
  Built with **Streamlit + Plotly**, offering Watchlists, Technical Charts, and AI-driven insights.

---

## 🔄 System Architecture & Workflow

Each pipeline stage builds on the last, with all data persisted in a central MySQL database.

```mermaid
graph TD
    A[Headline Scraper] -->|Raw News| B(MySQL)
    B --> C[Sentiment Engine]
    C -->|Sentiment Scores| B
    D[Market Data & Technicals] -->|OHLCV| B
    B --> E[AI Gatekeeper]
    E -->|Confidence Scores| B
    B --> F[Streamlit Dashboard]
````

---

## 📂 Pipeline Components

### Phase 1 — Data Ingestion

* **Script:** `phase1_headline_scraper.py`
* High-throughput concurrent scraper
* Handles rate limits, user-agent rotation, and duplicate detection

---

### Phase 2 — Sentiment Analysis

* **Script:** `phase2_sentiment_analysis.py`
* Uses an `IntegratedProcessor` pipeline

  * **FinBERT:** Financial nuance
  * **VADER:** General market tone
  * **Custom ML:** Dataset-adaptive learning

**Output:**

* `sentiment_combined` ∈ **[-1, 1]**, stored in MySQL

---

### Phase 3 — Market Context & Weekend Patch

* **Script:** `phase3_price_integration.py`
* Fetches OHLCV data using `yfinance`
* **Weekend Patch:** Assigns weekend news to last market close (Friday)
* **Computed Indicators:**

  * RSI (14)
  * MACD
  * Bollinger Band Width
  * SMA-50

---

### Phase 4 — AI Core

#### A. Gatekeeper (Binary Classifier)

* **Script:** `phase4_classifier_mysql.py`
* Predicts whether news causes a **significant price move** (e.g., >1.5%)
* Removes “cheat features” (VIX, weekday, etc.) to prevent leakage

#### B. Price Magnitude Regressor

* **Script:** `phase4_regressor_mysql.py`
* Estimates expected move size (e.g., +2.8%)

#### C. Live Inference Worker

* **Script:** `phase4_backfill_predictions.py`
* Applies trained models to incoming news
* Outputs `ml_confidence` ∈ **[0.00, 1.00]**

---

### Phase 5 — Dashboard

* **Script:** `phase5_dashboard.py`
* **Stack:** Streamlit + Plotly

**Dashboard Tabs:**

* 🔮 **Watchlist** — High-confidence upcoming signals
* 📈 **Technicals** — Interactive charts with indicators & news markers
* 🗂 **Data Explorer** — Raw database inspection

---

## 🛠️ Installation & Setup

### Prerequisites

* Python **3.12+**
* MySQL Server
* *(Optional)* NVIDIA GPU for accelerated FinBERT inference

---

### Clone & Install

```bash
git clone https://github.com/kvesu/Integrated-Sentiment-Analyzer-With-Dashboard.git
cd Integrated-Sentiment-Analyzer-With-Dashboard
pip install -r requirements.txt
```

---

### Database Configuration

Update credentials in `db_mysql.py` or via `.env`:

```python
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "your_password",
    "database": "sentiment_db"
}
```

---

## 🚦 Running the Pipeline

### Step 1 — Build the Dataset

```bash
python phase1_headline_scraper.py
python phase2_sentiment_analysis.py
python phase3_price_integration.py
```

---

### Step 2 — Train Gatekeeper

```bash
python phase4_classifier_mysql.py eod
```

> Precision >60% is strong for financial signal classification.

---

### Step 3 — Live Inference

```bash
python phase4_backfill_predictions.py
```

Run continuously or via cron.

---

### Step 4 — Launch Dashboard

```bash
streamlit run phase5_dashboard.py
```

---

## 📊 Interpreting the Dashboard

This system uses a **binary Signal vs. Noise decision framework**.

### Confidence Score (`ml_confidence`)

* 🟢 **Green — Signal (> 0.60)**
  High likelihood of price impact. **Actionable.**

* ⚪ **Grey — Noise (< 0.60)**
  Likely irrelevant or clickbait. **Ignore.**

### Weekend Behavior

On weekends, identical confidence scores may appear due to frozen technical indicators. This is expected behavior from the **Weekend Patch** logic.

---

## 🤝 Contributing

Pull Requests and Issues are welcome!

---

## 📄 License

MIT License. See `LICENSE` for details.

---

