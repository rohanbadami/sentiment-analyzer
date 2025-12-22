# Integrated Financial Sentiment Analyzer & Dashboard

A professional-grade, full-stack financial analysis system that transforms raw news headlines into actionable trading signals. It leverages an ensemble of NLP models (FinBERT, VADER, Custom ML) and a robust "Gatekeeper" classifier to filter out market noise, ensuring you only see high-confidence opportunities.

![Status](https://img.shields.io/badge/Status-Active-success) ![Python](https://img.shields.io/badge/Python-3.12%2B-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B) ![LightGBM](https://img.shields.io/badge/AI-LightGBM-orange) ![License](https://img.shields.io/badge/License-MIT-green)

## 🚀 Project Overview

Most sentiment analyzers fail because they treat every news article as important. This project is different. It implements a **"Signal vs. Noise" philosophy**:

1.  **Ingests** thousands of headlines from the web (24/7).
2.  **Scores** them using financial-specific LLMs (FinBERT).
3.  **Validates** them against historical price action (Technical Analysis).
4.  **Filters** them using a trained "Gatekeeper" AI that rejects 90% of fluff.
5.  **Visualizes** the top 10% of "High Confidence" signals on a real-time dashboard.

### Key Features
* **Ensemble NLP Engine:** Combines **FinBERT** (Financial LLM), **VADER**, and a **Custom Dictionary** for nuanced sentiment scoring.
* **The "Gatekeeper" Model:** A LightGBM classifier trained on historical data to predict if news will *actually* move price. It achieves **68% Precision** on high-confidence signals.
* **Robust Data Handling:** Features a custom **"Weekend Patch"** to handle Sunday news by intelligently mapping it to Friday's close, preventing data gaps.
* **Technical Integration:** Automatically calculates RSI (14), MACD, Bollinger Bands, and Volatility (VIX) to give context to every headline.
* **Live Dashboard:** A Streamlit interface with "Watchlist," "Technical Charts," and "AI Insights" tabs.

---

## 🔄 System Architecture & Workflow

The pipeline is modular. Each phase builds upon the last, storing all data in a central MySQL database.
mermaid graph TD A[Scraper (Phase 1)] -->|Raw Headlines| B(MySQL Database) B --> C[Sentiment Engine (Phase 2)] C -->|Scores| B D[Price Fetcher (Phase 3)] -->|OHLCV + Technicals| B B --> E[AI Gatekeeper (Phase 4)] E -->|Confidence Scores| B B --> F[Streamlit Dashboard (Phase 5)]
---

## 📂 Detailed Components

### 1. Data Ingestion (Phase 1)

* **Script:** `phase1_headline_scraper.py`
* **Function:** Turbo-charged scraper using `concurrent.futures` to monitor hundreds of tickers simultaneously.
* **Features:** Rate-limit handling, user-agent rotation, and duplicate detection.

### 2. Sentiment Engine (Phase 2)

* **Script:** `phase2_sentiment_analysis.py`
* **Logic:** Runs text through the `IntegratedProcessor`.
* **FinBERT:** Detects subtle financial tone.
* **VADER:** Catches general positive/negative hype.
* **Custom ML:** Learns from your specific dataset over time.


* **Output:** A `sentiment_combined` score (-1 to 1) stored in the DB.

### 3. Market Context & "The Weekend Patch" (Phase 3)

* **Script:** `phase3_price_integration.py`
* **Function:** Fetches OHLCV data via `yfinance`.
* **Crucial Feature:** **The Weekend Patch**. Standard datasets fail on Sundays. This script "patches" weekend news to the last known market close (Friday) so the AI can still grade Sunday news without crashing.
* **Indicators:** RSI, MACD, Bollinger Band Width, SMA50.

### 4. The AI Core (Phase 4)

This is the brain of the operation. It consists of three parts:

* **A. The Gatekeeper (Classifier):** `phase4_classifier_mysql.py`
* Trains a LightGBM model to predict "Significant Moves" (e.g., >1.5% change).
* **Safety Feature:** Explicitly removes "Cheat Features" (like VIX, Day of Week) to prevent overfitting. The AI is forced to read the news, not the calendar.


* **B. The Price Predictor (Regressor):** `phase4_regressor_mysql.py`
* Estimates the *magnitude* of the move (e.g., "This looks like a +3% jump").


* **C. The Live Grader (Inference):** `phase4_backfill_predictions.py`
* **This is the worker.** It takes the trained model and grades all incoming news in real-time. It assigns the `ml_confidence` score (0.00 to 1.00).



### 5. The Dashboard (Phase 5)

* **Script:** `phase5_dashboard.py`
* **Tech:** Streamlit + Plotly.
* **Tabs:**
* 🔮 **Watchlist:** Top picks for the next market open.
* 📈 **Technicals:** Interactive charts with Bollinger Bands & News markers.
* 📝 **Data Explorer:** Raw view of the database.



---

## 🛠️ Installation & Setup

### Prerequisites

* Python 3.12+
* MySQL Server (Local or Remote)
* *(Optional)* NVIDIA GPU for faster FinBERT processing.

### 1. Clone & Install
bash git clone [https://github.com/yourusername/financial-sentiment-ai.git](https://github.com/yourusername/financial-sentiment-ai.git) cd financial-sentiment-ai pip install -r requirements.txt
### 2. Database Config

Create a `.env` file or update `db_mysql.py` with your credentials:
python DB_CONFIG = { 'host': 'localhost', 'user': 'root', 'password': 'your_password', 'database': 'sentiment_db' }
---

## 🚦 Usage Guide (The Pipeline)

Run these commands in order to build your database and train the brain.

### Step 1: Build the Dataset
bash # 1. Scrape News python phase1_headline_scraper.py # 2. Score Sentiment python phase2_sentiment_analysis.py # 3. Fetch Prices & Patch Weekends python phase3_price_integration.py
### Step 2: Train the AI
bash # Train the Gatekeeper (Signal vs Noise) python phase4_classifier_mysql.py eod
*Look for "Precision" in the logs. >60% is good.*

### Step 3: Run Live Inference
bash # Grade the latest news using the trained model python phase4_backfill_predictions.py
*Run this continuously or via cron job to keep dashboard fresh.*

### Step 4: Launch Dashboard
bash streamlit run phase5_dashboard.py
---

## 📊 Interpreting the Dashboard

The dashboard uses a binary **"Signal vs. Noise"** logic to keep decisions simple.

* **Confidence Score (`ml_confidence`):**
* 🟢 **Green (Signal):** **High Confidence (> 0.60).** The AI has identified a pattern that historically leads to price movement. **Actionable.**
* ⚪ **Grey (Noise):** **Low Confidence (< 0.60).** The AI believes this news is fluff, clickbait, or irrelevant to price action. **Ignore.**


* **"The Sunday Effect":**
* If viewing on a weekend, you may see multiple articles with identical confidence scores. This is normal behavior for the "Weekend Patch" logic as market technicals are frozen until Monday open.



---

## 🤝 Contributing

Contributions are welcome! Please submit a Pull Request or open an Issue for discussion.

## 📄 License

MIT License. See `LICENSE` for details.

---

## 📧 Contact

For questions or feedback, please open an issue on GitHub.
