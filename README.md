Here is the updated, comprehensive README. It uses your template but expands it to showcase every file you uploaded, documents the new "Weekend Patch" and "Anti-Overfitting" logic, and includes the critical Inference (Backfill) step.

You can copy this directly into your README.md.

Markdown

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
* **The "Gatekeeper" AI:** A LightGBM classifier trained on historical data to predict if news will *actually* move price. It achieves **68% Precision** on high-confidence signals.
* **Robust Data Handling:** Features a custom **"Weekend Patch"** to handle Sunday news by intelligently mapping it to Friday's close, preventing data gaps during inference.
* **Anti-Overfitting Logic:** The AI is explicitly trained *without* "Cheat Features" (like VIX, Day of Week) to force it to analyze the actual news content rather than memorizing calendar patterns.
* **Interactive Dashboard:** A Streamlit interface to explore top picks, visualize sentiment trends vs. Bollinger Bands, and audit AI confidence.

---

## 🔄 System Architecture & Workflow

The pipeline is modular. Each phase builds upon the last, storing all data in a central MySQL database.

1. The Core Pipeline ("Phases")
phase1_headline_scraper.py

Role: The Data Ingestor.

Tech: Uses concurrent.futures and requests with rate-limit handling.

Function: Monitors hundreds of tickers, scrapes headlines, and handles duplicates.

phase2_sentiment_analysis.py

Role: The NLP Engine.

Tech: multiprocessing, transformers, nltk.

Function: Processes raw text through the IntegratedProcessor to generate a sentiment_combined score (-1 to 1).

phase3_price_integration.py

Role: Context Provider & Weekend Patcher.

Tech: yfinance, pandas_ta.

Function: Fetches OHLCV data and calculates Technical Indicators (RSI, MACD, Bollinger Bands). Crucial: Implements the "Weekend Patch" to map Saturday/Sunday news to the last known market close.

phase4_classifier_mysql.py

Role: The "Gatekeeper" Trainer.

Tech: LightGBM, Optuna (Hyperparameter tuning).

Function: Trains the binary classifier to distinguish "Signal" (Price Move > 1.5%) from "Noise". Explicitly excludes macro features to prevent overfitting.

phase4_backfill_predictions.py

Role: The Live Inference Worker.

Tech: joblib, sklearn.

Function: Loads the trained Gatekeeper model and grades new incoming news in real-time, assigning the ml_confidence score.

phase4_regressor_mysql.py

Role: The Price Predictor.

Function: (Optional) Trains a regression model to estimate the magnitude of a potential move.

phase5_dashboard.py

Role: The User Interface.

Tech: Streamlit, Plotly.

Function: Displays the "Watchlist" (Green Signals), Technical Charts, and Data Explorer.

2. The NLP Engine Room
integrated_processor.py

Role: The Brain.

Function: Orchestrates the weighted voting system between FinBERT, VADER, and the Custom Dictionary.

sentiment_scorer.py

Role: Model Wrapper.

Function: specialized wrapper for the Hugging Face ProsusAI/finbert model and VADER lexicon.

3. Utilities & Data Management
unified_price_scripts.py

Role: Robust Price Fetcher.

Function: Handles caching and API rate limits for yfinance to ensure reliable data retrieval.

db_mysql.py: Central database handler (Schema definition & connection logic).

🛠️ Installation & Setup
Prerequisites
Python 3.12+

MySQL Server (local or remote)

CUDA-enabled GPU (Recommended for FinBERT, but works on CPU)

1. Clone the Repository
Bash

git clone [https://github.com/kvesu/Integrated-Sentiment-Analyzer-With-Dashboard.git](https://github.com/kvesu/Integrated-Sentiment-Analyzer-With-Dashboard.git)
cd Integrated-Sentiment-Analyzer-With-Dashboard
2. Install Dependencies
Bash

pip install -r requirements.txt
Note: If you have an NVIDIA GPU, install PyTorch with CUDA support manually to speed up processing.

3. Configure Database
Create a .env file in the root directory or update db_mysql.py:

Ini, TOML

MYSQL_HOST=localhost
MYSQL_USER=root
MYSQL_PASSWORD=your_password
MYSQL_DATABASE=sentiment_db
MYSQL_PORT=3306
🚦 Usage Guide
The pipeline must be run in this specific order to build the dataset and train the brain.

Step 1: Data Gathering
Scrape the latest headlines.

Bash

python phase1_headline_scraper.py
Step 2: Sentiment Analysis
Score the headlines using FinBERT/VADER.

Bash

python phase2_sentiment_analysis.py
Step 3: Market Data Sync
Fetch prices and apply the Weekend Patch.

Bash

python phase3_price_integration.py
Step 4: Train the AI (Gatekeeper)
Train the model to learn what constitutes a "Signal".

Bash

python phase4_classifier_mysql.py eod
Look for >60% Precision in the logs.

Step 5: Live Inference (Critical)
Grade the latest news using the trained model. Run this continuously.

Bash

python phase4_backfill_predictions.py
Step 6: Launch Dashboard
View the results.

Bash

streamlit run phase5_dashboard.py
📊 Interpreting the Dashboard
The dashboard uses a binary "Signal vs. Noise" logic to keep decisions simple.

Confidence Score (ml_confidence):

🟢 Green (Signal): High Confidence (> 0.60). The AI has identified a pattern that historically leads to price movement. Actionable.

⚪ Grey (Noise): Low Confidence (< 0.60). The AI believes this news is fluff, clickbait, or irrelevant to price action. Ignore.

"The Sunday Effect":

If viewing on a weekend, you may see multiple articles with similar confidence scores. This is normal behavior for the "Weekend Patch" logic as market technicals are frozen until Monday open.

🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

📄 License
This project is licensed under the MIT License.

📧 Contact
For questions or feedback, please open an issue on GitHub.
