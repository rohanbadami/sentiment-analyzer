# Integrated Financial Sentiment Analyzer & Dashboard

A full-stack financial analysis pipeline that filters active tickers, scrapes news headlines, performs multi-model sentiment analysis (FinBERT, VADER, Custom ML), integrates market price data, and visualizes actionable "Signal vs. Noise" insights via an interactive dashboard.

![Dashboard Preview](https://img.shields.io/badge/Status-Active-success) ![Python](https://img.shields.io/badge/Python-3.12%2B-blue) ![License](https://img.shields.io/badge/License-MIT-green)

## 🚀 Project Overview

This system automates the process of finding correlations between financial news sentiment and stock price movements. It moves beyond simple "positive/negative" scoring by using a **Gatekeeper Model** to filter out irrelevant news ("Noise") and focus on high-confidence "Signals."

### Key Features

- **Smart Filtering:** Pre-scans the market to target only tickers with active news coverage
- **Advanced Sentiment Engine:** Combines **FinBERT** (Financial LLM), **VADER**, and a **Custom Dictionary/ML Learner** for robust scoring
- **"Gatekeeper" AI:** A dedicated classifier that predicts if a news item is likely to move the stock price (Signal) or not (Noise)
- **Price Integration:** Automatically fetches historical price data, calculates technical indicators (RSI, MACD, Bollinger Bands), and aligns them with news timestamps
- **Interactive Dashboard:** A Streamlit-based UI to explore top picks, visualize sentiment trends, and audit AI confidence

---

## 🔄 Workflow

The pipeline is designed to run in a specific order. **Ticker Filtering** must run first to generate the target list for Phase 1.

1. **Preparation:** `ticker_filter.py` (Generates target list)
2. **Phase 1:** `phase1_headline_scraper.py` (Fetches data)
3. **Phase 2:** `phase2_sentiment_analysis.py` (Analyzes sentiment)
4. **Phase 3:** `phase3_price_integration.py` (Syncs market data)
5. **Phase 4:** `phase4_classifier_mysql.py` (Trains AI models)
6. **Phase 5:** `phase5_dashboard.py` (Visualizes results)

---

## 📂 Repository Structure

### 1. Preparation (Pre-Phase)

- **`ticker_filter.py`**: The starting point. Scans the market to generate an optimized list of active tickers (e.g., `tickers_with_news.json`) so the scraper doesn't waste time on inactive stocks

### 2. The Core Pipeline ("Phases")

- **`phase1_headline_scraper.py`**: Scrapes latest headlines for the filtered list of tickers
- **`phase2_sentiment_analysis.py`**: Runs the text through the `IntegratedProcessor` to generate sentiment scores and "Signal/Noise" classification
- **`phase3_price_integration.py`**: Fetches OHLCV data, calculates technicals (RSI, MACD), and syncs them with news events
- **`phase4_regressor_mysql.py`**: Trains a regression model to predict the magnitude of price changes based on sentiment
- **`phase5_dashboard.py`**: Launches the web interface to view results

### 3. The Engine Room (Dependencies)

- **`db_mysql.py`**: Central database handler (Schema definition & connection logic)
- **`unified_price_scripts.py`**: Robust price fetcher with caching and rate-limit protection for the Yahoo Finance API
- **`integrated_processor.py`**: The "Brain" that combines Josh's Scorer (FinBERT) and Kris's Learner (Dictionary/ML) into a single score
- **`sentiment_scorer.py`**: Specific implementations for VADER and FinBERT models

### 4. Utilities

- **`check_db.py`**: Quick system status report (row counts, missing data checks)
- **`export_db_to_csv_enhanced.py`**: Exports your SQL database to a CSV dataset for external analysis
- **`reset_database.py`**: Wipes calculated data (sentiment/prices) while keeping scraped headlines safe

---

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.12+
- MySQL Server (local or remote)
- CUDA-enabled GPU (Recommended for FinBERT, but works on CPU)

### 1. Clone the Repository

```bash
git clone https://github.com/kvesu/Integrated-Sentiment-Analyzer-With-Dashboard.git
cd Integrated-Sentiment-Analyzer-With-Dashboard
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** If you have an NVIDIA GPU, install PyTorch with CUDA support manually to speed up processing.

### 3. Configure Database

Create a `.env` file in the root directory:

```ini
MYSQL_HOST=localhost
MYSQL_USER=root
MYSQL_PASSWORD=your_password
MYSQL_DATABASE=sentiment_db
MYSQL_PORT=3306
```

---

## 🚦 Usage Guide

### Step 1: Preparation (Ticker Filtering)

Run this first to determine which stocks to analyze.

```bash
python ticker_filter.py
```

### Step 2: Phase 1 (Data Gathering)

Scrape headlines for the tickers identified in Step 1.

```bash
python phase1_headline_scraper.py
```

### Step 3: Phase 2 (Sentiment Analysis)

Run the NLP engine to score headlines and classify them as "Signal" or "Noise."

```bash
python phase2_sentiment_analysis.py
```

### Step 4: Phase 3 (Market Data Sync)

Fetch price history and calculate technical indicators.

```bash
python phase3_price_integration.py
```

### Step 5: Phase 5 (Dashboard)

View the real-time results in your browser.

```bash
streamlit run phase5_dashboard.py
```

---

## 🧠 Model Architecture

The **Integrated Processor** (`integrated_processor.py`) uses a weighted ensemble approach:

- **FinBERT (ProsusAI)**: Specialized for financial language
- **Custom Dictionary**: Weighted keyword scoring with strength multipliers
- **VADER**: Rule-based baseline for general sentiment
- **Question Detector**: Specialized logic for handling clickbait/question headlines
- **Confidence Score**: The system calculates an "ML Confidence" score based on model agreement. High agreement = High Confidence ("Signal")

---

## 📊 Data Dictionary (Main Columns)

| Column | Description |
|--------|-------------|
| `sentiment_combined` | Weighted average of all sentiment models (-1 to 1) |
| `sentiment_category` | Signal (High confidence) or Noise (Low confidence/relevance) |
| `ml_confidence` | How much the AI trusts its own prediction (0.0 to 1.0) |
| `std_channel_width` | Volatility metric (Bollinger Band width) |
| `rsi_14` | Relative Strength Index (Technical Indicator) |

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📄 License

This project is licensed under the MIT License.

---

## 📧 Contact

For questions or feedback, please open an issue on GitHub.
