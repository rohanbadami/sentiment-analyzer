# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Financial sentiment analysis pipeline that scrapes news headlines, scores them with ensemble NLP (FinBERT + VADER + custom dictionary), validates against historical price action, and filters through a LightGBM "Gatekeeper" classifier to surface high-confidence trading signals on a Streamlit dashboard.

## Running the Pipeline

Scripts must be run **in order** — each phase depends on the previous one. All intermediate data is stored in a central MySQL database (`sentiment_db`).

```bash
# Phase 0: Filter active tickers
python ticker_filter.py

# Phase 1: Scrape headlines from Finviz
python phase1_headline_scraper.py

# Phase 2: Score sentiment (ensemble NLP, 8-worker multiprocessing)
python phase2_sentiment_analysis.py

# Phase 3: Fetch prices + technical indicators (RSI, MACD, Bollinger, SMA-50)
python phase3_price_integration.py

# Phase 4a: Train Gatekeeper classifier (target: price move > 1.5%)
python phase4_classifier_mysql.py eod

# Phase 4b: (Optional) Train price magnitude regressor
python phase4_regressor_mysql.py eod

# Phase 4c: Run inference — apply trained model to all articles
python phase4_backfill_predictions.py

# Phase 5: Launch dashboard
streamlit run phase5_dashboard.py
```

### Utilities
```bash
python check_db.py                    # System status report
python reset_database.py              # Hard reset (keeps headlines, wipes derived data)
python export_db_to_csv_enhanced.py   # Export to CSV
```

## Architecture

**5-phase sequential pipeline**, all data flowing through MySQL:

```
ticker_filter.py → phase1 (scrape) → phase2 (sentiment) → phase3 (prices) → phase4 (ML) → phase5 (dashboard)
```

### Core Modules
- **`db_mysql.py`** — SQLAlchemy database handler. Connection config from `.env` (MYSQL_HOST, USER, PASSWORD, DATABASE, PORT).
- **`integrated_processor.py`** — NLP orchestrator: weighted voting across FinBERT, VADER, and custom keyword dictionary. Handles question-format headline detection.
- **`sentiment_scorer.py`** — Wraps ProsusAI/finbert transformer and VADER. Returns `SentimentResult(vader, finbert)` normalized to [-1, 1]. Auto-detects CUDA.
- **`unified_price_scripts.py`** — Rate-limited yfinance wrapper with file-based caching (6-24hr TTL) and retry logic.
- **`sentiment_keywords.csv`** — Custom NLP dictionary: 200+ keywords with sentiment polarity and strength scores.

### Key Design Decisions
- **Weekend Patch**: Weekend news is mapped to Friday's market close to prevent data gaps during inference (implemented in phase3 and phase4_backfill).
- **Anti-overfitting**: The Gatekeeper model explicitly excludes VIX and calendar features, forcing it to learn from news content.
- **Trained models** are saved to `models/` directory as `.pkl` files via joblib.
- **Batch processing**: Phase 2 uses batch size 5000; Phase 4c uses batch size 5000.

## Dependencies

Python 3.12+, MySQL server, dependencies in `requirements.txt`. Optional CUDA GPU for faster FinBERT inference.

```bash
pip install -r requirements.txt
```

## Database

Single `articles` table with 40+ columns tracking the full lifecycle: raw headline → sentiment scores (6 per article) → price data + technicals → ML confidence score. Schema auto-created by `db_mysql.ensure_articles_table()`.
