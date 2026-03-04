"""
config.py - Central configuration for the gossip detection system.

Loads environment variables following the pattern from db_mysql.py.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================================
# SUBREDDIT CONFIGURATION
# ============================================================================
SUBREDDITS = [
    "wallstreetbets",
    "StockMarket",
    "stocks",
    "RealDayTrading",
    "investing",
    "robinhood"
]

# ============================================================================
# POLLING CONFIGURATION
# ============================================================================
POLL_INTERVAL_SECONDS = 60
ROLLING_WINDOWS = [5, 15, 60]  # minutes
BATCH_SIZE = 100
MAX_WORKERS = 5

# ============================================================================
# API ENDPOINTS
# ============================================================================
PULLPUSH_BASE_URL = "https://api.pullpush.io/reddit"
ARCTIC_SHIFT_BASE_URL = "https://arctic-shift.photon-reddit.com/api"

# ============================================================================
# REDIS CONFIGURATION
# ============================================================================
REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

# ============================================================================
# MYSQL CONFIGURATION (reused from db_mysql.py pattern)
# ============================================================================
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "password")
MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = os.getenv("MYSQL_PORT", "3306")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "sentiment_db")

# ============================================================================
# ANOMALY DETECTION THRESHOLDS
# ============================================================================
Z_SCORE_THRESHOLD = 2.5
MIN_BASELINE_POINTS = 10

# ============================================================================
# GOSSIP SCORING WEIGHTS (normalized to sum = 1.0)
# ============================================================================
GOSSIP_SCORE_WEIGHTS = {
    "velocity": 0.25,
    "acceleration": 0.25,
    "author_diversity": 0.15,
    "anomaly": 0.20,
    "confirmation": 0.15,
}

# Verify weights sum to 1.0
assert abs(sum(GOSSIP_SCORE_WEIGHTS.values()) - 1.0) < 0.001, "Gossip score weights must sum to 1.0"

# ============================================================================
# SCORE NORMALIZATION LIMITS (for individual sub-scores)
# ============================================================================
VELOCITY_MAX_MENTIONS_PER_MIN = 10.0
ACCELERATION_MAX = 5.0
AUTHOR_DIVERSITY_MAX = 50

# ============================================================================
# REDIS KEY PREFIXES
# ============================================================================
REDIS_PREFIX_LATEST_PULL = "gossip:latest_pull"
REDIS_PREFIX_LATEST_PULL_TS = "gossip:latest_pull_ts"
REDIS_PREFIX_TICKER_SNAPSHOT = "gossip:ticker"
REDIS_PREFIX_ALERTS = "gossip:alerts"

# ============================================================================
# REDIS TTL (seconds)
# ============================================================================
REDIS_TTL_LATEST_PULL = 600  # 10 minutes
REDIS_TTL_TICKER_SNAPSHOT = 300  # 5 minutes

# ============================================================================
# ALERT THRESHOLDS
# ============================================================================
ALERT_GOSSIP_SCORE_THRESHOLD = 0.7
ALERT_CONFIRMATION_STRENGTH_THRESHOLD = 0.5

# ============================================================================
# CHECKPOINT FILE
# ============================================================================
CHECKPOINT_FILE = "gossip_detection/.checkpoint_reddit_poller.json"
