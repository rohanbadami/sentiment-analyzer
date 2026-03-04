# db_mysql.py
import os
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

DB_USER = os.getenv("MYSQL_USER", "root")
DB_PASS = os.getenv("MYSQL_PASSWORD", "password") # Update this if needed
DB_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
DB_PORT = os.getenv("MYSQL_PORT", "3306")
DB_NAME = os.getenv("MYSQL_DATABASE", "sentiment_db")

def get_engine():
    # Construct connection string
    # Ensure you have installed: pip install mysql-connector-python sqlalchemy
    connection_string = f"mysql+mysqlconnector://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

    connect_args = {}
    ssl_ca = os.getenv("MYSQL_SSL_CA")
    if ssl_ca:
        # If the value looks like cert contents (not a file path), write to a temp file
        if ssl_ca.strip().startswith("-----"):
            import tempfile
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pem", mode="w")
            tmp.write(ssl_ca)
            tmp.close()
            ssl_ca = tmp.name
        connect_args["ssl_ca"] = ssl_ca
        connect_args["ssl_verify_cert"] = True

    try:
        engine = create_engine(
            connection_string,
            pool_recycle=3600,
            connect_args=connect_args,
        )
        return engine
    except Exception as e:
        print(f"Error creating engine: {e}")
        return None

def executemany_update(sql, params):
    """
    Executes a raw SQL update with a list of tuples (batch update).
    Now with proper error handling and guaranteed commits.
    """
    engine = get_engine()
    if not engine:
        logger.error("Failed to get database engine")
        return False

    if not params:
        logger.warning("No parameters provided to executemany_update")
        return True

    try:
        # Use raw_connection for DBAPI cursor with %s placeholders
        raw_conn = engine.raw_connection()
        try:
            cursor = raw_conn.cursor()
            
            # Execute the batch update
            cursor.executemany(sql, params)
            
            # CRITICAL: Explicitly commit the transaction
            raw_conn.commit()
            
            # Get affected rows count
            affected = cursor.rowcount
            logger.debug(f"Successfully updated {affected} rows")
            
            return True
            
        except Exception as cursor_error:
            # Rollback on error
            raw_conn.rollback()
            logger.error(f"❌ Error in cursor execution: {cursor_error}")
            logger.error(f"SQL: {sql}")
            logger.error(f"Sample params: {params[0] if params else 'None'}")
            return False
            
        finally:
            cursor.close()
            raw_conn.close()
            
    except Exception as e:
        logger.error(f"❌ Error executing batch update: {e}")
        return False

def execute_update(sql, params=None):
    """
    Execute a single SQL update with optional parameters.
    Alternative to executemany for single updates.
    """
    engine = get_engine()
    if not engine:
        return False
    
    try:
        with engine.begin() as conn:  # Auto-commits on success
            if params:
                conn.execute(text(sql), params)
            else:
                conn.execute(text(sql))
        return True
    except Exception as e:
        logger.error(f"❌ Error executing update: {e}")
        return False

def ensure_articles_table():
    """
    Creates the 'articles' table if it doesn't exist.
    Matches the 'Grand Unified Schema' from reset_db.py
    """
    engine = get_engine()
    if not engine:
        return

    create_table_sql = """
    CREATE TABLE IF NOT EXISTS articles (
        id INT AUTO_INCREMENT PRIMARY KEY,
        ticker VARCHAR(16),
        `datetime` DATETIME,
        headline TEXT,
        url VARCHAR(512) UNIQUE,
        `text` TEXT,
        
        -- KRIS & JOSH FEATURES
        tokens TEXT,
        mentions TEXT,
        pos_keywords TEXT,
        neg_keywords TEXT,
        total_keywords INT,
        text_length INT,
        keyword_density DOUBLE,
        sentiment_dynamic DOUBLE,
        sentiment_ml DOUBLE,
        sentiment_keyword DOUBLE,
        sentiment_combined DOUBLE,
        headline_sentiment DOUBLE,
        prediction_confidence DOUBLE,
        sentiment_category VARCHAR(64),
        ml_confidence DOUBLE,
        sentiment_strength DOUBLE,
        sentiment_score DOUBLE,

        -- BATES (GPT) & MIRZA (Prosus) FEATURES
        sentiment_gpt DOUBLE,
        gpt_reasoning TEXT,
        sentiment_vader DOUBLE,
        sentiment_finbert_tone DOUBLE,
        sentiment_finbert_prosus DOUBLE,
        
        -- MARKET DATA (Updated Schema)
        price_close DOUBLE,
        price_open DOUBLE,
        price_high DOUBLE,
        price_low DOUBLE,
        volume BIGINT,
        adj_close DOUBLE,
        
        -- Legacy columns for backwards compatibility
        `Close` DOUBLE,
        `Open` DOUBLE,
        `High` DOUBLE,
        `Low` DOUBLE,
        `Volume` DOUBLE,
        `Adj_Close` DOUBLE,
        
        -- Outcome columns
        pct_change_1h DOUBLE,
        pct_change_4h DOUBLE,
        pct_change_eod DOUBLE,
        pct_change_eow DOUBLE,
        direction_1h VARCHAR(16),
        direction_4h VARCHAR(16),
        direction_eod VARCHAR(16),
        direction_eow VARCHAR(16),
        
        -- Technical indicators
        rsi_14 DOUBLE,
        macd DOUBLE,
        macd_hist DOUBLE,
        price_vs_sma50 DOUBLE,
        std_upper DOUBLE,
        std_lower DOUBLE,
        std_channel_width DOUBLE,
        
        -- Market context
        vix_close DOUBLE,
        spy_daily_return DOUBLE,
        
        -- Temporal features
        hour_sin DOUBLE,
        hour_cos DOUBLE,
        day_of_week INT,
        
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_ticker (ticker),
        INDEX idx_datetime (`datetime`),
        INDEX idx_sentiment_category (sentiment_category),
        INDEX idx_pct_change_eod (pct_change_eod)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """
    
    try:
        with engine.begin() as conn:
            conn.execute(text(create_table_sql))
    except SQLAlchemyError as e:
        print(f"❌ Error creating table: {e}")

def bulk_insert_articles(articles_list):
    """
    Inserts a list of dictionaries. 
    Includes 'Smart Duplicate Handling': if a batch fails, it retries row-by-row.
    """
    if not articles_list:
        return

    df = pd.DataFrame(articles_list)
    engine = get_engine()
    
    if not engine:
        return

    try:
        # 1. Try inserting the whole batch efficiently
        df.to_sql('articles', con=engine, if_exists='append', index=False, chunksize=1000)
    except Exception as e:
        # 2. If we hit a duplicate error, switch to safe mode
        if "Duplicate entry" in str(e) or "1062" in str(e):
            # Fallback: Insert one by one to save the good ones and skip the bad ones
            with engine.begin() as conn:
                for _, row in df.iterrows():
                    try:
                        single_row_df = pd.DataFrame([row])
                        single_row_df.to_sql('articles', con=conn, if_exists='append', index=False)
                    except Exception:
                        # Silently ignore individual duplicates
                        pass
        else:
            print(f"❌ Critical DB Error: {e}")

def verify_outcomes_exist():
    """Helper function to check if outcomes are populated."""
    engine = get_engine()
    if not engine:
        return 0
    
    try:
        with engine.connect() as conn:
            result = conn.execute(text(
                "SELECT COUNT(*) FROM articles WHERE pct_change_eod IS NOT NULL"
            ))
            return result.scalar()
    except Exception as e:
        logger.error(f"Error checking outcomes: {e}")
        return 0

if __name__ == "__main__":
    print("Initializing MySQL connection…")
    engine = get_engine()
    print("Connected to MySQL!")
    ensure_articles_table()
    print("Table schema checked.")
    
    # Test outcomes
    outcome_count = verify_outcomes_exist()
    print(f"Articles with outcomes: {outcome_count}")