# db_mysql.py
import os
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv

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
    try:
        engine = create_engine(connection_string, pool_recycle=3600)
        return engine
    except Exception as e:
        print(f"Error creating engine: {e}")
        return None

def executemany_update(sql, params):
    """
    Executes a raw SQL update with a list of tuples (batch update).
    Uses engine.raw_connection() to support %s style placeholders 
    compatible with mysql-connector for high performance.
    """
    engine = get_engine()
    if not engine:
        return

    try:
        # We use raw_connection to get a DBAPI connection 
        # that supports standard cursor.executemany with %s syntax
        conn = engine.raw_connection()
        try:
            cursor = conn.cursor()
            cursor.executemany(sql, params)
            conn.commit()
        finally:
            cursor.close()
            conn.close()
    except Exception as e:
        print(f"❌ Error executing batch update: {e}")

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
        
        -- MARKET DATA
        `Close` DOUBLE,
        `Open` DOUBLE,
        `High` DOUBLE,
        `Low` DOUBLE,
        `Volume` DOUBLE,
        `Adj_Close` DOUBLE,
        pct_change_1h DOUBLE,
        pct_change_4h DOUBLE,
        pct_change_eod DOUBLE,
        pct_change_eow DOUBLE,
        direction_1h VARCHAR(16),
        direction_4h VARCHAR(16),
        direction_eod VARCHAR(16),
        direction_eow VARCHAR(16),
        
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_ticker (ticker),
        INDEX idx_datetime (`datetime`)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """
    
    try:
        with engine.connect() as conn:
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
            with engine.connect() as conn:
                for _, row in df.iterrows():
                    try:
                        single_row_df = pd.DataFrame([row])
                        single_row_df.to_sql('articles', con=conn, if_exists='append', index=False)
                    except Exception:
                        # Silently ignore individual duplicates
                        pass
        else:
            print(f"❌ Critical DB Error: {e}")

if __name__ == "__main__":
    print("Initializing MySQL connection…")
    engine = get_engine()
    print("Connected to MySQL!")
    ensure_articles_table()
    print("Table schema checked.")