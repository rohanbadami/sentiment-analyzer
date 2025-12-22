import pandas as pd
import pandas_ta as ta
import numpy as np
import logging
import pytz
import yfinance as yf
from datetime import timedelta
from sqlalchemy import text
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Import your existing project modules
from db_mysql import get_engine, executemany_update
from unified_price_scripts import get_price_history

# === CONFIGURATION ===
MAX_WORKERS = 5
BATCH_SIZE = 500
LOOKBACK_DAYS = 365 * 3
EASTERN = pytz.timezone('US/Eastern')

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Phase3_Full_Fix")

# ==========================================
#        SCHEMA CHECK & SELF-HEALING
# ==========================================
def ensure_all_columns_exist():
    engine = get_engine()
    columns = [
        # STD Channels
        "std_upper DOUBLE DEFAULT NULL",
        "std_lower DOUBLE DEFAULT NULL",
        "std_channel_width DOUBLE DEFAULT NULL",
        # Technical Indicators
        "rsi_14 DOUBLE DEFAULT NULL",
        "macd DOUBLE DEFAULT NULL",
        "macd_hist DOUBLE DEFAULT NULL",
        "price_vs_sma50 DOUBLE DEFAULT NULL",
        # Market Context
        "vix_close DOUBLE DEFAULT NULL",
        "spy_daily_return DOUBLE DEFAULT NULL",
        # Temporal Features
        "hour_sin DOUBLE DEFAULT NULL",
        "hour_cos DOUBLE DEFAULT NULL",
        "day_of_week INT DEFAULT NULL"
    ]
    with engine.connect() as conn:
        for col in columns:
            col_name = col.split()[0]
            try:
                result = conn.execute(text(f"SHOW COLUMNS FROM articles LIKE '{col_name}'"))
                if not result.fetchone():
                    logger.info(f"🔧 Adding missing column: {col_name}...")
                    conn.execute(text(f"ALTER TABLE articles ADD COLUMN {col}"))
                    logger.info(f"✅ Column added: {col_name}")
            except Exception as e:
                logger.debug(f"Column check failed for {col_name}: {e}")

# ==========================================
#          PRICE & TECHNICALS REPAIR
# ==========================================
def get_all_tickers():
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text(
            "SELECT DISTINCT ticker FROM articles WHERE ticker IS NOT NULL AND ticker != ''"
        ))
        return [r[0] for r in result]

def get_articles_for_ticker(ticker):
    engine = get_engine()
    query = text("SELECT id, datetime FROM articles WHERE ticker = :t")
    return pd.read_sql(query, engine, params={"t": ticker})

def fetch_continuous_market_data(ticker):
    yahoo_ticker = ticker.replace('.', '-')
    end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
    start_date = (pd.Timestamp.now() - timedelta(days=LOOKBACK_DAYS)).strftime('%Y-%m-%d')
    try:
        df = get_price_history(yahoo_ticker, start_date, end_date, interval="1d")
        if df.empty: return df
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None).dt.normalize()
        df = df.set_index('date').sort_index()
        return df
    except Exception:
        return pd.DataFrame()

def process_ticker_repair(ticker):
    try:
        articles_df = get_articles_for_ticker(ticker)
        if articles_df.empty: return []

        prices_df = fetch_continuous_market_data(ticker)
        if prices_df.empty or len(prices_df) < 15: 
            return []

        # === Vectorized Technicals on full price history
        prices_df['rsi_14'] = ta.rsi(prices_df['close'], length=14)
        macd = ta.macd(prices_df['close'])
        prices_df['macd'] = macd['MACD_12_26_9'] if macd is not None else None
        prices_df['macd_hist'] = macd['MACDh_12_26_9'] if macd is not None else None
        sma50 = ta.sma(prices_df['close'], length=50)
        prices_df['price_vs_sma50'] = (prices_df['close'] - sma50) / sma50 if sma50 is not None else None

        rolling_mean = prices_df['close'].rolling(20).mean()
        rolling_std = prices_df['close'].rolling(20).std()
        prices_df['std_upper'] = rolling_mean + 2 * rolling_std
        prices_df['std_lower'] = rolling_mean - 2 * rolling_std
        prices_df['std_channel_width'] = np.where(
            prices_df['close'] != 0,
            (prices_df['std_upper'] - prices_df['std_lower']) / prices_df['close'],
            0
        )

        # Map back to articles
        updates = []
        price_dates_np = np.array(prices_df.index.values, dtype='datetime64[ns]')
        for _, row in articles_df.iterrows():
            try:
                ts = pd.to_datetime(row['datetime'])
                utc_dt = ts if ts.tzinfo else ts.tz_localize('UTC')
                target_date = utc_dt.astimezone(EASTERN).date()
                target_np = np.datetime64(pd.Timestamp(target_date))

                loc_idx = price_dates_np.searchsorted(target_np, side='right') - 1
                if loc_idx < 0: continue

                matched_date = pd.Timestamp(price_dates_np[loc_idx])
                if (pd.Timestamp(target_date) - matched_date).days > 5: continue

                day_data = prices_df.iloc[loc_idx]
                def val(col): return float(day_data[col]) if col in day_data and not pd.isna(day_data[col]) else None

                updates.append((
                    val('std_upper'), val('std_lower'), val('std_channel_width'),
                    val('rsi_14'), val('macd'), val('macd_hist'), val('price_vs_sma50'),
                    int(row['id'])
                ))
            except Exception: continue

        return updates
    except Exception:
        return []

def push_price_updates(updates):
    clean_updates = [u for u in updates if u[-1] is not None]
    sql = """
        UPDATE articles 
        SET std_upper=%s, std_lower=%s, std_channel_width=%s,
            rsi_14=%s, macd=%s, macd_hist=%s, price_vs_sma50=%s
        WHERE id=%s
    """
    try:
        executemany_update(sql, clean_updates)
        logger.info(f"💾 Saved {len(clean_updates)} price/technical records.")
    except Exception as e:
        logger.error(f"Write Error: {e}")

# ==========================================
#            TEMPORAL FEATURES
# ==========================================
def calculate_temporal_features():
    engine = get_engine()
    query = text("SELECT id, datetime FROM articles WHERE hour_sin IS NULL AND datetime IS NOT NULL")
    with engine.connect() as conn:
        result = conn.execute(query).fetchall()
    if not result: return
    updates = []
    for row in tqdm(result, desc="Temporal"):
        try:
            article_id, dt = row[0], row[1]
            dt_obj = pd.to_datetime(dt)
            hour = dt_obj.hour
            updates.append((
                float(np.sin(2*np.pi*hour/24)),
                float(np.cos(2*np.pi*hour/24)),
                int(dt_obj.dayofweek),
                int(article_id)
            ))
            if len(updates) >= BATCH_SIZE:
                push_temporal_updates(updates)
                updates = []
        except Exception: continue
    if updates: push_temporal_updates(updates)

def push_temporal_updates(updates):
    sql = "UPDATE articles SET hour_sin=%s, hour_cos=%s, day_of_week=%s WHERE id=%s"
    try:
        executemany_update(sql, updates)
        logger.info(f"💾 Saved {len(updates)} temporal records.")
    except Exception as e:
        logger.error(f"Temporal Write Error: {e}")

# ==========================================
#            MARKET CONTEXT (VIX/SPY)
# ==========================================
def process_market_context():
    engine = get_engine()
    with engine.connect() as conn:
        if conn.execute(text("SELECT count(*) FROM articles WHERE vix_close IS NULL")).scalar() == 0:
            return

    end_date = pd.Timestamp.now()
    start_date = end_date - timedelta(days=365*3)
    try:
        vix = yf.download("^VIX", start=start_date, end=end_date, progress=False)['Close']
        spy = yf.download("SPY", start=start_date, end=end_date, progress=False)['Close']
        spy_returns = spy.pct_change()
        vix.index = pd.to_datetime(vix.index)
        spy_returns.index = pd.to_datetime(spy_returns.index)

        dates_query = "SELECT DISTINCT DATE(datetime) as date_str FROM articles WHERE vix_close IS NULL"
        dates = pd.read_sql(dates_query, engine)['date_str'].tolist()

        vix_dates_np = np.array(vix.index.date, dtype='datetime64[D]')
        for d in tqdm(dates, desc="Market Context"):
            target_np = np.datetime64(d)
            idx = vix_dates_np.searchsorted(target_np, side='right') - 1
            if idx >= 0:
                found_date = vix.index[idx].date()
                if (d - found_date).days <= 5:
                    v_val = float(vix.iloc[idx])
                    s_val = float(spy_returns.asof(d) or 0.0)
                    with engine.connect() as conn:
                        conn.execute(text(f"UPDATE articles SET vix_close={v_val}, spy_daily_return={s_val} WHERE DATE(datetime)='{d}'"))
                        conn.commit()
    except Exception as e:
        logger.error(f"Market Context Error: {e}")

# ==========================================
#                 MAIN
# ==========================================
def main():
    logger.info("🚀 Starting Phase 3 Full Pipeline")
    ensure_all_columns_exist()

    # PRICE & TECHNICALS
    tickers = get_all_tickers()
    update_buffer = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(tqdm(executor.map(process_ticker_repair, tickers), total=len(tickers), unit="ticker"))
        for res in results:
            if res:
                update_buffer.extend(res)
                if len(update_buffer) >= BATCH_SIZE:
                    push_price_updates(update_buffer)
                    update_buffer = []
    if update_buffer:
        push_price_updates(update_buffer)

    # TEMPORAL
    calculate_temporal_features()

    # MARKET CONTEXT
    process_market_context()

    logger.info("✅ Phase 3 Fully Complete.")

if __name__ == "__main__":
    main()
