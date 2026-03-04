# =====================================================
# STREAMLIT APP: DIRECTIONAL TRADING + RANDOM FOREST REGIME
# =====================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import class_weight
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="BIST 100 Directional Trading", layout="wide")

# =====================================================
# 1. CONFIGURATION — SIDEBAR
# =====================================================
st.sidebar.header("Configuration")

TICKER           = st.sidebar.text_input("Ticker", value="XU100.IS")
START            = st.sidebar.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
END              = st.sidebar.date_input("End Date",   value=pd.to_datetime("2027-02-28"))
TRANSACTION_COST = st.sidebar.number_input(
    "Transaction Cost per Trade", value=0.0, step=0.001, format="%.3f")

st.sidebar.subheader("Peak / Trough Detection")
SMOOTHING_SIGMA    = st.sidebar.slider("Smoothing Sigma",    min_value=1,   max_value=10,  value=3,   step=1)
MIN_DISTANCE       = st.sidebar.slider("Min Peak Distance",  min_value=3,   max_value=30,  value=10,  step=1)
PROMINENCE_FACTOR  = st.sidebar.slider("Prominence Factor",  min_value=0.1, max_value=2.0, value=0.5, step=0.1)

st.sidebar.subheader("Aspect Orbs")
ORB_APPLY = st.sidebar.slider(
    "Applying Orb (degrees)",
    min_value=0.5, max_value=10.0, value=4.4, step=0.1,
    help="Max degrees a planet can be from exact aspect while still approaching it.")
ORB_SEP = st.sidebar.slider(
    "Separating Orb (degrees)",
    min_value=0.0, max_value=5.0, value=0.3, step=0.1,
    help="Max degrees a planet can be past exact aspect while still separating. "
         "Set to 0 to require exact-only separation.")

st.title(f"Directional Trading on {TICKER}")
st.write(
    f"Period: {START} to {END}  |  "
    f"Transaction Cost: {TRANSACTION_COST*100:.2f}%  |  "
    f"Applying Orb: {ORB_APPLY}°  |  Separating Orb: {ORB_SEP}°"
)

# =====================================================
# 2. DOWNLOAD DATA
# =====================================================
@st.cache_data
def download_stock_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if data.empty:
        return None
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data[['Open', 'High', 'Low', 'Close']].dropna()
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
    data = data.dropna(subset=['Close'])
    return data

data = download_stock_data(TICKER, START, END)

if data is None:
    st.error(f"No data found for {TICKER}. Exiting.")
    st.stop()

st.success(f"Downloaded {len(data)} trading days from {data.index[0].date()} to {data.index[-1].date()}")

close_prices = data['Close']

# =====================================================
# 3. PEAK / TROUGH DETECTION
# =====================================================
st.subheader("Peak/Trough Detection")

def detect_peaks_and_troughs(prices, smoothing_sigma=3, min_distance=10, prominence_factor=0.5):
    prices_smooth = gaussian_filter1d(prices.values, sigma=smoothing_sigma)
    volatility = prices.pct_change().std()
    avg_price = prices.mean()
    prominence_threshold = avg_price * volatility * prominence_factor

    peaks, peak_props = find_peaks(prices_smooth, distance=min_distance, prominence=prominence_threshold)
    troughs, trough_props = find_peaks(-prices_smooth, distance=min_distance, prominence=prominence_threshold)

    return {
        'prices_smooth': prices_smooth,
        'peaks': peaks,
        'troughs': troughs,
        'peak_properties': peak_props,
        'trough_properties': trough_props
    }

detection = detect_peaks_and_troughs(
    close_prices,
    smoothing_sigma=SMOOTHING_SIGMA,
    min_distance=MIN_DISTANCE,
    prominence_factor=PROMINENCE_FACTOR
)

signal_events = []

for idx in detection['peaks']:
    signal_events.append({
        'Date': close_prices.index[idx],
        'Type': 'PEAK',
        'Signal': 'SHORT',
        'Price': close_prices.iloc[idx]
    })

for idx in detection['troughs']:
    signal_events.append({
        'Date': close_prices.index[idx],
        'Type': 'TROUGH',
        'Signal': 'LONG',
        'Price': close_prices.iloc[idx]
    })

signal_events.sort(key=lambda x: x['Date'])
signals_df = pd.DataFrame(signal_events)
st.write(signals_df.head(10))

# =====================================================
# 4. BACKTEST STRATEGY
# =====================================================
st.subheader("Backtesting")

def backtest_directional_strategy(prices, peaks, troughs, transaction_cost=0.001):
    n = len(prices)
    returns = np.zeros(n)
    positions = np.zeros(n)
    trades = []

    all_signals = [(i, -1, 'PEAK') for i in peaks] + [(i, 1, 'TROUGH') for i in troughs]
    all_signals.sort(key=lambda x: x[0])

    current_position = 0
    entry_price = 0
    entry_date = None
    entry_signal = None

    for signal_idx, pos_type, sig_name in all_signals:
        signal_date = prices.index[signal_idx]
        signal_price = prices.iloc[signal_idx]
        if pos_type != current_position:
            if current_position != 0:
                price_change = (signal_price - entry_price) / entry_price
                exit_return = price_change - transaction_cost if current_position==1 else -price_change - transaction_cost
                trades.append({
                    'Entry_Date': entry_date,
                    'Entry_Signal': entry_signal,
                    'Entry_Price': entry_price,
                    'Exit_Date': signal_date,
                    'Exit_Signal': sig_name,
                    'Exit_Price': signal_price,
                    'Position': 'LONG' if current_position==1 else 'SHORT',
                    'Return_%': exit_return*100,
                    'Holding_Days': (signal_date-entry_date).days
                })
            current_position = pos_type
            entry_price = signal_price
            entry_date = signal_date
            entry_signal = sig_name
        positions[signal_idx:] = current_position

    for i in range(1,n):
        if positions[i-1]==1:
            returns[i] = (prices.iloc[i]-prices.iloc[i-1])/prices.iloc[i-1]
        elif positions[i-1]==-1:
            returns[i] = -(prices.iloc[i]-prices.iloc[i-1])/prices.iloc[i-1]

    cum_log_returns = np.cumsum(np.log1p(returns))
    cumulative_wealth = np.exp(cum_log_returns)
    running_max = np.maximum.accumulate(cumulative_wealth)
    drawdown = (cumulative_wealth - running_max)/running_max

    total_return = cumulative_wealth[-1]-1
    num_years = len(prices)/252
    annualized_return = (1+total_return)**(1/num_years)-1
    sharpe_ratio = np.mean(returns)/ (np.std(returns)+1e-10) * np.sqrt(252)
    max_drawdown = np.min(drawdown)
    winning_trades = [t for t in trades if t['Return_%']>0]
    losing_trades  = [t for t in trades if t['Return_%']<=0]
    win_rate = len(winning_trades)/len(trades) if len(trades)>0 else 0
    total_wins = sum([t['Return_%'] for t in winning_trades])
    total_losses = abs(sum([t['Return_%'] for t in losing_trades]))
    profit_factor = total_wins/total_losses if total_losses>0 else np.inf
    buy_hold_return = (prices.iloc[-1]-prices.iloc[0])/prices.iloc[0]
    outperformance = total_return - buy_hold_return

    return {
        'returns': returns,
        'cumulative_log_returns': cum_log_returns,
        'positions': positions,
        'trades': trades,
        'metrics': {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': len(trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'buy_hold_return': buy_hold_return,
            'outperformance': outperformance
        }
    }

backtest_result = backtest_directional_strategy(
    prices=close_prices,
    peaks=detection['peaks'],
    troughs=detection['troughs'],
    transaction_cost=TRANSACTION_COST
)

metrics = backtest_result['metrics']
st.write("### Performance Metrics")
st.json(metrics)

# =====================================================
# 5. PLOTS (with Regression Channel)
# =====================================================
st.subheader("Price & Signals with Regression Channel")
fig, axes = plt.subplots(2,1,figsize=(16,10))

from sklearn.linear_model import LinearRegression

x_numeric = np.arange(len(close_prices)).reshape(-1,1)
y = close_prices.values
reg = LinearRegression().fit(x_numeric, y)
y_fit = reg.predict(x_numeric)

residuals = y - y_fit
sigma = np.std(residuals)
upper_band = y_fit + 2*sigma
lower_band = y_fit - 2*sigma

axes[0].plot(close_prices.index, close_prices.values, label='Actual Price')
axes[0].plot(close_prices.index, detection['prices_smooth'], label='Smoothed Price', linewidth=2)
axes[0].scatter(close_prices.index[detection['peaks']], close_prices.iloc[detection['peaks']],
                color='red', marker='v', s=80, label='PEAK')
axes[0].scatter(close_prices.index[detection['troughs']], close_prices.iloc[detection['troughs']],
                color='green', marker='^', s=80, label='TROUGH')
axes[0].plot(close_prices.index, y_fit, color='blue', linestyle='--', linewidth=2, label='Regression Line')
axes[0].fill_between(close_prices.index, lower_band, upper_band, color='blue', alpha=0.15, label='±2 Sigma')
axes[0].legend(); axes[0].grid(True)

axes[1].plot(close_prices.index, backtest_result['cumulative_log_returns'], label='Strategy')
axes[1].fill_between(close_prices.index, 0, backtest_result['cumulative_log_returns'],
                     where=backtest_result['positions']==1, color='green', alpha=0.2)
axes[1].fill_between(close_prices.index, 0, backtest_result['cumulative_log_returns'],
                     where=backtest_result['positions']==-1, color='red', alpha=0.2)
axes[1].axhline(0, color='black', linestyle='--')
axes[1].legend(); axes[1].grid(True)

st.pyplot(fig)


# =====================================================
# 6. BUILD REGIME FROM PEAK/TROUGH DETECTION
#    (replaces the dummy next-day-return label)
# =====================================================

def build_regime(prices, detection):
    """
    State-machine regime consistent with the backtest and multi-ticker code:
      1 = LONG  (from trough → next peak)
      0 = SHORT (from peak  → next trough)
    This ensures the RF is trained on the same signal used by the backtest,
    fixing the bug where the old code ignored peak/trough detection entirely.
    """
    n           = len(prices)
    peaks_set   = set(detection['peaks'])
    troughs_set = set(detection['troughs'])
    regime      = np.zeros(n, dtype=int)
    current_state = 1          # start assuming LONG
    for i in range(n):
        regime[i] = current_state
        if current_state == 1 and i in peaks_set:
            current_state = 0
        elif current_state == 0 and i in troughs_set:
            current_state = 1
    return pd.Series(regime, index=prices.index, name='Regime')


# =====================================================
# 7. RANDOM FOREST REGIME MODEL
# =====================================================
st.subheader("Random Forest Regime Prediction")

# Feature Engineering
df_rf = data.copy()
df_rf['Return'] = df_rf['Close'].pct_change()
df_rf['CumRet_3'] = df_rf['Return'].rolling(3).sum()
df_rf['CumRet_4'] = df_rf['Return'].rolling(4).sum()
df_rf['CumRet_5'] = df_rf['Return'].rolling(5).sum()
df_rf['CumRet_14'] = df_rf['Return'].rolling(14).sum()
delta = df_rf['Close'].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.rolling(9).mean()
avg_loss = loss.rolling(9).mean()
rs = avg_gain/(avg_loss+1e-10)
df_rf['RSI_14'] = 100-(100/(1+rs))
df_rf['High_4'] = df_rf['High'].rolling(4).max()
df_rf['Low_4'] = df_rf['Low'].rolling(4).min()
df_rf['High_14'] = df_rf['High'].rolling(14).max() - df_rf['High'].rolling(2).max()
df_rf['Low_14'] = df_rf['Low'].rolling(14).min() - df_rf['Low'].rolling(2).min()

# ── FIX: assign regime from peak/trough state machine ──────────────────────
df_rf['Regime'] = build_regime(close_prices, detection).reindex(df_rf.index)
# ───────────────────────────────────────────────────────────────────────────

df_rf = df_rf.dropna()

FEATURES = ['CumRet_3','CumRet_4','CumRet_5','High_4','Low_4','RSI_14','High_14','Low_14']
TARGET = 'Regime'

X = df_rf[FEATURES].values
y = df_rf[TARGET].values

split = int(0.8*len(df_rf))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Check unique classes
unique_classes = np.unique(y_train)
rf_trained = False
if len(unique_classes) < 2:
    st.warning("Not enough class variation in training data to train RF.")
else:
    class_weights_array = class_weight.compute_class_weight('balanced', classes=unique_classes, y=y_train)
    class_weights_dict = dict(zip(unique_classes, class_weights_array))
    rf = RandomForestClassifier(n_estimators=500, max_depth=5, min_samples_leaf=5,
                                class_weight=class_weights_dict, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_trained = True
    st.success("Random Forest trained successfully.")


# =====================================================
# 8. CONFUSION MATRICES & CLASSIFICATION REPORTS
# =====================================================
if rf_trained:
    st.subheader("Confusion Matrices & Classification Reports")

    y_pred_train = rf.predict(X_train)
    y_pred_test = rf.predict(X_test)

    fig, ax = plt.subplots(1,2,figsize=(14,5))
    disp_train = ConfusionMatrixDisplay(confusion_matrix(y_train, y_pred_train))
    disp_train.plot(ax=ax[0], colorbar=False)
    ax[0].set_title("In-Sample Confusion Matrix")

    disp_test = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_test))
    disp_test.plot(ax=ax[1], colorbar=False)
    ax[1].set_title("Out-of-Sample Confusion Matrix")
    st.pyplot(fig)

    report_train = classification_report(y_train, y_pred_train, output_dict=True)
    report_test = classification_report(y_test, y_pred_test, output_dict=True)

    st.write("### In-Sample Classification Report")
    st.dataframe(pd.DataFrame(report_train).transpose().round(3))

    st.write("### Out-of-Sample Classification Report")
    st.dataframe(pd.DataFrame(report_test).transpose().round(3))

    # Latest Prediction
    latest_row = df_rf.iloc[[-1]]
    latest_date = latest_row.index[0]
    latest_close = latest_row['Close'].values[0]
    X_latest = scaler.transform(latest_row[FEATURES].values)
    prob_latest = rf.predict_proba(X_latest)[:,1][0]
    position_latest = 'HOLD'
    if prob_latest >= 0.6: position_latest='LONG'
    elif prob_latest <=0.4: position_latest='SHORT'

    st.write(f"**Latest Date:** {latest_date.date()}, **Close Price:** {latest_close:.2f}")
    st.write(f"Predicted Probability of LONG: {prob_latest:.4f}")
    st.write(f"Recommended Position: {position_latest}")



# # ============================================================
# #  PLANETARY ASPECT SCORER — Streamlit App
# #
# #  Ephemeris loaded from GitHub (planet_degrees.csv)
# #  User inputs: ticker, natal date, data start, chart end,
# #               table horizon, orb apply/sep
# #  Outputs: 3 charts + 2 aspect tables
# # ============================================================

# import warnings, datetime, itertools
# warnings.filterwarnings('ignore')

# import numpy as np
# import pandas as pd
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
# import matplotlib.patches as mpatches
# from matplotlib.lines import Line2D
# import yfinance as yf
# import streamlit as st

# # ============================================================
# #  PAGE CONFIG
# # ============================================================

# st.set_page_config(
#     page_title="🪐 Planetary Aspect Scorer",
#     page_icon="🪐",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# st.markdown("""
# <style>
#   /* Dark background to match chart aesthetic */
#   .stApp { background-color: #0A0A1A; color: #E8E8F4; }
#   section[data-testid="stSidebar"] { background-color: #0D0D28; }
#   section[data-testid="stSidebar"] * { color: #E8E8F4 !important; }
#   .stTextInput > div > div > input,
#   .stNumberInput > div > div > input,
#   .stDateInput > div > div > input {
#       background-color: #1A1A38;
#       color: #E8E8F4;
#       border: 1px solid #2A2A4A;
#   }
#   .stSlider > div { color: #E8E8F4; }
#   .stButton > button {
#       background-color: #C8A84B;
#       color: #0A0A1A;
#       font-weight: bold;
#       border: none;
#       border-radius: 4px;
#       padding: 0.5rem 2rem;
#       width: 100%;
#   }
#   .stButton > button:hover { background-color: #E8C86B; }
#   h1, h2, h3 { color: #C8A84B !important; }
#   .stDataFrame { background-color: #0D0D28; }
#   div[data-testid="stMetric"] {
#       background-color: #0D0D28;
#       border: 1px solid #2A2A4A;
#       border-radius: 6px;
#       padding: 0.5rem 1rem;
#   }
#   div[data-testid="stMetric"] label { color: #C8A84B !important; }
# </style>
# """, unsafe_allow_html=True)

# # ============================================================
# #  EPHEMERIS — loaded from GitHub (cached)
# # ============================================================

# GITHUB_EPH_URL = (
#     "https://raw.githubusercontent.com/"
#     "goncuahm/kozmik_finans/main/planet_degrees.csv"
#     # ↑ Replace with your actual GitHub raw URL
# )

# @st.cache_data(show_spinner="Loading ephemeris from GitHub …")
# def load_ephemeris(url):
#     eph_raw = pd.read_csv(url, index_col='date', parse_dates=True)
#     return eph_raw

# # ============================================================
# #  SIDEBAR — USER INPUTS
# # ============================================================

# with st.sidebar:
#     st.markdown("## 🪐 Planetary Aspect Scorer")
#     st.markdown("---")

#     st.markdown("### 📈 Asset")
#     ticker = st.text_input(
#         "Ticker (yfinance)", value="GLD",
#         help="Any yfinance ticker: GLD, AAPL, XU100.IS, BTC-USD …")

#     st.markdown("### 🌟 Natal Chart")
#     natal_date_input = st.text_input(
#         "Natal / birth date (YYYY-MM-DD)",
#         value="1933-01-30",
#         help="Founding or listing date of the asset. Leave blank to skip natal aspects.")

#     st.markdown("### 📅 Date Range")
#     data_start = st.text_input(
#         "Price data start (YYYY-MM-DD)",
#         value="2023-01-01",
#         help="Start date for downloading OHLC price data.")

#     chart_end_input = st.text_input(
#         "Chart end / forecast to (YYYY-MM-DD)",
#         value=(datetime.date.today() + datetime.timedelta(days=365)).strftime("%Y-%m-%d"),
#         help="Extend charts into the future to show upcoming aspect scores.")

#     st.markdown("### 🔭 Orb Settings")
#     orb_apply = st.slider(
#         "Applying orb (degrees)", min_value=1, max_value=6, value=4,
#         help="How many degrees before exact to start counting an aspect.")
#     orb_sep = st.slider(
#         "Separating orb (degrees)", min_value=1, max_value=3, value=1,
#         help="How many degrees after exact to keep counting an aspect.")

#     st.markdown("### 📋 Table Horizon")
#     table_days = st.slider(
#         "Days ahead in tables", min_value=7, max_value=60, value=15,
#         help="How many future days to include in the aspect tables.")

#     st.markdown("---")
#     run_btn = st.button("▶  Run Analysis", type="primary")

# # ============================================================
# #  HEADER
# # ============================================================

# st.markdown("# 🪐 Planetary Aspect Scorer")
# st.markdown(
#     "Scores daily planetary aspects (natal × transit and transit × transit) "
#     "and overlays them on candlestick price charts. "
#     "Positive = bullish planetary conditions. Negative = bearish."
# )

# if not run_btn:
#     st.info("👈 Configure settings in the sidebar, then click **▶ Run Analysis**.")
#     st.stop()

# # ============================================================
# #  CONSTANTS
# # ============================================================

# EPH_PLANET_COLS = [
#     'sun', 'moon', 'mercury', 'venus', 'mars',
#     'jupiter', 'saturn', 'uranus', 'neptune',
#     'pluto', 'true_node', 'mean_node',
# ]

# ASPECTS   = [0, 60, 90, 120, 180]
# ASP_NAMES = {0:'Conj', 60:'Sext', 90:'Sqr', 120:'Trine', 180:'Opp'}
# SIGNS     = ['Aries','Taurus','Gemini','Cancer','Leo','Virgo',
#              'Libra','Scorpio','Sagittarius','Capricorn','Aquarius','Pisces']

# PLANET_WEIGHT = {
#     'sun':       +1.5,
#     'moon':      +1.0,
#     'mercury':   +0.5,
#     'venus':     +2.0,
#     'mars':      -1.5,
#     'jupiter':   +3.0,
#     'saturn':    -2.5,
#     'uranus':    -0.5,
#     'neptune':   +0.5,
#     'pluto':     -1.0,
#     'true_node': +0.5,
#     'mean_node': +0.5,
# }

# ASPECT_MULT = {
#     0:   +1.0,
#     60:  +1.5,
#     90:  -1.8,
#     120: +2.0,
#     180: -1.5,
# }

# PHASE_FACTOR = {'apply': 1.0, 'sep': 0.6}

# # Colours (match existing chart palette)
# BG     = '#0A0A1A';  PANEL  = '#0D0D28';  GOLD   = '#C8A84B'
# TEAL   = '#00D4B4';  WHITE  = '#E8E8F4';  GREY   = '#2A2A4A'
# GREEN  = '#44DD88';  RED    = '#E84040';  ORANGE = '#FF8844'
# PURPLE = '#CC44FF'

# # ============================================================
# #  LOAD EPHEMERIS
# # ============================================================

# with st.spinner("Loading ephemeris …"):
#     try:
#         eph_raw = load_ephemeris(GITHUB_EPH_URL)
#         avail_planets = [p for p in EPH_PLANET_COLS if p in eph_raw.columns]
#         eph = eph_raw[avail_planets].copy()
#         st.success(
#             f"Ephemeris loaded: {len(eph):,} days  "
#             f"({eph.index[0].date()} → {eph.index[-1].date()})"
#         )
#     except Exception as e:
#         st.error(
#             f"Failed to load ephemeris from GitHub.\n\n"
#             f"**Update `GITHUB_EPH_URL`** at the top of this script "
#             f"with your actual raw GitHub URL.\n\nError: {e}"
#         )
#         st.stop()

# # ============================================================
# #  NATAL CHART
# # ============================================================

# USE_NATAL = bool(natal_date_input and natal_date_input.strip())
# natal = {}

# if USE_NATAL:
#     try:
#         natal_ts = pd.Timestamp(natal_date_input)
#         if natal_ts not in eph.index:
#             idx      = eph.index.get_indexer([natal_ts], method='nearest')[0]
#             natal_ts = eph.index[idx]
#         natal_row = eph.loc[natal_ts]
#         natal = {p: float(natal_row[p]) % 360 for p in avail_planets}

#         with st.expander("🌟 Natal Chart Positions", expanded=False):
#             natal_df = pd.DataFrame([
#                 {
#                     'Planet': p.capitalize(),
#                     'Longitude': f"{lon:.3f}°",
#                     'Sign': SIGNS[int(lon // 30)],
#                     'Degree': f"{int(lon % 30):02d}°{int((lon%1)*60):02d}′"
#                 }
#                 for p, lon in natal.items()
#             ])
#             st.dataframe(natal_df, use_container_width=True, hide_index=True)
#     except Exception as e:
#         st.error(f"Invalid natal date: {e}")
#         st.stop()
# else:
#     st.info("No natal date entered — only transit × transit aspects will be scored.")

# # ============================================================
# #  DOWNLOAD PRICE DATA
# # ============================================================

# DATA_END = datetime.date.today().strftime("%Y-%m-%d")

# with st.spinner(f"Downloading {ticker} price data …"):
#     try:
#         raw = yf.download(ticker, start=data_start, end=DATA_END,
#                           progress=False, auto_adjust=False)
#         if isinstance(raw.columns, pd.MultiIndex):
#             raw.columns = raw.columns.get_level_values(0)
#         for col in ['Open','High','Low','Close']:
#             if col in raw.columns:
#                 raw[col] = pd.to_numeric(raw[col], errors='coerce')
#         price_df = raw[['Open','High','Low','Close']].dropna()
#         if len(price_df) == 0:
#             st.error(f"No price data found for '{ticker}'. Check the ticker symbol.")
#             st.stop()
#         dates_px = price_df.index
#         st.success(
#             f"{ticker}: {len(price_df):,} trading days  "
#             f"({dates_px[0].date()} → {dates_px[-1].date()})  |  "
#             f"Last close: {price_df['Close'].iloc[-1]:.4f}"
#         )
#     except Exception as e:
#         st.error(f"Price download failed: {e}")
#         st.stop()

# # ============================================================
# #  CORE HELPERS
# # ============================================================

# def angular_diff(lon_a, lon_b):
#     d = (lon_a - lon_b) % 360
#     return np.where(d > 180, d - 360, d)

# def orb_factor(abs_gap, orb_max):
#     return np.clip(1.0 - abs_gap / orb_max, 0.0, 1.0)

# def aspect_score_single(pw_a, pw_b, asp, orb_f, phase):
#     magnitude = (abs(pw_a) + abs(pw_b)) / 2.0
#     if asp == 0:
#         net_polarity = pw_a + pw_b
#         direction    = float(np.sign(net_polarity)) if net_polarity != 0 else 1.0
#         asp_strength = abs(ASPECT_MULT[0])
#     else:
#         direction    = float(np.sign(ASPECT_MULT[asp]))
#         asp_strength = abs(ASPECT_MULT[asp])
#     return direction * magnitude * asp_strength * orb_f * PHASE_FACTOR[phase]

# def compute_natal_score(date_index):
#     eph_a  = eph.reindex(date_index, method='ffill')
#     n      = len(date_index)
#     scores = np.zeros(n)
#     detail = []
#     for tp in avail_planets:
#         if tp not in eph_a.columns: continue
#         t_lons  = eph_a[tp].values.astype(float) % 360
#         motion  = np.gradient(np.unwrap(t_lons, period=360))
#         pw_t    = PLANET_WEIGHT.get(tp, 0.0)
#         for np_ in avail_planets:
#             n_lon = natal[np_]
#             pw_n  = PLANET_WEIGHT.get(np_, 0.0)
#             for asp in ASPECTS:
#                 target  = (n_lon + asp) % 360
#                 gap     = angular_diff(t_lons, target)
#                 abs_gap = np.abs(gap)
#                 applying = ((motion > 0) & (gap < 0)) | ((motion < 0) & (gap > 0))
#                 mask_a = applying & (abs_gap <= orb_apply)
#                 mask_s = (~applying) & (abs_gap <= orb_sep)
#                 for i in np.where(mask_a)[0]:
#                     of = float(orb_factor(abs_gap[i], orb_apply))
#                     sc = aspect_score_single(pw_t, pw_n, asp, of, 'apply')
#                     scores[i] += sc
#                     detail.append({'date': date_index[i], 'transit': tp,
#                                    'natal': np_, 'aspect': ASP_NAMES[asp],
#                                    'phase': 'Applying',
#                                    'orb': round(float(abs_gap[i]), 3),
#                                    'score': round(sc, 4)})
#                 for i in np.where(mask_s)[0]:
#                     of = float(orb_factor(abs_gap[i], orb_sep))
#                     sc = aspect_score_single(pw_t, pw_n, asp, of, 'sep')
#                     scores[i] += sc
#                     detail.append({'date': date_index[i], 'transit': tp,
#                                    'natal': np_, 'aspect': ASP_NAMES[asp],
#                                    'phase': 'Separating',
#                                    'orb': round(float(abs_gap[i]), 3),
#                                    'score': round(sc, 4)})
#     return pd.Series(scores, index=date_index), detail

# def compute_transit_score(date_index):
#     eph_a  = eph.reindex(date_index, method='ffill')
#     n      = len(date_index)
#     scores = np.zeros(n)
#     detail = []
#     pairs  = list(itertools.combinations(avail_planets, 2))
#     for (pA, pB) in pairs:
#         if pA not in eph_a.columns or pB not in eph_a.columns: continue
#         lon_A   = eph_a[pA].values.astype(float) % 360
#         lon_B   = eph_a[pB].values.astype(float) % 360
#         motion  = np.gradient(np.unwrap(lon_A, period=360))
#         pw_A    = PLANET_WEIGHT.get(pA, 0.0)
#         pw_B    = PLANET_WEIGHT.get(pB, 0.0)
#         for asp in ASPECTS:
#             target  = (lon_B + asp) % 360
#             gap     = angular_diff(lon_A, target)
#             abs_gap = np.abs(gap)
#             applying = ((motion > 0) & (gap < 0)) | ((motion < 0) & (gap > 0))
#             mask_a = applying & (abs_gap <= orb_apply)
#             mask_s = (~applying) & (abs_gap <= orb_sep)
#             for i in np.where(mask_a)[0]:
#                 of = float(orb_factor(abs_gap[i], orb_apply))
#                 sc = aspect_score_single(pw_A, pw_B, asp, of, 'apply')
#                 scores[i] += sc
#                 detail.append({'date': date_index[i], 'planet_a': pA,
#                                'planet_b': pB, 'aspect': ASP_NAMES[asp],
#                                'phase': 'Applying',
#                                'orb': round(float(abs_gap[i]), 3),
#                                'score': round(sc, 4)})
#             for i in np.where(mask_s)[0]:
#                 of = float(orb_factor(abs_gap[i], orb_sep))
#                 sc = aspect_score_single(pw_A, pw_B, asp, of, 'sep')
#                 scores[i] += sc
#                 detail.append({'date': date_index[i], 'planet_a': pA,
#                                'planet_b': pB, 'aspect': ASP_NAMES[asp],
#                                'phase': 'Separating',
#                                'orb': round(float(abs_gap[i]), 3),
#                                'score': round(sc, 4)})
#     return pd.Series(scores, index=date_index), detail

# # ============================================================
# #  BUILD DATE INDEX & COMPUTE SCORES
# # ============================================================

# try:
#     chart_end_ts = pd.Timestamp(chart_end_input)
# except Exception:
#     chart_end_ts = dates_px[-1] + pd.Timedelta(days=365)

# table_future_end = dates_px[-1] + pd.Timedelta(days=table_days + 7)
# score_end        = max(chart_end_ts, table_future_end)

# future_score_dates = pd.date_range(
#     start = dates_px[-1] + pd.Timedelta(days=1),
#     end   = score_end, freq='D')
# full_index = dates_px.append(future_score_dates)

# if chart_end_ts > dates_px[-1]:
#     chart_future_dates = pd.date_range(
#         start = dates_px[-1] + pd.Timedelta(days=1),
#         end   = chart_end_ts, freq='B')
# else:
#     chart_future_dates = pd.DatetimeIndex([])

# x_end = chart_end_ts if chart_end_ts > dates_px[-1] else dates_px[-1]

# with st.spinner("Computing natal aspect scores …"):
#     if USE_NATAL:
#         natal_scores_full, natal_detail_full = compute_natal_score(full_index)
#     else:
#         natal_scores_full   = pd.Series(np.zeros(len(full_index)), index=full_index)
#         natal_detail_full   = []

# with st.spinner("Computing transit aspect scores …"):
#     transit_scores_full, transit_detail_full = compute_transit_score(full_index)

# # Slice to price dates
# natal_scores_px   = natal_scores_full.reindex(dates_px).fillna(0)
# transit_scores_px = transit_scores_full.reindex(dates_px).fillna(0)

# # Future extension
# if len(chart_future_dates):
#     natal_scores_fut   = natal_scores_full.reindex(
#         chart_future_dates, method='ffill').fillna(0)
#     transit_scores_fut = transit_scores_full.reindex(
#         chart_future_dates, method='ffill').fillna(0)
# else:
#     natal_scores_fut   = pd.Series(dtype=float)
#     transit_scores_fut = pd.Series(dtype=float)

# # Summary metrics
# col1, col2, col3, col4 = st.columns(4)
# col1.metric("Natal score today",
#             f"{natal_scores_px.iloc[-1]:.2f}",
#             delta=f"{natal_scores_px.iloc[-1]-natal_scores_px.iloc[-2]:.2f}")
# col2.metric("Transit score today",
#             f"{transit_scores_px.iloc[-1]:.2f}",
#             delta=f"{transit_scores_px.iloc[-1]-transit_scores_px.iloc[-2]:.2f}")
# col3.metric("Last close", f"{price_df['Close'].iloc[-1]:.4f}")
# col4.metric("Forecast to", str(chart_end_ts.date()))

# # ============================================================
# #  PLOT HELPERS
# # ============================================================

# def plot_candlestick(ax, df, width=0.6):
#     for idx, row in df.iterrows():
#         o, h, l, c = row['Open'], row['High'], row['Low'], row['Close']
#         color = GREEN if c >= o else RED
#         ax.bar(idx, abs(c - o), bottom=min(o, c),
#                width=width, color=color, alpha=0.85, linewidth=0, zorder=3)
#         ax.plot([idx, idx], [l, h], color=color, lw=0.8, alpha=0.7, zorder=2)

# def style_ax(ax):
#     ax.set_facecolor(PANEL)
#     for sp in ax.spines.values(): sp.set_color(GREY)
#     ax.tick_params(colors=WHITE, labelsize=8)

# def draw_shading(ax, dates, scores, alpha_cap, alpha_denom):
#     for i in range(len(dates)-1):
#         sc = scores.iloc[i]
#         if sc == 0: continue
#         color = GREEN if sc > 0 else RED
#         ax.axvspan(dates[i], dates[i+1],
#                    alpha=min(alpha_cap, abs(sc)/alpha_denom),
#                    color=color, zorder=1)

# def draw_future_shading(ax, dates, scores, alpha_cap, alpha_denom):
#     if not len(dates): return
#     for i in range(len(dates)-1):
#         sc = scores.iloc[i]
#         if sc == 0: continue
#         color = GREEN if sc > 0 else RED
#         ax.axvspan(dates[i], dates[i+1],
#                    alpha=min(alpha_cap, abs(sc)/alpha_denom),
#                    color=color, zorder=1)

# def draw_today(ax, y_ref, is_price=True):
#     ax.axvline(dates_px[-1], color=GOLD, lw=1.5, ls='--', alpha=0.8, zorder=5)
#     if is_price:
#         ax.text(dates_px[-1], y_ref, ' Today',
#                 color=GOLD, fontsize=7.5, va='top', ha='left', fontweight='bold')

# def format_xaxis(ax):
#     ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
#     ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
#     plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=7)
#     ax.set_xlim(dates_px[0], x_end + pd.Timedelta(days=2))

# def smooth(series, window=7):
#     return series.rolling(window=window, center=True, min_periods=1).mean()

# legend_els = [
#     mpatches.Patch(color=GREEN, alpha=0.8, label='Bullish candle / +score'),
#     mpatches.Patch(color=RED,   alpha=0.8, label='Bearish candle / −score'),
#     Line2D([0],[0], color=GOLD, lw=1.5, label='Score 7d smoothed'),
#     Line2D([0],[0], color=GOLD, lw=1.5, ls='--', label='Today'),
# ]

# # ============================================================
# #  CHART 1: CANDLESTICK + NATAL SCORE
# # ============================================================

# st.markdown("---")
# st.markdown("## Chart 1 — Natal Aspect Score")
# st.caption("Transit planets aspecting the natal chart positions.")

# fig1, (ax_p1, ax_s1) = plt.subplots(
#     2, 1, figsize=(18, 9), facecolor=BG,
#     gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.08}, sharex=True)

# style_ax(ax_p1)
# ax_p1.set_ylabel('Price', color=WHITE, fontsize=10)
# ax_p1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
# plot_candlestick(ax_p1, price_df)
# draw_shading(ax_p1, dates_px, natal_scores_px, 0.15, 20)
# draw_future_shading(ax_p1, chart_future_dates, natal_scores_fut, 0.15, 20)
# draw_today(ax_p1, price_df['High'].max(), is_price=True)
# ax_p1.legend(handles=legend_els, fontsize=8, facecolor='#1A1A38',
#              labelcolor=WHITE, loc='upper left')

# style_ax(ax_s1)
# ax_s1.set_ylabel('Natal Score', color=WHITE, fontsize=9)
# ax_s1.axhline(0, color=GREY, lw=1.0, zorder=2)

# n_vals = natal_scores_px.values
# ax_s1.bar(dates_px, n_vals,
#           color=[GREEN if v >= 0 else RED for v in n_vals],
#           alpha=0.75, width=1.0, zorder=3)
# if len(natal_scores_fut):
#     nf_vals = natal_scores_fut.values
#     ax_s1.bar(natal_scores_fut.index, nf_vals,
#               color=[GREEN if v >= 0 else RED for v in nf_vals],
#               alpha=0.75, width=1.0, zorder=3)

# combined1 = pd.concat([natal_scores_px, natal_scores_fut])
# sc1_smooth = smooth(combined1)
# ax_s1.plot(dates_px, sc1_smooth.reindex(dates_px).values,
#            color=GOLD, lw=1.8, zorder=4, label='7-day smoothed')
# if len(natal_scores_fut):
#     ax_s1.plot(natal_scores_fut.index,
#                sc1_smooth.reindex(natal_scores_fut.index).values,
#                color=GOLD, lw=1.8, ls='--', zorder=4, alpha=0.85)
# draw_today(ax_s1, 0, is_price=False)
# ax_s1.legend(fontsize=7, facecolor='#1A1A38', labelcolor=WHITE, loc='upper left')
# format_xaxis(ax_s1)

# natal_label = f"Natal: {natal_date_input}  |  " if USE_NATAL else "No natal chart  |  "
# fig1.suptitle(
#     f"{ticker}  |  Candlestick + Natal Aspect Score\n"
#     f"{natal_label}Apply≤{orb_apply}°  Sep≤{orb_sep}°  |  "
#     f"Green=Bullish  Red=Bearish  |  Gold dashed = Today",
#     color=GOLD, fontsize=11, fontweight='bold')
# fig1.tight_layout()
# st.pyplot(fig1, use_container_width=True)
# plt.close(fig1)

# # ============================================================
# #  CHART 2: CANDLESTICK + TRANSIT SCORE
# # ============================================================

# st.markdown("---")
# st.markdown("## Chart 2 — Transit × Transit Aspect Score")
# st.caption("All transit planet pairs aspecting each other. No natal chart used.")

# fig2, (ax_p2, ax_s2) = plt.subplots(
#     2, 1, figsize=(18, 9), facecolor=BG,
#     gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.08}, sharex=True)

# style_ax(ax_p2)
# ax_p2.set_ylabel('Price', color=WHITE, fontsize=10)
# ax_p2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
# plot_candlestick(ax_p2, price_df)
# draw_shading(ax_p2, dates_px, transit_scores_px, 0.15, 30)
# draw_future_shading(ax_p2, chart_future_dates, transit_scores_fut, 0.15, 30)
# draw_today(ax_p2, price_df['High'].max(), is_price=True)
# ax_p2.legend(handles=legend_els, fontsize=8, facecolor='#1A1A38',
#              labelcolor=WHITE, loc='upper left')

# style_ax(ax_s2)
# ax_s2.set_ylabel('Transit Score', color=WHITE, fontsize=9)
# ax_s2.axhline(0, color=GREY, lw=1.0, zorder=2)

# t_vals = transit_scores_px.values
# ax_s2.bar(dates_px, t_vals,
#           color=[GREEN if v >= 0 else RED for v in t_vals],
#           alpha=0.75, width=1.0, zorder=3)
# if len(transit_scores_fut):
#     tf_vals = transit_scores_fut.values
#     ax_s2.bar(transit_scores_fut.index, tf_vals,
#               color=[GREEN if v >= 0 else RED for v in tf_vals],
#               alpha=0.75, width=1.0, zorder=3)

# combined2 = pd.concat([transit_scores_px, transit_scores_fut])
# sc2_smooth = smooth(combined2)
# ax_s2.plot(dates_px, sc2_smooth.reindex(dates_px).values,
#            color=GOLD, lw=1.8, zorder=4, label='7-day smoothed')
# if len(transit_scores_fut):
#     ax_s2.plot(transit_scores_fut.index,
#                sc2_smooth.reindex(transit_scores_fut.index).values,
#                color=GOLD, lw=1.8, ls='--', zorder=4, alpha=0.85)
# draw_today(ax_s2, 0, is_price=False)
# ax_s2.legend(fontsize=7, facecolor='#1A1A38', labelcolor=WHITE, loc='upper left')
# format_xaxis(ax_s2)

# fig2.suptitle(
#     f"{ticker}  |  Candlestick + Transit × Transit Aspect Score\n"
#     f"Apply≤{orb_apply}°  Sep≤{orb_sep}°  |  "
#     f"Green=Bullish  Red=Bearish  |  Gold dashed = Today",
#     color=GOLD, fontsize=11, fontweight='bold')
# fig2.tight_layout()
# st.pyplot(fig2, use_container_width=True)
# plt.close(fig2)

# # ============================================================
# #  CHART 3: CANDLESTICK + CUMULATIVE SCORE
# # ============================================================

# st.markdown("---")
# st.markdown("## Chart 3 — Cumulative Aspect Score")
# st.caption(
#     "Running total of all aspect scores since data start. "
#     "Rising = improving planetary conditions. Falling = deteriorating.")

# natal_hist    = natal_scores_px.copy()
# transit_hist  = transit_scores_px.copy()
# combined_hist = natal_hist.add(transit_hist, fill_value=0)

# if len(natal_scores_fut) and len(transit_scores_fut):
#     combined_fut = natal_scores_fut.add(
#         transit_scores_fut.reindex(natal_scores_fut.index, fill_value=0),
#         fill_value=0)
# elif len(natal_scores_fut):
#     combined_fut = natal_scores_fut.copy()
# elif len(transit_scores_fut):
#     combined_fut = transit_scores_fut.copy()
# else:
#     combined_fut = pd.Series(dtype=float)

# natal_cum_hist   = natal_hist.cumsum()
# transit_cum_hist = transit_hist.cumsum()
# combined_cum_hist = combined_hist.cumsum()

# natal_cum_fut    = pd.Series(dtype=float)
# transit_cum_fut  = pd.Series(dtype=float)
# combined_cum_fut = pd.Series(dtype=float)

# if len(natal_scores_fut):
#     nat_full      = pd.concat([natal_hist, natal_scores_fut])
#     natal_cum_fut = nat_full.cumsum().reindex(natal_scores_fut.index)

# if len(transit_scores_fut):
#     tr_full        = pd.concat([transit_hist, transit_scores_fut])
#     transit_cum_fut = tr_full.cumsum().reindex(transit_scores_fut.index)

# if len(combined_fut):
#     comb_full       = pd.concat([combined_hist, combined_fut])
#     combined_cum_fut = comb_full.cumsum().reindex(combined_fut.index)

# fig3, (ax_p3, ax_s3) = plt.subplots(
#     2, 1, figsize=(18, 9), facecolor=BG,
#     gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.08}, sharex=True)

# style_ax(ax_p3)
# ax_p3.set_ylabel('Price', color=WHITE, fontsize=10)
# ax_p3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
# plot_candlestick(ax_p3, price_df)
# draw_shading(ax_p3, dates_px, combined_hist, 0.12, 40)
# draw_future_shading(ax_p3, chart_future_dates, combined_fut, 0.12, 40)
# draw_today(ax_p3, price_df['High'].max(), is_price=True)

# leg3 = [
#     mpatches.Patch(color=GREEN, alpha=0.8, label='Bullish candle / +score'),
#     mpatches.Patch(color=RED,   alpha=0.8, label='Bearish candle / −score'),
#     Line2D([0],[0], color=TEAL,   lw=2.0,        label='Combined cumulative'),
#     Line2D([0],[0], color=ORANGE, lw=1.4, ls='--', label='Natal cumulative'),
#     Line2D([0],[0], color=PURPLE, lw=1.4, ls='--', label='Transit cumulative'),
#     Line2D([0],[0], color=GOLD,   lw=1.5, ls='--', label='Today'),
# ]
# ax_p3.legend(handles=leg3, fontsize=8, facecolor='#1A1A38',
#              labelcolor=WHITE, loc='upper left')

# style_ax(ax_s3)
# ax_s3.set_ylabel('Cumulative Score', color=WHITE, fontsize=9)
# ax_s3.axhline(0, color=GREY, lw=1.0, zorder=2)

# ax_s3.plot(dates_px, natal_cum_hist.values,
#            color=ORANGE, lw=1.3, ls='--', alpha=0.8, zorder=3, label='Natal')
# ax_s3.plot(dates_px, transit_cum_hist.values,
#            color=PURPLE, lw=1.3, ls='--', alpha=0.8, zorder=3, label='Transit')
# ax_s3.plot(dates_px, combined_cum_hist.values,
#            color=TEAL, lw=2.2, zorder=4, label='Combined')
# ax_s3.fill_between(dates_px, combined_cum_hist.values, 0,
#                    where=(combined_cum_hist.values >= 0),
#                    color=GREEN, alpha=0.15, zorder=1)
# ax_s3.fill_between(dates_px, combined_cum_hist.values, 0,
#                    where=(combined_cum_hist.values < 0),
#                    color=RED, alpha=0.15, zorder=1)

# if len(natal_cum_fut):
#     conn = pd.Series([natal_cum_hist.iloc[-1], natal_cum_fut.iloc[0]],
#                      index=[dates_px[-1], natal_cum_fut.index[0]])
#     ax_s3.plot(conn.index, conn.values, color=ORANGE, lw=1.3, ls='--', alpha=0.5)
#     ax_s3.plot(natal_cum_fut.index, natal_cum_fut.values,
#                color=ORANGE, lw=1.3, ls=':', alpha=0.7, zorder=3)

# if len(transit_cum_fut):
#     conn = pd.Series([transit_cum_hist.iloc[-1], transit_cum_fut.iloc[0]],
#                      index=[dates_px[-1], transit_cum_fut.index[0]])
#     ax_s3.plot(conn.index, conn.values, color=PURPLE, lw=1.3, ls='--', alpha=0.5)
#     ax_s3.plot(transit_cum_fut.index, transit_cum_fut.values,
#                color=PURPLE, lw=1.3, ls=':', alpha=0.7, zorder=3)

# if len(combined_cum_fut):
#     conn = pd.Series([combined_cum_hist.iloc[-1], combined_cum_fut.iloc[0]],
#                      index=[dates_px[-1], combined_cum_fut.index[0]])
#     ax_s3.plot(conn.index, conn.values, color=TEAL, lw=2.2, alpha=0.6)
#     ax_s3.plot(combined_cum_fut.index, combined_cum_fut.values,
#                color=TEAL, lw=2.2, ls='--', alpha=0.7, zorder=4)
#     last_val = combined_cum_hist.iloc[-1]
#     ax_s3.fill_between(combined_cum_fut.index,
#                         combined_cum_fut.values, last_val,
#                         where=(combined_cum_fut.values >= last_val),
#                         color=GREEN, alpha=0.10, zorder=1)
#     ax_s3.fill_between(combined_cum_fut.index,
#                         combined_cum_fut.values, last_val,
#                         where=(combined_cum_fut.values < last_val),
#                         color=RED, alpha=0.10, zorder=1)

# draw_today(ax_s3, 0, is_price=False)
# ax_s3.legend(fontsize=7, facecolor='#1A1A38', labelcolor=WHITE, loc='upper left')
# format_xaxis(ax_s3)

# fig3.suptitle(
#     f"{ticker}  |  Candlestick + Cumulative Aspect Score\n"
#     f"Teal=Combined  Orange=Natal  Purple=Transit  |  "
#     f"Solid=History  Dashed=Forecast  |  Gold dashed = Today",
#     color=GOLD, fontsize=11, fontweight='bold')
# fig3.tight_layout()
# st.pyplot(fig3, use_container_width=True)
# plt.close(fig3)

# # ============================================================
# #  ASPECT TABLES
# # ============================================================

# table_start = dates_px[-1] - pd.Timedelta(days=7)
# table_end   = dates_px[-1] + pd.Timedelta(days=table_days)

# def filter_window(detail_list):
#     rows = []
#     for r in detail_list:
#         d = pd.Timestamp(r['date'])
#         if table_start <= d <= table_end:
#             r2 = r.copy()
#             r2['date']   = d.date()
#             r2['period'] = 'Past' if d <= dates_px[-1] else 'Future'
#             rows.append(r2)
#     if not rows:
#         return pd.DataFrame()
#     return (pd.DataFrame(rows)
#             .sort_values(['date','score'], ascending=[True, False])
#             .reset_index(drop=True))

# natal_win   = filter_window(natal_detail_full)
# transit_win = filter_window(transit_detail_full)

# def score_color(val):
#     """Colour score cells green/red for Streamlit dataframe."""
#     if isinstance(val, float):
#         if val > 0:  return 'color: #44DD88'
#         if val < 0:  return 'color: #E84040'
#     return ''

# st.markdown("---")
# st.markdown("## 📋 Aspect Tables")
# st.caption(
#     f"Last 7 calendar days + next {table_days} days. "
#     "Green score = bullish, Red = bearish.")

# tab1, tab2 = st.tabs(["🌟 Natal Aspects", "🔄 Transit × Transit Aspects"])

# with tab1:
#     if not USE_NATAL:
#         st.info("No natal date entered — natal aspects not computed.")
#     elif natal_win.empty:
#         st.info("No natal aspects active in the selected window.")
#     else:
#         # Rename for display
#         display_n = natal_win.rename(columns={
#             'date':'Date','transit':'Transit','natal':'Natal Planet',
#             'aspect':'Aspect','phase':'Phase','orb':'Orb°','score':'Score',
#             'period':'Period'})

#         col_order = ['Date','Period','Transit','Natal Planet','Aspect','Phase','Orb°','Score']
#         display_n = display_n[[c for c in col_order if c in display_n.columns]]
#         display_n['Transit']      = display_n['Transit'].str.capitalize()
#         display_n['Natal Planet'] = display_n['Natal Planet'].str.capitalize()

#         st.markdown("### Past 7 days")
#         past_n = display_n[display_n['Period']=='Past']
#         if past_n.empty:
#             st.info("No natal aspects in the past 7 days.")
#         else:
#             st.dataframe(
#                 past_n.drop(columns='Period').style.applymap(
#                     score_color, subset=['Score']),
#                 use_container_width=True, hide_index=True)

#         st.markdown(f"### Next {table_days} days")
#         fut_n = display_n[display_n['Period']=='Future']
#         if fut_n.empty:
#             st.info("No natal aspects in the next {table_days} days.")
#         else:
#             st.dataframe(
#                 fut_n.drop(columns='Period').style.applymap(
#                     score_color, subset=['Score']),
#                 use_container_width=True, hide_index=True)

#         # Daily net summary
#         st.markdown("### Daily Net Natal Score")
#         daily_n = (natal_win.groupby(['date','period'])
#                    .agg(Aspects=('score','count'), Net_Score=('score','sum'))
#                    .reset_index().sort_values('date'))
#         daily_n['Bias'] = daily_n['Net_Score'].apply(
#             lambda x: '▲ Bullish' if x > 0 else '▼ Bearish')
#         daily_n['date'] = daily_n['date'].astype(str)
#         daily_n.columns = ['Date','Period','# Aspects','Net Score','Bias']
#         st.dataframe(
#             daily_n.style.applymap(score_color, subset=['Net Score']),
#             use_container_width=True, hide_index=True)

# with tab2:
#     if transit_win.empty:
#         st.info("No transit aspects active in the selected window.")
#     else:
#         display_t = transit_win.rename(columns={
#             'date':'Date','planet_a':'Planet A','planet_b':'Planet B',
#             'aspect':'Aspect','phase':'Phase','orb':'Orb°','score':'Score',
#             'period':'Period'})
#         col_order_t = ['Date','Period','Planet A','Planet B','Aspect','Phase','Orb°','Score']
#         display_t = display_t[[c for c in col_order_t if c in display_t.columns]]
#         display_t['Planet A'] = display_t['Planet A'].str.capitalize()
#         display_t['Planet B'] = display_t['Planet B'].str.capitalize()

#         st.markdown("### Past 7 days")
#         past_t = display_t[display_t['Period']=='Past']
#         if past_t.empty:
#             st.info("No transit aspects in the past 7 days.")
#         else:
#             st.dataframe(
#                 past_t.drop(columns='Period').style.applymap(
#                     score_color, subset=['Score']),
#                 use_container_width=True, hide_index=True)

#         st.markdown(f"### Next {table_days} days")
#         fut_t = display_t[display_t['Period']=='Future']
#         if fut_t.empty:
#             st.info(f"No transit aspects in the next {table_days} days.")
#         else:
#             st.dataframe(
#                 fut_t.drop(columns='Period').style.applymap(
#                     score_color, subset=['Score']),
#                 use_container_width=True, hide_index=True)

#         st.markdown("### Daily Net Transit Score")
#         daily_t = (transit_win.groupby(['date','period'])
#                    .agg(Aspects=('score','count'), Net_Score=('score','sum'))
#                    .reset_index().sort_values('date'))
#         daily_t['Bias'] = daily_t['Net_Score'].apply(
#             lambda x: '▲ Bullish' if x > 0 else '▼ Bearish')
#         daily_t['date'] = daily_t['date'].astype(str)
#         daily_t.columns = ['Date','Period','# Aspects','Net Score','Bias']
#         st.dataframe(
#             daily_t.style.applymap(score_color, subset=['Net Score']),
#             use_container_width=True, hide_index=True)

# # ============================================================
# #  SCORING LEGEND
# # ============================================================

# st.markdown("---")
# with st.expander("📖 Scoring Methodology", expanded=False):
#     st.markdown("""
# **Score = direction × magnitude × aspect_strength × orb_proximity × phase_factor**

# | Component | Rule |
# |---|---|
# | **direction** | Sign of aspect type: Trine/Sext = +1, Sq/Opp = −1, Conj = sign of planet weight sum |
# | **magnitude** | (\\|Planet A weight\\| + \\|Planet B weight\\|) / 2 |
# | **aspect_strength** | \\|aspect multiplier\\| |
# | **orb_proximity** | Linear fade: 1.0 at exact → 0.0 at orb edge |
# | **phase_factor** | Applying = 1.0 &nbsp;&nbsp; Separating = 0.6 |

# **Planet Weights:**

# | Bullish | Weight | Bearish | Weight |
# |---|---|---|---|
# | Jupiter | +3.0 | Saturn | −2.5 |
# | Venus | +2.0 | Mars | −1.5 |
# | Sun | +1.5 | Pluto | −1.0 |
# | Moon | +1.0 | Uranus | −0.5 |
# | Neptune | +0.5 | | |
# | Mercury | +0.5 | | |
# | North Node | +0.5 | | |

# **Aspect Multipliers:** Trine +2.0 · Sextile +1.5 · Conj ±1.0 · Opposition −1.5 · Square −1.8

# **Interpretation:** Score > +5 = strongly bullish · Score < −5 = strongly bearish · Score ≈ 0 = neutral
#     """)
