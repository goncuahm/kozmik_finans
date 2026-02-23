# ============================================================
# PLANETARY REGRESSION — STREAMLIT VERSION
# ============================================================

import streamlit as st
import warnings, datetime, itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
import statsmodels.api as sm
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")
st.set_page_config(layout="wide")

# ============================================================
# CONSTANTS (UNCHANGED)
# ============================================================

EPH_PLANET_COLS = [
    'sun','moon','mercury','venus','mars',
    'jupiter','saturn','uranus','neptune',
    'pluto','true_node','mean_node'
]

ASPECTS   = [0,60,90,120,180]
ASP_NAMES = {0:'Conj',60:'Sext',90:'Sqr',120:'Trin',180:'Opp'}
SIGNS     = ['Aries','Taurus','Gemini','Cancer','Leo','Virgo',
             'Libra','Scorpio','Sagittarius','Capricorn',
             'Aquarius','Pisces']

ORB_APPLY  = 3
ORB_SEP    = 1

ANN_LAYERS   = (128,64,32)
ANN_ALPHA    = 0.01
ANN_MAXITER  = 1000
ANN_VALFRAC  = 0.15
ANN_PATIENCE = 50

DAYS_SHORT = 10
DAYS_LONG  = 365

# 🔴 CHANGE THIS PATH
EPHE_PATH = "planet_degrees.csv"


# ============================================================
# USER INPUT
# ============================================================

st.title("🌌 Planetary Regression Forecast Engine")

col1, col2, col3 = st.columns(3)

with col1:
    ticker = st.text_input("Ticker", value="XU100.IS")

with col2:
    data_start = st.date_input("Price Start Date",
                               datetime.date(2005,1,1))

with col3:
    use_birth = st.checkbox("Birth Date Available?")

natal_date = None
if use_birth:
    natal_date = st.date_input("Birth Date")

run_button = st.button("Run Forecast")

if not run_button:
    st.stop()

# ============================================================
# LOAD EPHEMERIS
# ============================================================

@st.cache_data
def load_ephemeris(path):
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    return df

eph_raw = load_ephemeris(EPHE_PATH)
avail_planets = [p for p in EPH_PLANET_COLS if p in eph_raw.columns]
eph = eph_raw[avail_planets].copy()

USE_NATAL = bool(use_birth)

# ============================================================
# BUILD NATAL
# ============================================================

natal = {}
if USE_NATAL:
    natal_ts = pd.Timestamp(natal_date)
    if natal_ts not in eph.index:
        idx = eph.index.get_indexer([natal_ts], method='nearest')[0]
        natal_ts = eph.index[idx]

    natal_row = eph.loc[natal_ts]
    natal = {p: float(natal_row[p]) % 360 for p in avail_planets}


# ============================================================
# DOWNLOAD PRICE
# ============================================================

with st.spinner("Downloading price data..."):
    raw = yf.download(ticker,
                      start=str(data_start),
                      progress=False)

raw['Close'] = pd.to_numeric(raw['Close'], errors='coerce')
price_df = raw[['Close']].dropna()
price_df['log_price'] = np.log(price_df['Close'])

dates_all = price_df.index
y_all     = price_df['log_price'].values
n_total   = len(dates_all)

# ============================================================
# HELPER FUNCTIONS (UNCHANGED LOGIC)
# ============================================================

def angular_diff(a,b):
    d = (a-b)%360
    return np.where(d>180,d-360,d)

def add_const_trend(X,trend):
    return np.hstack([np.ones((len(trend),1)),
                      trend.reshape(-1,1),X])

def build_natal_transit_aspects(date_index):
    eph_a = eph.reindex(date_index, method='ffill')
    feat = {}
    for tp in avail_planets:
        t_lons = eph_a[tp].values%360
        motion = np.gradient(np.unwrap(t_lons,period=360))
        for np_ in natal:
            n_lon = natal[np_]
            for asp in ASPECTS:
                target = (n_lon+asp)%360
                gap = angular_diff(t_lons,target)
                abs_gap = np.abs(gap)

                applying = ((motion>0)&(gap<0)) | ((motion<0)&(gap>0))
                separating = ~applying

                col_a = f"{tp}__{np_}__{asp}__apply"
                col_s = f"{tp}__{np_}__{asp}__sep"

                feat[col_a]=(applying&(abs_gap<=ORB_APPLY)).astype(float)
                feat[col_s]=(separating&(abs_gap<=ORB_SEP)).astype(float)

    df=pd.DataFrame(feat,index=date_index)
    return df.loc[:,(df>0).any(axis=0)]

def build_transit_transit(date_index):
    eph_a=eph.reindex(date_index,method='ffill')
    feat={}
    pairs=list(itertools.combinations(avail_planets,2))
    for pA,pB in pairs:
        lonA=eph_a[pA].values%360
        lonB=eph_a[pB].values%360
        motion=np.gradient(np.unwrap(lonA,period=360))
        for asp in ASPECTS:
            target=(lonB+asp)%360
            gap=angular_diff(lonA,target)
            abs_gap=np.abs(gap)

            applying=((motion>0)&(gap<0))|((motion<0)&(gap>0))
            separating=~applying

            col_a=f"{pA}_x_{pB}__{asp}__apply"
            col_s=f"{pA}_x_{pB}__{asp}__sep"
            feat[col_a]=(applying&(abs_gap<=ORB_APPLY)).astype(float)
            feat[col_s]=(separating&(abs_gap<=ORB_SEP)).astype(float)

    df=pd.DataFrame(feat,index=date_index)
    return df.loc[:,(df>0).any(axis=0)]

def build_features(date_index):
    if USE_NATAL:
        return build_natal_transit_aspects(date_index)
    else:
        return build_transit_transit(date_index)

# ============================================================
# BUILD FEATURES
# ============================================================

feat_all = build_features(dates_all)
feat_names = feat_all.columns.tolist()

trend_all = np.arange(n_total)/n_total
X_all = add_const_trend(feat_all.values,trend_all)

scaler=StandardScaler()
X_sc=scaler.fit_transform(X_all[:,2:])

# ============================================================
# OLS
# ============================================================

ols=sm.OLS(y_all,X_all).fit()
y_ols=np.asarray(ols.fittedvalues)
r2_ols=r2_score(y_all,y_ols)

# ============================================================
# ANN (DETREND)
# ============================================================

ols_trend=sm.OLS(y_all,X_all[:,:2]).fit()
trend_fit=ols_trend.predict(X_all[:,:2])
resid_all=y_all-trend_fit

ann=MLPRegressor(hidden_layer_sizes=ANN_LAYERS,
                 alpha=ANN_ALPHA,
                 max_iter=ANN_MAXITER,
                 early_stopping=True,
                 validation_fraction=ANN_VALFRAC,
                 n_iter_no_change=ANN_PATIENCE,
                 random_state=42)

ann.fit(X_sc,resid_all)
resid_pred=ann.predict(X_sc)

y_ann=trend_fit+resid_pred
r2_ann=r2_score(y_all,y_ann)

# ============================================================
# FORECAST
# ============================================================

fore_start=dates_all[-1]+pd.Timedelta(days=1)
fut_dates=pd.date_range(fore_start,periods=DAYS_LONG,freq='B')
n_fut=len(fut_dates)

feat_fut=build_features(fut_dates)
X_fut_asp=feat_fut.reindex(columns=feat_names,fill_value=0.0).values
trend_fut=(n_total+np.arange(n_fut))/n_total
X_fut=add_const_trend(X_fut_asp,trend_fut)

y_fore_ols=ols.predict(X_fut)

trend_fore=ols_trend.predict(X_fut[:,:2])
X_fut_sc=scaler.transform(X_fut[:,2:])
resid_fore=ann.predict(X_fut_sc)
y_fore_ann=trend_fore+resid_fore

last_log=y_all[-1]

y_fore_ols=last_log+(y_fore_ols-y_fore_ols[0])
y_fore_ann=last_log+(y_fore_ann-y_fore_ann[0])

actual_px=np.exp(y_all)
ann_px=np.exp(y_ann)
ols_px=np.exp(y_ols)

fore_ann_px=np.exp(y_fore_ann)
fore_ols_px=np.exp(y_fore_ols)

# ============================================================
# PLOTS
# ============================================================

st.subheader("Short-Term Forecast (10 Days)")
fig1,ax1=plt.subplots(figsize=(12,5))
ax1.plot(dates_all[-120:],actual_px[-120:],label="Actual")
ax1.plot(dates_all[-120:],ann_px[-120:],label="ANN Fit")
ax1.plot(dates_all[-120:],ols_px[-120:],label="OLS Fit")

ax1.plot(fut_dates[:10],fore_ann_px[:10],'--',label="ANN Forecast")
ax1.plot(fut_dates[:10],fore_ols_px[:10],'--',label="OLS Forecast")
ax1.legend()
st.pyplot(fig1)

st.subheader("1-Year Forecast")
fig2,ax2=plt.subplots(figsize=(14,6))
ax2.plot(dates_all[-250:],actual_px[-250:],label="Actual")
ax2.plot(fut_dates,fore_ann_px,'--',label="ANN Forecast")
ax2.plot(fut_dates,fore_ols_px,'--',label="OLS Forecast")
ax2.legend()
st.pyplot(fig2)

# ============================================================
# ACTIVE FEATURES TABLE
# ============================================================

win_future=pd.date_range(dates_all[-1]+pd.Timedelta(days=1),
                         periods=7,freq='B')
win_all=win_future

feat_win=build_features(win_all)

rows=[]
for col in feat_win.columns:
    for date in feat_win.index[feat_win[col]==1]:
        rows.append({
            "Date":date.date(),
            "Feature":col,
            "OLS_Coef":float(ols.params[2+feat_names.index(col)])
        })

asp_df=pd.DataFrame(rows)
st.subheader("Upcoming Active Aspects (Next 7 Days)")
st.dataframe(asp_df,use_container_width=True)

# ============================================================
# SUMMARY
# ============================================================

st.markdown("---")
st.markdown("### Model Performance")
st.write("OLS R²:",round(r2_ols,4))
st.write("ANN R²:",round(r2_ann,4))
st.write("Features:",len(feat_names))
