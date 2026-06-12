import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import warnings, os
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TSLA Stock Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main { background-color: #0d1117; }

.stApp { background-color: #0d1117; color: #e6edf3; }

h1, h2, h3 { font-family: 'Space Mono', monospace !important; }

.metric-card {
    background: linear-gradient(135deg, #161b22, #21262d);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    margin: 5px;
}
.metric-value { font-size: 2rem; font-weight: 700; font-family: 'Space Mono', monospace; }
.metric-label { font-size: 0.8rem; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; }

.winner-badge {
    background: linear-gradient(90deg, #238636, #2ea043);
    color: white;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
}

.stSelectbox label, .stSlider label { color: #e6edf3 !important; }
.stSidebar { background-color: #161b22 !important; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(file):
    df = pd.read_csv(file, parse_dates=['Date'])
    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['Adj Close'] = df['Adj Close'].ffill().bfill()
    return df

def create_sequences(data_array, look_back):
    X, y = [], []
    for i in range(look_back, len(data_array)):
        X.append(data_array[i - look_back:i, 0])
        y.append(data_array[i, 0])
    return np.array(X), np.array(y)

def build_model(model_type, units, dropout, lr, look_back):
    model = Sequential()
    if model_type == "SimpleRNN":
        model.add(SimpleRNN(units, return_sequences=True, input_shape=(look_back, 1), activation='tanh'))
        model.add(Dropout(dropout))
        model.add(SimpleRNN(units // 2, return_sequences=False, activation='tanh'))
    else:
        model.add(LSTM(units, return_sequences=True, input_shape=(look_back, 1)))
        model.add(Dropout(dropout))
        model.add(LSTM(units // 2, return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error', metrics=['mae'])
    return model

def multi_step_forecast(model, last_seq, scaler, n_steps, look_back):
    seq = last_seq.copy()
    preds = []
    for _ in range(n_steps):
        inp = seq[-look_back:].reshape(1, look_back, 1)
        pred = model.predict(inp, verbose=0)[0, 0]
        preds.append(pred)
        seq = np.append(seq, pred)
    return scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚗 TSLA Predictor")
    st.markdown("---")
    
    uploaded_file = st.file_uploader("Upload TSLA.csv", type=['csv'])
    
    st.markdown("### ⚙️ Model Settings")
    look_back   = st.slider("Look-back Window (days)", 30, 120, 60, 10)
    units       = st.selectbox("LSTM/RNN Units", [32, 64, 128], index=1)
    dropout     = st.slider("Dropout Rate", 0.1, 0.5, 0.2, 0.05)
    lr          = st.selectbox("Learning Rate", [0.001, 0.0005, 0.0001], index=0)
    epochs      = st.slider("Max Epochs", 20, 100, 50, 10)
    train_split = st.slider("Train Split %", 60, 90, 80, 5)
    
    st.markdown("### 📈 Forecast")
    forecast_days = st.selectbox("Forecast Horizon", [1, 5, 10], index=1)
    
    run_btn = st.button("🚀 Train & Predict", use_container_width=True, type="primary")

# ── Main ───────────────────────────────────────────────────────────────────────
st.markdown("# 🚗 Tesla Stock Price Prediction")
st.markdown("##### SimpleRNN vs LSTM — Deep Learning Time Series Analysis")
st.markdown("---")

if uploaded_file is None:
    st.info("👈 Upload **TSLA.csv** in the sidebar to get started.")
    
    # Show sample layout
    col1, col2, col3, col4 = st.columns(4)
    for col, label, val, color in [
        (col1, "RMSE", "~41.78", "#e74c3c"),
        (col2, "MAE",  "~22.16", "#f39c12"),
        (col3, "MAPE", "~6.56%", "#2ecc71"),
        (col4, "R²",   "~0.66",  "#3498db"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:{color}">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)
    st.stop()

# Load data
df = load_data(uploaded_file)

# ── EDA Tab ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Data Overview", "🤖 Model Training", "🔮 Forecast"])

with tab1:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Date Range", f"{df['Date'].min().year}–{df['Date'].max().year}")
    col3.metric("Max Price", f"${df['Adj Close'].max():.2f}")
    col4.metric("Min Price", f"${df['Adj Close'].min():.2f}")
    
    # Price chart
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), facecolor='#0d1117')
    for ax in axes:
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#8b949e')
        for spine in ax.spines.values(): spine.set_color('#30363d')
    
    axes[0].plot(df['Date'], df['Adj Close'], color='#58a6ff', linewidth=1.2)
    axes[0].fill_between(df['Date'], df['Adj Close'], alpha=0.1, color='#58a6ff')
    axes[0].plot(df['Date'], df['Adj Close'].rolling(90).mean(), color='#f85149', linewidth=1.5, label='90-Day MA')
    axes[0].set_title('TSLA Adjusted Closing Price', color='#e6edf3', fontsize=13, pad=10)
    axes[0].set_ylabel('Price (USD)', color='#8b949e')
    axes[0].legend(facecolor='#21262d', edgecolor='#30363d', labelcolor='#e6edf3')
    
    df['Daily_Return'] = df['Adj Close'].pct_change() * 100
    axes[1].bar(df['Date'], df['Daily_Return'], color='#3fb950', alpha=0.6, width=1.5)
    axes[1].axhline(0, color='#f85149', linewidth=0.8)
    axes[1].set_title('Daily Return (%)', color='#e6edf3', fontsize=13, pad=10)
    axes[1].set_ylabel('Return (%)', color='#8b949e')
    
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    st.dataframe(df[['Date','Open','High','Low','Close','Adj Close','Volume']].tail(10),
                 use_container_width=True)

with tab2:
    if not run_btn:
        st.info("👈 Configure settings in the sidebar and click **Train & Predict**")
        st.stop()
    
    # Prepare data
    prices = df['Adj Close'].values.reshape(-1, 1)
    split_idx = int(len(prices) * train_split / 100)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(prices[:split_idx])
    test_scaled  = scaler.transform(prices[split_idx:])
    full_scaled  = np.concatenate([train_scaled, test_scaled])
    
    X_train, y_train = create_sequences(train_scaled, look_back)
    test_input = full_scaled[len(train_scaled) - look_back:]
    X_test, y_test = create_sequences(test_input, look_back)
    
    X_train = X_train.reshape(*X_train.shape, 1)
    X_test  = X_test.reshape(*X_test.shape, 1)
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=0)
    ]
    
    results = {}
    histories = {}
    
    for model_type, color in [("SimpleRNN", "#e74c3c"), ("LSTM", "#2ecc71")]:
        with st.spinner(f"Training {model_type}..."):
            model = build_model(model_type, units, dropout, lr, look_back)
            history = model.fit(
                X_train, y_train,
                epochs=epochs, batch_size=32,
                validation_split=0.1,
                callbacks=callbacks, verbose=0
            )
            preds_scaled = model.predict(X_test, verbose=0)
            preds  = scaler.inverse_transform(preds_scaled)
            actual = scaler.inverse_transform(y_test.reshape(-1, 1))
            
            rmse = np.sqrt(mean_squared_error(actual, preds))
            mae  = mean_absolute_error(actual, preds)
            mape = np.mean(np.abs((actual - preds) / actual)) * 100
            r2   = r2_score(actual, preds)
            
            results[model_type] = {
                'model': model, 'preds': preds, 'actual': actual,
                'rmse': rmse, 'mae': mae, 'mape': mape, 'r2': r2,
                'color': color, 'history': history
            }
            histories[model_type] = history
        st.success(f"✅ {model_type} trained! RMSE: ${rmse:.2f} | R²: {r2:.4f}")
    
    # Store in session state
    st.session_state['results']     = results
    st.session_state['scaler']      = scaler
    st.session_state['full_scaled'] = full_scaled
    st.session_state['look_back']   = look_back
    st.session_state['df']          = df
    st.session_state['split_idx']   = split_idx
    
    # Metrics comparison
    st.markdown("### 📊 Model Comparison")
    col1, col2 = st.columns(2)
    
    for col, mtype in zip([col1, col2], ["SimpleRNN", "LSTM"]):
        r = results[mtype]
        with col:
            st.markdown(f"#### {mtype}")
            m1, m2 = st.columns(2)
            m1.metric("RMSE", f"${r['rmse']:.2f}")
            m2.metric("MAE",  f"${r['mae']:.2f}")
            m3, m4 = st.columns(2)
            m3.metric("MAPE", f"{r['mape']:.2f}%")
            m4.metric("R²",   f"{r['r2']:.4f}")
    
    # Winner
    winner = "SimpleRNN" if results["SimpleRNN"]["rmse"] < results["LSTM"]["rmse"] else "LSTM"
    st.markdown(f"### 🏆 Winner: **{winner}** (lower RMSE)")
    
    # Actual vs Predicted
    st.markdown("### 📈 Actual vs Predicted")
    test_dates = df['Date'].values[split_idx + look_back - len(y_test):]
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), facecolor='#0d1117')
    for ax, mtype in zip(axes, ["SimpleRNN", "LSTM"]):
        r = results[mtype]
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#8b949e')
        for spine in ax.spines.values(): spine.set_color('#30363d')
        ax.plot(test_dates[-len(r['actual']):], r['actual'], color='#58a6ff', linewidth=1.5, label='Actual')
        ax.plot(test_dates[-len(r['preds']):],  r['preds'],  color=r['color'], linewidth=1.2,
                linestyle='--', label=f'{mtype} Predicted')
        ax.set_title(f"{mtype} — RMSE: ${r['rmse']:.2f}", color='#e6edf3', fontsize=12)
        ax.set_ylabel('Price (USD)', color='#8b949e')
        ax.legend(facecolor='#21262d', edgecolor='#30363d', labelcolor='#e6edf3')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with tab3:
    if 'results' not in st.session_state:
        st.info("👈 Train models first in the **Model Training** tab.")
        st.stop()
    
    results     = st.session_state['results']
    scaler      = st.session_state['scaler']
    full_scaled = st.session_state['full_scaled']
    look_back   = st.session_state['look_back']
    
    last_seq = full_scaled[-look_back:, 0]
    last_price = scaler.inverse_transform([[last_seq[-1]]])[0, 0]
    
    st.markdown(f"### 🔮 {forecast_days}-Day Price Forecast")
    st.markdown(f"**Last Known Price:** `${last_price:.2f}`")
    
    fig, ax = plt.subplots(figsize=(10, 5), facecolor='#0d1117')
    ax.set_facecolor('#161b22')
    ax.tick_params(colors='#8b949e')
    for spine in ax.spines.values(): spine.set_color('#30363d')
    
    days = np.arange(1, forecast_days + 1)
    
    for mtype, color in [("SimpleRNN", "#e74c3c"), ("LSTM", "#2ecc71")]:
        forecast = multi_step_forecast(results[mtype]['model'], last_seq, scaler, forecast_days, look_back)
        ax.plot(days, forecast, 'o-', color=color, label=mtype, linewidth=2, markersize=8)
        
        # Show values on chart
        for d, p in zip(days, forecast):
            ax.annotate(f'${p:.0f}', (d, p), textcoords="offset points",
                       xytext=(0, 10), ha='center', color=color, fontsize=9)
    
    ax.axhline(last_price, color='#58a6ff', linestyle='--', alpha=0.7, label='Last Known')
    ax.set_title(f'TSLA {forecast_days}-Day Forecast', color='#e6edf3', fontsize=13)
    ax.set_xlabel('Days Ahead', color='#8b949e')
    ax.set_ylabel('Price (USD)', color='#8b949e')
    ax.set_xticks(days)
    ax.legend(facecolor='#21262d', edgecolor='#30363d', labelcolor='#e6edf3')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # Forecast table
    st.markdown("### 📋 Forecast Table")
    forecast_data = {'Day': [f'Day {d}' for d in days]}
    for mtype in ["SimpleRNN", "LSTM"]:
        forecast = multi_step_forecast(results[mtype]['model'], last_seq, scaler, forecast_days, look_back)
        forecast_data[mtype] = [f"${p:.2f}" for p in forecast]
    
    st.dataframe(pd.DataFrame(forecast_data), use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("<center style='color:#8b949e; font-size:0.8rem'>Tesla Stock Prediction • SimpleRNN vs LSTM • Deep Learning Project</center>",
            unsafe_allow_html=True)
