"""
MarketPulse — AI Financial Market Intelligence
Run: streamlit run app.py
Upload a CSV containing Date and Close. Optional columns:
Open, High, Low, Volume, Sentiment_Score, Inflation_Rate,
Interest_Rate, Unemployment_Rate.
"""
from __future__ import annotations

import io
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

st.set_page_config(
    page_title="MarketPulse | Financial Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------- Theme -----------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: #08111f; color: #e8eef8; }
    [data-testid="stHeader"] { background: rgba(8,17,31,.92); }
    [data-testid="stSidebar"] { background: #101d31; border-right: 1px solid #22344d; }
    [data-testid="stSidebar"] * { color: #e8eef8; }
    h1,h2,h3 { font-family: 'Space Grotesk', sans-serif !important; color: #f4f7ff; }
    p, label, .stMarkdown { color: #c5d0e0; }
    .hero {
      padding: 28px 32px; border: 1px solid #28496b; border-radius: 20px;
      background: linear-gradient(115deg,#14243c 0%,#123b5b 55%,#075b68 100%);
      box-shadow: 0 12px 35px rgba(0,0,0,.20); margin: 4px 0 22px;
    }
    .hero h1 { margin:0; font-size:2.35rem; color:#fff; }
    .hero p { margin:8px 0 0; color:#d4e8f6; font-size:1rem; }
    .eyebrow { color:#54d6c8; font-weight:700; letter-spacing:.14em; font-size:.72rem; }
    div[data-testid="stMetric"] {
      background: linear-gradient(145deg,#14243a,#101d30);
      border:1px solid #263d58; padding:15px 17px; border-radius:15px;
    }
    div[data-testid="stMetricLabel"] { color:#9fb2ca !important; }
    div[data-testid="stMetricValue"] { color:#f4f7ff !important; }
    div[data-testid="stMetricDelta"] { font-weight:600; }
    .panel {
      background:#101d30; border:1px solid #263d58; border-radius:15px;
      padding:16px 18px; margin:8px 0 14px;
    }
    .callout {
      background:#102b37; border-left:4px solid #35c7b6; padding:13px 16px;
      border-radius:8px; color:#d8e9f0; margin:12px 0;
    }
    .muted { color:#91a5bf; font-size:.86rem; }
    [data-testid="stDataFrame"] { border:1px solid #263d58; border-radius:10px; }
    .stButton button, .stDownloadButton button {
      border-radius:10px; border:1px solid #35bdb2; font-weight:700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

COLORS = {
    "teal": "#35c7b6",
    "blue": "#5b9dff",
    "orange": "#ffb454",
    "pink": "#f27eaa",
    "purple": "#a78bfa",
    "green": "#6bd49a",
}
MODEL_NAMES = ["Linear Regression", "Random Forest", "Gradient Boosting", "Extra Trees"]


# ---------------------------- Data handling ---------------------------------
@st.cache_data(show_spinner=False)
def read_csv(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


def add_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Create indicators using current and past observations only."""
    df = data.copy()
    close = df["Close"]
    df["Daily_Return_%"] = close.pct_change() * 100
    df["Log_Return"] = np.log(close.where(close > 0)).diff()
    df["MA_5"] = close.rolling(5, min_periods=1).mean()
    df["MA_20"] = close.rolling(20, min_periods=1).mean()
    df["MA_50"] = close.rolling(50, min_periods=1).mean()
    df["Volatility_10"] = df["Daily_Return_%"].rolling(10, min_periods=2).std()
    rolling_mean = close.rolling(20, min_periods=2).mean()
    rolling_std = close.rolling(20, min_periods=2).std()
    df["BB_Upper"] = rolling_mean + 2 * rolling_std
    df["BB_Lower"] = rolling_mean - 2 * rolling_std

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=2).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=2).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI_14"] = 100 - (100 / (1 + rs))
    df["RSI_14"] = df["RSI_14"].fillna(50)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["Volume_Change_%"] = df["Volume"].pct_change() * 100 if "Volume" in df else np.nan
    df["Drawdown_%"] = (close / close.cummax() - 1) * 100
    df["Date_Year"] = df["Date"].dt.year
    df["Date_Month"] = df["Date"].dt.month
    df["Date_Day"] = df["Date"].dt.day
    return df.replace([np.inf, -np.inf], np.nan)


def prepare_data(raw: pd.DataFrame):
    if raw.empty:
        raise ValueError("The uploaded CSV contains no records.")
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    required = {"Date", "Close"}
    if not required.issubset(df.columns):
        raise ValueError("CSV must contain at least 'Date' and 'Close' columns.")

    original_rows = len(df)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for col in df.columns:
        if col != "Date":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Date", "Close"])
    df = df.sort_values("Date").drop_duplicates("Date", keep="last").reset_index(drop=True)
    if (df["Close"] <= 0).any():
        df = df[df["Close"] > 0].reset_index(drop=True)
    if len(df) < 12:
        raise ValueError("At least 12 valid dated observations are recommended.")

    # Forward fill only: do not interpolate using future observations.
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    for col in numeric_cols:
        df[col] = df[col].ffill()

    df = add_indicators(df)
    df["Target_Next_Close"] = df["Close"].shift(-1)

    base_features = [
        "Open", "High", "Low", "Close", "Volume", "Sentiment_Score",
        "Inflation_Rate", "Interest_Rate", "Unemployment_Rate",
        "Daily_Return_%", "Log_Return", "MA_5", "MA_20", "MA_50",
        "Volatility_10", "BB_Upper", "BB_Lower", "RSI_14",
        "MACD", "MACD_Signal", "Volume_Change_%", "Drawdown_%",
        "Date_Year", "Date_Month", "Date_Day",
    ]
    features = [c for c in base_features if c in df.columns and df[c].notna().any()]
    model_df = df.dropna(subset=["Target_Next_Close"]).copy()
    # Remaining initial rolling-window NaNs are replaced using training-safe neutral values.
    model_df[features] = model_df[features].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df, model_df, features, original_rows


def make_model(name: str):
    if name == "Linear Regression":
        return make_pipeline(StandardScaler(), LinearRegression())
    if name == "Random Forest":
        return RandomForestRegressor(
            n_estimators=300, max_depth=10, min_samples_leaf=2,
            random_state=42, n_jobs=-1,
        )
    if name == "Gradient Boosting":
        return GradientBoostingRegressor(
            n_estimators=220, learning_rate=0.035, max_depth=2,
            loss="huber", random_state=42,
        )
    return ExtraTreesRegressor(
        n_estimators=300, max_depth=10, min_samples_leaf=2,
        random_state=42, n_jobs=-1,
    )


def evaluate_models(model_df: pd.DataFrame, features: list[str], train_pct: int):
    split = int(len(model_df) * train_pct / 100)
    split = min(max(split, 5), len(model_df) - 2)
    train, test = model_df.iloc[:split], model_df.iloc[split:]
    actual = test["Target_Next_Close"].to_numpy()
    output, fitted = [], {}
    for name in MODEL_NAMES:
        model = make_model(name)
        model.fit(train[features], train["Target_Next_Close"])
        pred = model.predict(test[features])
        mae = mean_absolute_error(actual, pred)
        rmse = np.sqrt(mean_squared_error(actual, pred))
        nonzero = np.abs(actual) > 1e-12
        mape = float(np.mean(np.abs((actual[nonzero] - pred[nonzero]) / actual[nonzero])) * 100) if nonzero.any() else np.nan
        actual_direction = np.sign(actual - test["Close"].to_numpy())
        predicted_direction = np.sign(pred - test["Close"].to_numpy())
        direction_acc = float(np.mean(actual_direction == predicted_direction) * 100)
        output.append({
            "Algorithm": name,
            "MAE": mae,
            "RMSE": rmse,
            "MAPE (%)": mape,
            "R²": r2_score(actual, pred) if len(actual) > 1 else np.nan,
            "Directional Accuracy (%)": direction_acc,
        })
        fitted[name] = (model, pred)
    metrics = pd.DataFrame(output).sort_values("RMSE").reset_index(drop=True)
    dates = test["Date"].to_numpy()
    prediction_frames = {}
    for name, (_, pred) in fitted.items():
        prediction_frames[name] = pd.DataFrame({
            "Date": dates,
            "Actual Next Close": actual,
            "Predicted Next Close": pred,
        })
    return metrics, prediction_frames


def money(value):
    return f"{value:,.2f}"


# ---------------------------- Sidebar ---------------------------------------
with st.sidebar:
    st.markdown("## 📈 MarketPulse")
    st.caption("AI Financial Market Intelligence")
    st.markdown("---")
    uploaded = st.file_uploader("Upload market dataset", type=["csv"])
    page = st.radio(
        "WORKSPACE",
        ["Executive Overview", "Exploratory Data Analysis", "Algorithm Benchmark",
         "Financial Forecast", "Technical Analysis", "Data Quality"],
    )
    st.markdown("---")
    st.markdown("**Required CSV fields**")
    st.caption("Date, Close")
    st.markdown("**Recommended fields**")
    st.caption("Open, High, Low, Volume")
    st.caption("Sentiment_Score, Inflation_Rate, Interest_Rate, Unemployment_Rate")
    st.markdown("---")
    st.caption("Market research prototype · Not investment advice")

if uploaded is None:
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">AI-POWERED MARKET ANALYTICS</div>
          <h1>MarketPulse</h1>
          <p>Explore market behaviour, evaluate forecasting algorithms and inspect data-driven financial indicators.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info("Upload your financial market CSV using the sidebar to open the dashboard.")
    st.markdown("### Dashboard modules")
    cards = st.columns(3)
    modules = [
        ("01 · Overview", "Market snapshot, price movement, volume and key statistics."),
        ("02 · EDA Studio", "Distributions, correlations, returns, volatility and feature relationships."),
        ("03 · Model Benchmark", "Chronological model evaluation with comparable error metrics."),
        ("04 · Forecast Lab", "Next-observation closing-price estimates from four algorithms."),
        ("05 · Technical Analysis", "RSI, MACD, moving averages and Bollinger Bands."),
        ("06 · Data Quality", "Missing values, duplicates, descriptive statistics and exports."),
    ]
    for i, (heading, description) in enumerate(modules):
        with cards[i % 3]:
            st.markdown(f'<div class="panel"><b>{heading}</b><p class="muted">{description}</p></div>', unsafe_allow_html=True)
    st.stop()

try:
    raw = read_csv(uploaded.getvalue())
    df, model_df, features, original_rows = prepare_data(raw)
except Exception as exc:
    st.error(f"Unable to prepare this dataset: {exc}")
    st.stop()

latest = df.iloc[-1]
first_close, last_close = float(df["Close"].iloc[0]), float(latest["Close"])
period_change = last_close - first_close
period_pct = period_change / first_close * 100 if first_close else 0
date_min, date_max = df["Date"].min().date(), df["Date"].max().date()

st.markdown(
    f"""
    <div class="hero">
      <div class="eyebrow">FINANCIAL MARKET INTELLIGENCE PLATFORM</div>
      <h1>MarketPulse <span style="color:#55d8ca">/</span> Analytics</h1>
      <p>Market overview · Exploratory analytics · Machine learning benchmarks · Financial forecasting</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Global date filter
f1, f2, f3 = st.columns([1, 1, 2])
with f1:
    start_date = st.date_input("Start date", value=date_min, min_value=date_min, max_value=date_max)
with f2:
    end_date = st.date_input("End date", value=date_max, min_value=date_min, max_value=date_max)
with f3:
    st.markdown(
        f'<div class="muted" style="padding-top:32px">Dataset: {date_min} to {date_max} &nbsp;·&nbsp; '
        f'{len(df):,} valid rows &nbsp;·&nbsp; {len(features)} model features</div>',
        unsafe_allow_html=True,
    )
if start_date <= end_date:
    view = df[df["Date"].dt.date.between(start_date, end_date)].copy()
else:
    st.warning("Start date must be before end date. The full dataset is displayed.")
    view = df.copy()
if view.empty:
    view = df.copy()
    st.info("No observations match this date range; showing the full dataset.")

# ---------------------------- Overview --------------------------------------
if page == "Executive Overview":
    st.markdown("### Executive market snapshot")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Latest closing price", money(last_close), f"{latest['Daily_Return_%']:+.2f}% daily" if pd.notna(latest["Daily_Return_%"]) else None)
    k2.metric("Period performance", f"{period_pct:+.2f}%", f"{period_change:+,.2f} price units")
    k3.metric("Observations", f"{len(df):,}", f"{len(df.columns)} columns")
    sentiment = latest.get("Sentiment_Score", np.nan)
    k4.metric("Latest sentiment", f"{sentiment:.3f}" if pd.notna(sentiment) else "Not provided")

    left, right = st.columns([1.7, 1])
    with left:
        st.markdown("#### Closing price & trend")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=view["Date"], y=view["Close"], name="Close", line=dict(color=COLORS["teal"], width=2.6)))
        fig.add_trace(go.Scatter(x=view["Date"], y=view["MA_20"], name="20-period MA", line=dict(color=COLORS["orange"], width=1.8, dash="dash")))
        fig.update_layout(height=390, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", hovermode="x unified", margin=dict(l=8,r=8,t=25,b=8))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("#### Latest available market data")
        fields = [c for c in ["Open", "High", "Low", "Close", "Volume", "Sentiment_Score", "Inflation_Rate", "Interest_Rate", "Unemployment_Rate"] if c in df]
        st.dataframe(pd.DataFrame({"Indicator": fields, "Latest value": [latest[c] for c in fields]}), hide_index=True, use_container_width=True, height=350)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Trading activity")
        if "Volume" in view:
            fig = px.bar(view, x="Date", y="Volume", color="Daily_Return_%", color_continuous_scale="RdYlGn", title="Volume coloured by daily return")
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=320)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Volume is not available in this dataset.")
    with c2:
        st.markdown("#### Return profile")
        fig = px.histogram(view.dropna(subset=["Daily_Return_%"]), x="Daily_Return_%", nbins=35, marginal="box", title="Daily return distribution")
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=320)
        st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="callout">Forecasts are model-based estimates derived from the uploaded historical dataset. They are not live quotes, guaranteed outcomes or investment recommendations.</div>', unsafe_allow_html=True)

# ---------------------------- EDA -------------------------------------------
elif page == "Exploratory Data Analysis":
    st.markdown("### Exploratory Data Analysis")
    st.caption("Use the date filter above to explore the selected period.")
    numeric = [c for c in view.select_dtypes(include=np.number).columns if c != "Target_Next_Close"]
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Price & volume", "Distributions", "Returns & risk", "Correlation", "Feature explorer"])

    with tab1:
        st.markdown("#### Price series")
        price_fig = go.Figure()
        price_fig.add_trace(go.Scatter(x=view["Date"], y=view["Close"], name="Close", line=dict(color=COLORS["teal"], width=2.4)))
        price_fig.add_trace(go.Scatter(x=view["Date"], y=view["MA_5"], name="MA 5", line=dict(color=COLORS["orange"], dash="dot")))
        price_fig.add_trace(go.Scatter(x=view["Date"], y=view["MA_20"], name="MA 20", line=dict(color=COLORS["blue"], dash="dash")))
        price_fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=400, hovermode="x unified")
        st.plotly_chart(price_fig, use_container_width=True)
        if "Volume" in view:
            vf = px.area(view, x="Date", y="Volume", title="Trading volume over time")
            vf.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=300)
            st.plotly_chart(vf, use_container_width=True)

    with tab2:
        selected_dist = st.selectbox("Choose a numeric variable", numeric, key="dist_feature")
        a, b = st.columns(2)
        with a:
            fig = px.histogram(view, x=selected_dist, nbins=40, marginal="rug", title=f"{selected_dist}: histogram")
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=370)
            st.plotly_chart(fig, use_container_width=True)
        with b:
            fig = px.box(view, y=selected_dist, points="outliers", title=f"{selected_dist}: box plot")
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=370)
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(view[selected_dist].describe().to_frame("Statistics").round(4), use_container_width=True)

    with tab3:
        a, b = st.columns(2)
        with a:
            fig = px.line(view, x="Date", y="Daily_Return_%", title="Daily percentage returns", color_discrete_sequence=[COLORS["pink"]])
            fig.add_hline(y=0, line_dash="dash", line_color="#9aa9bd")
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=350)
            st.plotly_chart(fig, use_container_width=True)
        with b:
            fig = px.line(view, x="Date", y="Volatility_10", title="10-period rolling volatility", color_discrete_sequence=[COLORS["orange"]])
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=350)
            st.plotly_chart(fig, use_container_width=True)
        fig = px.line(view, x="Date", y="Drawdown_%", title="Historical drawdown", color_discrete_sequence=[COLORS["purple"]])
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=300)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        corr_cols = [c for c in numeric if view[c].nunique(dropna=True) > 1]
        if len(corr_cols) >= 2:
            corr = view[corr_cols].corr()
            fig = px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r", title="Pearson correlation matrix")
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=max(500, len(corr_cols) * 32))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("At least two varying numeric columns are needed for correlation analysis.")

    with tab5:
        xcol = st.selectbox("X-axis feature", numeric, index=0, key="eda_x")
        y_options = [c for c in numeric if c != xcol]
        if y_options:
            ycol = st.selectbox("Y-axis feature", y_options, key="eda_y")
            fig = px.scatter(view, x=xcol, y=ycol, color="Daily_Return_%" if "Daily_Return_%" in view else None,
                             hover_data=["Date"], trendline="ols", title=f"{ycol} vs {xcol}")
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=420)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Another numeric feature is required for a scatter plot.")

# ---------------------------- Model benchmark -------------------------------
elif page == "Algorithm Benchmark":
    st.markdown("### Machine learning algorithm benchmark")
    st.caption("All algorithms use the same chronological train/test split and feature set for a fair comparison.")
    train_pct = st.slider("Training observations (%)", 60, 90, 80, 5, key="benchmark_train")
    if len(model_df) < 15:
        st.warning("A larger dataset is recommended for a meaningful model comparison.")
    metrics, predictions = evaluate_models(model_df, features, train_pct)
    best = metrics.iloc[0]
    m1, m2, m3 = st.columns(3)
    m1.metric("Lowest RMSE", best["Algorithm"])
    m2.metric("Best model RMSE", f"{best['RMSE']:.4f}")
    m3.metric("Best directional accuracy", f"{metrics['Directional Accuracy (%)'].max():.1f}%")
    st.dataframe(metrics.style.format({
        "MAE": "{:.4f}", "RMSE": "{:.4f}", "MAPE (%)": "{:.2f}",
        "R²": "{:.4f}", "Directional Accuracy (%)": "{:.2f}",
    }), use_container_width=True, hide_index=True)

    chart_metric = st.selectbox("Compare algorithms by metric", ["RMSE", "MAE", "MAPE (%)", "R²", "Directional Accuracy (%)"])
    fig = px.bar(metrics.sort_values(chart_metric, ascending=(chart_metric != "R²" and chart_metric != "Directional Accuracy (%)")),
                 x="Algorithm", y=chart_metric, color="Algorithm",
                 color_discrete_sequence=list(COLORS.values()), title=f"Algorithm comparison — {chart_metric}")
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False, height=380)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Actual vs predicted closing prices")
    selected_model = st.selectbox("Select algorithm", MODEL_NAMES, key="benchmark_model")
    result = predictions[selected_model]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result["Date"], y=result["Actual Next Close"], name="Actual", line=dict(color=COLORS["teal"], width=2.4)))
    fig.add_trace(go.Scatter(x=result["Date"], y=result["Predicted Next Close"], name="Predicted", line=dict(color=COLORS["orange"], width=2, dash="dash")))
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=400, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
    st.download_button("Download algorithm comparison (CSV)", metrics.to_csv(index=False).encode("utf-8"), "algorithm_comparison.csv", "text/csv")
    st.download_button("Download selected model predictions", result.to_csv(index=False).encode("utf-8"), "model_predictions.csv", "text/csv")
    st.markdown('<div class="callout">Interpretation: lower MAE, RMSE and MAPE indicate smaller forecast errors. Higher R² and directional accuracy are generally preferable. No single metric guarantees future performance.</div>', unsafe_allow_html=True)

# ---------------------------- Forecast --------------------------------------
elif page == "Financial Forecast":
    st.markdown("### Financial forecast laboratory")
    st.caption("The forecast estimates the next row in the uploaded sequence, not a guaranteed calendar-day or live-market price.")
    train_pct = st.slider("Training observations (%)", 60, 90, 80, 5, key="forecast_train")
    metrics, predictions = evaluate_models(model_df, features, train_pct)
    latest_features = df.iloc[[-1]][features].replace([np.inf, -np.inf], np.nan).fillna(0)
    forecasts = []
    for name in MODEL_NAMES:
        model = make_model(name)
        model.fit(model_df[features], model_df["Target_Next_Close"])
        predicted = float(model.predict(latest_features)[0])
        delta = predicted - last_close
        forecasts.append({
            "Algorithm": name,
            "Estimated Next Close": predicted,
            "Change vs Latest (%)": (delta / last_close * 100) if last_close else np.nan,
        })
    forecast_df = pd.DataFrame(forecasts)
    chosen = st.selectbox("Highlight forecast from", MODEL_NAMES, index=metrics["Algorithm"].tolist().index(metrics.iloc[0]["Algorithm"]))
    chosen_row = forecast_df[forecast_df["Algorithm"] == chosen].iloc[0]
    f1, f2, f3 = st.columns(3)
    f1.metric("Estimated next close", money(chosen_row["Estimated Next Close"]))
    f2.metric("Implied price movement", f"{chosen_row['Change vs Latest (%)']:+.2f}%")
    f3.metric("Selected algorithm", chosen)
    st.markdown("#### Forecast comparison across algorithms")
    st.dataframe(forecast_df.style.format({"Estimated Next Close": "{:,.4f}", "Change vs Latest (%)": "{:+.3f}%"}), hide_index=True, use_container_width=True)
    fig = px.bar(forecast_df, x="Algorithm", y="Estimated Next Close", color="Algorithm",
                 color_discrete_sequence=list(COLORS.values()), title="Next-observation closing-price estimates")
    fig.add_hline(y=last_close, line_dash="dash", line_color="#ffffff", annotation_text="Latest actual close")
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False, height=390)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("#### Historical test performance")
    st.dataframe(metrics.style.format({"MAE": "{:.4f}", "RMSE": "{:.4f}", "MAPE (%)": "{:.2f}", "R²": "{:.3f}", "Directional Accuracy (%)": "{:.1f}%"}), hide_index=True, use_container_width=True)
    st.download_button("Download forecast estimates", forecast_df.to_csv(index=False).encode("utf-8"), "financial_forecasts.csv", "text/csv")
    st.markdown('<div class="callout">Important: this is an experimental one-step statistical estimate using historical input features. It does not account for all market events, transaction costs, liquidity, or unexpected news. It is not financial advice.</div>', unsafe_allow_html=True)

# ---------------------------- Technical analysis ----------------------------
elif page == "Technical Analysis":
    st.markdown("### Technical analysis workspace")
    st.caption("Indicators are calculated from historical price and volume observations in the uploaded file.")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=view["Date"], y=view["Close"], name="Close", line=dict(color=COLORS["teal"], width=2.2)))
    fig.add_trace(go.Scatter(x=view["Date"], y=view["MA_20"], name="MA 20", line=dict(color=COLORS["orange"], dash="dash")))
    fig.add_trace(go.Scatter(x=view["Date"], y=view["BB_Upper"], name="Bollinger upper", line=dict(color=COLORS["purple"], width=1, dash="dot")))
    fig.add_trace(go.Scatter(x=view["Date"], y=view["BB_Lower"], name="Bollinger lower", line=dict(color=COLORS["purple"], width=1, dash="dot"), fill="tonexty", fillcolor="rgba(167,139,250,.08)"))
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=420, hovermode="x unified", title="Price, moving average and Bollinger Bands")
    st.plotly_chart(fig, use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=view["Date"], y=view["RSI_14"], name="RSI", line=dict(color=COLORS["pink"], width=2)))
        fig.add_hline(y=70, line_dash="dash", line_color="#ffb454")
        fig.add_hline(y=30, line_dash="dash", line_color="#35c7b6")
        fig.update_yaxes(range=[0, 100])
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=330, title="Relative Strength Index (14)")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=view["Date"], y=view["MACD"], name="MACD", line=dict(color=COLORS["blue"], width=2)))
        fig.add_trace(go.Scatter(x=view["Date"], y=view["MACD_Signal"], name="Signal", line=dict(color=COLORS["orange"], width=2)))
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=330, title="MACD and signal line")
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("#### Latest indicator readings")
    indicator_cols = ["MA_5", "MA_20", "MA_50", "RSI_14", "MACD", "MACD_Signal", "Volatility_10", "Drawdown_%"]
    st.dataframe(pd.DataFrame({"Indicator": indicator_cols, "Latest reading": [latest[c] for c in indicator_cols]}).round(4), hide_index=True, use_container_width=True)
    st.caption("RSI thresholds and moving-average signals are commonly used technical-analysis conventions, not standalone buy/sell instructions.")

# ---------------------------- Data quality ----------------------------------
elif page == "Data Quality":
    st.markdown("### Data quality & preparation")
    missing = raw.isna().sum()
    quality = pd.DataFrame({
        "Column": raw.columns,
        "Data type": [str(raw[c].dtype) for c in raw.columns],
        "Missing values": [int(missing[c]) for c in raw.columns],
        "Missing (%)": [(missing[c] / max(len(raw), 1) * 100).round(2) for c in raw.columns],
        "Unique values": [int(raw[c].nunique(dropna=True)) for c in raw.columns],
    })
    duplicates = int(raw.duplicated().sum())
    invalid_dates = int(pd.to_datetime(raw["Date"], errors="coerce").isna().sum()) if "Date" in raw else len(raw)
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Uploaded rows", f"{original_rows:,}")
    q2.metric("Valid rows", f"{len(df):,}")
    q3.metric("Duplicate rows", f"{duplicates:,}")
    q4.metric("Invalid dates", f"{invalid_dates:,}")
    st.markdown("#### Column-level quality report")
    st.dataframe(quality, hide_index=True, use_container_width=True)
    st.markdown("#### Descriptive statistics")
    st.dataframe(df.select_dtypes(include=np.number).describe().T.round(4), use_container_width=True)
    st.markdown("#### Data preparation methodology")
    st.markdown("""
    - Column names are trimmed and numeric columns are converted to numeric data types.
    - Invalid dates and missing closing prices are removed; records are sorted chronologically.
    - Duplicate dates are reduced to the last available record for that date.
    - Predictor gaps are forward-filled only; future observations are not used to interpolate earlier predictor values.
    - Derived indicators include daily/log returns, moving averages, RSI, MACD, Bollinger Bands, rolling volatility and drawdown.
    - The supervised target is the next row's closing price. The final row is excluded from model evaluation because its next close is not known.
    - Model evaluation uses a chronological split, with earlier observations for training and later observations for testing.
    """)
    st.download_button("Download cleaned dataset", df.to_csv(index=False).encode("utf-8"), "market_data_cleaned.csv", "text/csv")
    st.download_button("Download data quality report", quality.to_csv(index=False).encode("utf-8"), "data_quality_report.csv", "text/csv")

st.markdown("---")
st.markdown(
    '<div class="muted">MarketPulse · Internship project prototype · Historical-data research only · '
    'Model forecasts are uncertain and must not be treated as financial advice.</div>',
    unsafe_allow_html=True,
)
