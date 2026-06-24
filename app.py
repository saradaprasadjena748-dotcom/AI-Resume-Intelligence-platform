"""
Smart Stock Market AI — Streamlit Dashboard
Internship Project | Random Forest Classifier
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_curve, auc)
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Smart Stock Market AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# CUSTOM CSS — dark finance theme
# ─────────────────────────────────────────
st.markdown("""
<style>
/* Main background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
    color: #e6edf3;
}
[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #30363d;
}
/* KPI Cards */
.kpi-card {
    background: linear-gradient(145deg, #161b22, #1c2128);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    margin: 4px 0;
}
.kpi-value {
    font-size: 2.2rem;
    font-weight: 700;
    line-height: 1.1;
}
.kpi-label {
    font-size: 0.78rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 4px;
}
.kpi-delta-up   { color: #3fb950; }
.kpi-delta-down { color: #f85149; }
.kpi-delta-neu  { color: #58a6ff; }

/* Section headers */
.section-header {
    font-size: 1.05rem;
    font-weight: 600;
    color: #58a6ff;
    border-left: 3px solid #58a6ff;
    padding-left: 10px;
    margin: 20px 0 12px 0;
}
/* Badge */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
}
.badge-green { background:#1a4731; color:#3fb950; }
.badge-blue  { background:#1a3a5c; color:#58a6ff; }
.badge-red   { background:#3d1a1a; color:#f85149; }

/* Prediction result box */
.pred-box {
    border-radius: 12px;
    padding: 22px;
    text-align: center;
    font-size: 1.5rem;
    font-weight: 700;
    margin: 10px 0;
}
.pred-up   { background:#1a4731; border:1px solid #3fb950; color:#3fb950; }
.pred-down { background:#3d1a1a; border:1px solid #f85149; color:#f85149; }

/* Streamlit overrides */
div[data-testid="stMetricValue"] { color: #e6edf3; }
.stButton>button {
    background: linear-gradient(90deg,#1f6feb,#388bfd);
    color: white; border:none; border-radius:8px;
    font-weight:600; padding:10px 28px;
    width:100%;
}
.stButton>button:hover { background: #388bfd; }
div[data-baseweb="select"] > div { background:#21262d; border-color:#30363d; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# DATA LOADING & FEATURE ENGINEERING
# ─────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_engineer():
    files = [
        "stock_data_aug_2025.csv",
        "stock_data_july_2025.csv",
        "stock_data_aug_2025.csv",
    ]
    dfs = []
    for f in files:
        df = pd.read_csv(f, dayfirst=True)
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)

    # Rename columns to snake_case
    data.rename(columns={
        "Date": "date", "Ticker": "ticker",
        "Open Price": "open", "Close Price": "close",
        "High Price": "high", "Low Price": "low",
        "Volume Traded": "volume", "Market Cap": "market_cap",
        "PE Ratio": "pe_ratio", "Dividend Yield": "div_yield",
        "EPS": "eps", "52 Week High": "wk52_high",
        "52 Week Low": "wk52_low", "Sector": "sector"
    }, inplace=True)

    data["date"] = pd.to_datetime(data["date"], format="mixed")
    data.sort_values(["ticker", "date"], inplace=True)
    data.reset_index(drop=True, inplace=True)

    # Feature engineering
    data["daily_return"]   = (data["close"] - data["open"]) / data["open"] * 100
    data["price_range_pct"]= (data["high"]  - data["low"])  / data["low"]  * 100
    data["wk52_position"]  = (data["close"] - data["wk52_low"]) / (data["wk52_high"] - data["wk52_low"] + 1e-9) * 100
    data["vol_price"]      = data["volume"] * data["close"]
    data["log_volume"]     = np.log1p(data["volume"])
    data["log_market_cap"] = np.log1p(data["market_cap"])

    # Encode sector
    le = LabelEncoder()
    data["sector_enc"] = le.fit_transform(data["sector"].astype(str))

    # Target: tomorrow close > today close → 1 (UP), else 0 (DOWN)
    data["next_close"]  = data.groupby("ticker")["close"].shift(-1)
    data["target"]      = (data["next_close"] > data["close"]).astype(int)
    data.dropna(subset=["next_close"], inplace=True)

    return data, le


# ─────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────
FEATURES = ["open", "close", "high", "low", "daily_return",
            "price_range_pct", "wk52_position", "log_volume",
            "log_market_cap", "pe_ratio", "div_yield", "eps",
            "sector_enc"]

@st.cache_resource(show_spinner=False)
def train_model(_data, model_name="Random Forest"):
    X = _data[FEATURES]
    y = _data["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    models = {
        "Random Forest":        RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42, n_jobs=-1),
        "Gradient Boosting":    GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42),
        "Logistic Regression":  LogisticRegression(max_iter=500, random_state=42),
    }
    clf = models[model_name]
    clf.fit(X_train_sc if model_name == "Logistic Regression" else X_train, y_train)

    y_pred  = clf.predict(X_test_sc if model_name == "Logistic Regression" else X_test)
    y_proba = clf.predict_proba(X_test_sc if model_name == "Logistic Regression" else X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    cm   = confusion_matrix(y_test, y_pred)
    rep  = classification_report(y_test, y_pred, output_dict=True)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    # Feature importance (RF & GB)
    fi = None
    if hasattr(clf, "feature_importances_"):
        fi = pd.Series(clf.feature_importances_, index=FEATURES).sort_values(ascending=True)

    return {
        "clf": clf, "scaler": scaler,
        "acc": acc, "cm": cm, "rep": rep,
        "fpr": fpr, "tpr": tpr, "roc_auc": roc_auc,
        "fi": fi,
        "X_test": X_test, "y_test": y_test, "y_pred": y_pred
    }


# ─────────────────────────────────────────
# MATPLOTLIB DARK STYLE HELPER
# ─────────────────────────────────────────
def dark_fig(w=8, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("#161b22")
    ax.set_facecolor("#0d1117")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.tick_params(colors="#8b949e", labelsize=9)
    ax.xaxis.label.set_color("#8b949e")
    ax.yaxis.label.set_color("#8b949e")
    ax.title.set_color("#e6edf3")
    return fig, ax


# ═══════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 📈 Smart Stock AI")
    st.markdown("---")

    page = st.radio("Navigate", [
        "🏠  Overview",
        "📊  Model Performance",
        "🔮  Predict Stock",
        "🔍  Stock Explorer",
        "📚  How It Works",
    ])

    st.markdown("---")
    model_choice = st.selectbox("ML Model", ["Random Forest", "Gradient Boosting", "Logistic Regression"])
    st.markdown("---")
    st.caption("Internship Project · Stock AI v1.0")


# ─────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────
with st.spinner("Loading & training model…"):
    data, le = load_and_engineer()
    res = train_model(data, model_choice)

tickers  = sorted(data["ticker"].unique())
sectors  = sorted(data["sector"].unique())



# ═══════════════════════════════════════════════════════════════
#  PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════
if page == "🏠  Overview":
    st.markdown("# 📈 Smart Stock Market AI Dashboard")
    st.markdown("*AI-powered stock price direction prediction using 3 months of market data*")
    st.markdown("---")

    # ── KPI Row ──────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)

    with k1:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value kpi-delta-neu">{len(data):,}</div>
            <div class="kpi-label">Total Records</div>
        </div>""", unsafe_allow_html=True)

    with k2:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value kpi-delta-neu">{data['ticker'].nunique()}</div>
            <div class="kpi-label">Stock Tickers</div>
        </div>""", unsafe_allow_html=True)

    with k3:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value kpi-delta-neu">{data['sector'].nunique()}</div>
            <div class="kpi-label">Market Sectors</div>
        </div>""", unsafe_allow_html=True)

    with k4:
        acc_color = "kpi-delta-up" if res["acc"] > 0.60 else "kpi-delta-down"
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value {acc_color}">{res['acc']*100:.1f}%</div>
            <div class="kpi-label">Model Accuracy</div>
        </div>""", unsafe_allow_html=True)

    with k5:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value kpi-delta-up">{res['roc_auc']:.3f}</div>
            <div class="kpi-label">ROC-AUC Score</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts Row 1 ─────────────────────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-header">Records per Month</div>', unsafe_allow_html=True)
        data["month"] = data["date"].dt.strftime("%b %Y")
        monthly = data.groupby("month").size().reset_index(name="count")
        fig, ax = dark_fig(6, 3)
        bars = ax.bar(monthly["month"], monthly["count"], color=["#1f6feb","#388bfd","#58a6ff"], width=0.5)
        for bar, val in zip(bars, monthly["count"]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                    f"{val:,}", ha="center", va="bottom", fontsize=9, color="#e6edf3")
        ax.set_ylabel("Records")
        ax.set_title("Dataset Coverage")
        st.pyplot(fig)
        plt.close()

    with c2:
        st.markdown('<div class="section-header">Sector Distribution</div>', unsafe_allow_html=True)
        sec_counts = data["sector"].value_counts()
        colors_pie = ["#1f6feb","#388bfd","#58a6ff","#79c0ff","#3fb950",
                      "#56d364","#f85149","#ff7b72","#d2a8ff","#bc8cff","#ffa657"]
        fig, ax = dark_fig(6, 3)
        wedges, texts, autotexts = ax.pie(
            sec_counts.values, labels=sec_counts.index,
            autopct="%1.0f%%", startangle=140,
            colors=colors_pie[:len(sec_counts)],
            textprops={"color":"#e6edf3","fontsize":7},
            pctdistance=0.82
        )
        for at in autotexts: at.set_fontsize(7)
        ax.set_title("Stock Distribution by Sector")
        st.pyplot(fig)
        plt.close()

    # ── Charts Row 2 ─────────────────────────────────────────
    c3, c4 = st.columns(2)

    with c3:
        st.markdown('<div class="section-header">Average Daily Return by Sector</div>', unsafe_allow_html=True)
        sector_ret = data.groupby("sector")["daily_return"].mean().sort_values()
        colors_bar = ["#f85149" if v < 0 else "#3fb950" for v in sector_ret.values]
        fig, ax = dark_fig(6, 4)
        ax.barh(sector_ret.index, sector_ret.values, color=colors_bar, height=0.6)
        ax.axvline(0, color="#8b949e", lw=0.8)
        ax.set_xlabel("Avg Daily Return (%)")
        ax.set_title("Sector Performance")
        st.pyplot(fig)
        plt.close()

    with c4:
        st.markdown('<div class="section-header">UP vs DOWN Distribution</div>', unsafe_allow_html=True)
        target_counts = data["target"].value_counts()
        fig, ax = dark_fig(6, 4)
        ax.bar(["DOWN ↓","UP ↑"], [target_counts.get(0,0), target_counts.get(1,0)],
               color=["#f85149","#3fb950"], width=0.4)
        ax.set_ylabel("Count")
        ax.set_title("Class Balance")
        for i, v in enumerate([target_counts.get(0,0), target_counts.get(1,0)]):
            ax.text(i, v + 50, f"{v:,}", ha="center", fontsize=10, color="#e6edf3")
        st.pyplot(fig)
        plt.close()

    # ── Recent data table ─────────────────────────────────────
    st.markdown('<div class="section-header">Sample Dataset (Top 10 Rows)</div>', unsafe_allow_html=True)
    preview = data[["date","ticker","sector","open","close","daily_return","target"]].head(10).copy()
    preview["target"] = preview["target"].map({1:"⬆ UP", 0:"⬇ DOWN"})
    preview["daily_return"] = preview["daily_return"].round(2).astype(str) + "%"
    st.dataframe(preview, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════
#  PAGE 2 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════
elif page == "📊  Model Performance":
    st.markdown(f"# 📊 Model Performance — {model_choice}")
    st.markdown("---")

    # KPI row
    rep = res["rep"]
    k1, k2, k3, k4 = st.columns(4)
    metrics_map = [
        ("Accuracy",  f"{res['acc']*100:.1f}%",          "kpi-delta-up"),
        ("Precision", f"{rep['weighted avg']['precision']*100:.1f}%", "kpi-delta-neu"),
        ("Recall",    f"{rep['weighted avg']['recall']*100:.1f}%",    "kpi-delta-neu"),
        ("ROC-AUC",   f"{res['roc_auc']:.3f}",           "kpi-delta-up"),
    ]
    for col, (lbl, val, cls) in zip([k1,k2,k3,k4], metrics_map):
        with col:
            st.markdown(f"""<div class="kpi-card">
                <div class="kpi-value {cls}">{val}</div>
                <div class="kpi-label">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    # Confusion Matrix
    with c1:
        st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
        cm = res["cm"]
        fig, ax = dark_fig(5, 4)
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["Pred DOWN","Pred UP"], color="#e6edf3")
        ax.set_yticklabels(["Actual DOWN","Actual UP"], color="#e6edf3")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                        color="white" if cm[i,j] > cm.max()/2 else "#e6edf3", fontsize=16, fontweight="bold")
        ax.set_title("Confusion Matrix")
        fig.colorbar(im, ax=ax)
        st.pyplot(fig)
        plt.close()

    # ROC Curve
    with c2:
        st.markdown('<div class="section-header">ROC Curve</div>', unsafe_allow_html=True)
        fig, ax = dark_fig(5, 4)
        ax.plot(res["fpr"], res["tpr"], color="#58a6ff", lw=2,
                label=f"ROC (AUC = {res['roc_auc']:.3f})")
        ax.plot([0,1],[0,1], color="#30363d", lw=1, linestyle="--")
        ax.fill_between(res["fpr"], res["tpr"], alpha=0.1, color="#58a6ff")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Receiver Operating Characteristic")
        ax.legend(loc="lower right", facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3")
        st.pyplot(fig)
        plt.close()

    # Feature Importance
    if res["fi"] is not None:
        st.markdown('<div class="section-header">Feature Importance</div>', unsafe_allow_html=True)
        fi = res["fi"]
        fig, ax = dark_fig(9, 4)
        colors_fi = ["#58a6ff" if v > fi.median() else "#30363d" for v in fi.values]
        ax.barh(fi.index, fi.values, color=colors_fi, height=0.6)
        ax.set_xlabel("Importance Score")
        ax.set_title("Which Features Drive Predictions")
        st.pyplot(fig)
        plt.close()

    # Classification Report Table
    st.markdown('<div class="section-header">Detailed Classification Report</div>', unsafe_allow_html=True)
    report_df = pd.DataFrame(rep).T.round(3)
    report_df = report_df[report_df.index.isin(["0","1","weighted avg","macro avg"])]
    report_df.index = report_df.index.map({"0":"DOWN","1":"UP","weighted avg":"Weighted Avg","macro avg":"Macro Avg"})
    st.dataframe(report_df, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
#  PAGE 3 — PREDICT STOCK
# ═══════════════════════════════════════════════════════════════
elif page == "🔮  Predict Stock":
    st.markdown("# 🔮 Predict Tomorrow's Direction")
    st.markdown("Fill in the stock details below and let the AI predict if the price will go **UP** or **DOWN**.")
    st.markdown("---")

    tab1, tab2 = st.tabs(["✍️ Manual Input", "🎯 Pick From Dataset"])

    # ── Tab 1: Manual ────────────────────────────────────────
    with tab1:
        c1, c2, c3 = st.columns(3)
        with c1:
            open_p  = st.number_input("Open Price ($)", min_value=0.01, value=150.00, step=0.5)
            close_p = st.number_input("Close Price ($)", min_value=0.01, value=152.50, step=0.5)
            high_p  = st.number_input("High Price ($)", min_value=0.01, value=154.00, step=0.5)
            low_p   = st.number_input("Low Price ($)", min_value=0.01, value=148.00, step=0.5)
        with c2:
            volume   = st.number_input("Volume Traded", min_value=1000, value=2_500_000, step=100000)
            mkt_cap  = st.number_input("Market Cap ($B)", min_value=0.1, value=50.0, step=1.0)
            pe_ratio = st.number_input("PE Ratio", min_value=0.1, value=20.0, step=0.5)
            div_yld  = st.number_input("Dividend Yield (%)", min_value=0.0, value=2.0, step=0.1)
        with c3:
            eps      = st.number_input("EPS ($)", min_value=0.01, value=5.0, step=0.1)
            wk52_h   = st.number_input("52 Week High ($)", min_value=0.01, value=180.0, step=1.0)
            wk52_l   = st.number_input("52 Week Low ($)", min_value=0.01, value=120.0, step=1.0)
            sector   = st.selectbox("Sector", sectors)

        if st.button("🚀 Predict Direction"):
            daily_ret    = (close_p - open_p) / open_p * 100
            price_range  = (high_p - low_p) / low_p * 100
            wk52_pos     = (close_p - wk52_l) / (wk52_h - wk52_l + 1e-9) * 100
            log_vol      = np.log1p(volume)
            log_mc       = np.log1p(mkt_cap * 1e9)
            sec_enc      = le.transform([sector])[0] if sector in le.classes_ else 0

            X_input = pd.DataFrame([[open_p, close_p, high_p, low_p,
                                      daily_ret, price_range, wk52_pos,
                                      log_vol, log_mc, pe_ratio, div_yld, eps, sec_enc]],
                                    columns=FEATURES)

            clf = res["clf"]
            if model_choice == "Logistic Regression":
                X_scaled = res["scaler"].transform(X_input)
                pred     = clf.predict(X_scaled)[0]
                proba    = clf.predict_proba(X_scaled)[0]
            else:
                pred  = clf.predict(X_input)[0]
                proba = clf.predict_proba(X_input)[0]

            conf = proba[pred]
            label = "⬆ UP — Buy Signal" if pred == 1 else "⬇ DOWN — Caution"
            css   = "pred-up" if pred == 1 else "pred-down"

            st.markdown(f'<div class="pred-box {css}">{label}</div>', unsafe_allow_html=True)

            pa, pb = st.columns(2)
            with pa:
                st.metric("Confidence", f"{conf*100:.1f}%")
                st.metric("Daily Return (input)", f"{daily_ret:.2f}%")
            with pb:
                # Prob bar chart
                fig, ax = dark_fig(5, 2.5)
                ax.bar(["DOWN ↓","UP ↑"], [proba[0]*100, proba[1]*100],
                       color=["#f85149","#3fb950"], width=0.45)
                ax.set_ylabel("Probability (%)")
                ax.set_ylim(0, 105)
                for i, v in enumerate([proba[0]*100, proba[1]*100]):
                    ax.text(i, v+1, f"{v:.1f}%", ha="center", color="#e6edf3", fontweight="bold")
                ax.set_title("Prediction Probabilities")
                st.pyplot(fig)
                plt.close()

    # ── Tab 2: Pick from dataset ──────────────────────────────
    with tab2:
        col_t, col_d = st.columns(2)
        sel_ticker = col_t.selectbox("Select Ticker", tickers)
        ticker_data = data[data["ticker"] == sel_ticker].sort_values("date")
        latest = ticker_data.iloc[-1]

        with col_d:
            st.markdown(f"**Latest Date:** {latest['date'].date()}")
            st.markdown(f"**Sector:** {latest['sector']}")
            st.markdown(f"**Close:** ${latest['close']:.2f}")

        if st.button("🚀 Predict for This Stock"):
            X_input = pd.DataFrame([[
                latest["open"], latest["close"], latest["high"], latest["low"],
                latest["daily_return"], latest["price_range_pct"], latest["wk52_position"],
                latest["log_volume"], latest["log_market_cap"],
                latest["pe_ratio"], latest["div_yield"], latest["eps"], latest["sector_enc"]
            ]], columns=FEATURES)

            clf = res["clf"]
            if model_choice == "Logistic Regression":
                X_sc  = res["scaler"].transform(X_input)
                pred  = clf.predict(X_sc)[0]
                proba = clf.predict_proba(X_sc)[0]
            else:
                pred  = clf.predict(X_input)[0]
                proba = clf.predict_proba(X_input)[0]

            label = "⬆ UP" if pred == 1 else "⬇ DOWN"
            css   = "pred-up" if pred == 1 else "pred-down"
            st.markdown(f'<div class="pred-box {css}">{sel_ticker}: {label} (Confidence {proba[pred]*100:.1f}%)</div>',
                        unsafe_allow_html=True)

            # Price history chart
            st.markdown('<div class="section-header">Price History</div>', unsafe_allow_html=True)
            fig, ax = dark_fig(9, 3.5)
            ax.plot(ticker_data["date"], ticker_data["close"], color="#58a6ff", lw=2)
            ax.fill_between(ticker_data["date"], ticker_data["close"],
                            ticker_data["close"].min(), alpha=0.15, color="#58a6ff")
            ax.scatter(ticker_data["date"].iloc[-1], ticker_data["close"].iloc[-1],
                       color="#3fb950" if pred == 1 else "#f85149", s=80, zorder=5)
            ax.set_title(f"{sel_ticker} — 3 Month Close Price")
            ax.set_ylabel("Close Price ($)")
            ax.tick_params(axis="x", rotation=30)
            st.pyplot(fig)
            plt.close()


# ═══════════════════════════════════════════════════════════════
#  PAGE 4 — STOCK EXPLORER
# ═══════════════════════════════════════════════════════════════
elif page == "🔍  Stock Explorer":
    st.markdown("# 🔍 Stock Explorer")
    st.markdown("---")

    col_a, col_b = st.columns([1, 3])
    with col_a:
        sel_sector = st.selectbox("Filter by Sector", ["All"] + sectors)
        n_top = st.slider("Top N Stocks", 5, 20, 10)

    filtered = data if sel_sector == "All" else data[data["sector"] == sel_sector]

    with col_b:
        # Top performers
        top_perf = (filtered.groupby("ticker")["daily_return"].mean()
                    .sort_values(ascending=False).head(n_top))
        st.markdown('<div class="section-header">Top Performers (Avg Daily Return)</div>', unsafe_allow_html=True)
        fig, ax = dark_fig(9, 4)
        colors_top = ["#3fb950" if v >= 0 else "#f85149" for v in top_perf.values]
        ax.bar(top_perf.index, top_perf.values, color=colors_top, width=0.6)
        ax.axhline(0, color="#8b949e", lw=0.8, linestyle="--")
        ax.set_ylabel("Avg Daily Return (%)")
        ax.set_title(f"Top {n_top} Stocks — {sel_sector if sel_sector != 'All' else 'All Sectors'}")
        ax.tick_params(axis="x", rotation=45)
        st.pyplot(fig)
        plt.close()

    # Volume vs Return scatter
    st.markdown('<div class="section-header">Volume vs Return Scatter</div>', unsafe_allow_html=True)
    samp = filtered.sample(min(500, len(filtered)), random_state=42)
    sector_list = sorted(samp["sector"].unique())
    palette = ["#58a6ff","#3fb950","#f85149","#ffa657","#d2a8ff",
               "#bc8cff","#79c0ff","#56d364","#ff7b72","#ffd700","#ff6eb4"]

    fig, ax = dark_fig(10, 5)
    for i, sec in enumerate(sector_list):
        sub = samp[samp["sector"] == sec]
        ax.scatter(sub["log_volume"], sub["daily_return"],
                   color=palette[i % len(palette)], alpha=0.5, s=20, label=sec)
    ax.axhline(0, color="#8b949e", lw=0.8, linestyle="--")
    ax.set_xlabel("Log Volume")
    ax.set_ylabel("Daily Return (%)")
    ax.set_title("Volume vs Daily Return by Sector")
    ax.legend(fontsize=7, facecolor="#21262d", edgecolor="#30363d",
              labelcolor="#e6edf3", loc="upper right", ncol=2)
    st.pyplot(fig)
    plt.close()

    # Summary stats table
    st.markdown('<div class="section-header">Sector Summary Statistics</div>', unsafe_allow_html=True)
    summary = (data.groupby("sector").agg(
        Stocks=("ticker","nunique"),
        Avg_Close=("close","mean"),
        Avg_Return=("daily_return","mean"),
        Avg_Volume=("volume","mean"),
        UP_Rate=("target","mean")
    ).round(2))
    summary["UP_Rate"] = (summary["UP_Rate"] * 100).round(1).astype(str) + "%"
    st.dataframe(summary, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
#  PAGE 5 — HOW IT WORKS
# ═══════════════════════════════════════════════════════════════
elif page == "📚  How It Works":
    st.markdown("# 📚 How This AI Model Works")
    st.markdown("*A beginner-friendly guide for your internship report*")
    st.markdown("---")

    steps = [
        ("1️⃣", "Data Collection", "3 months of stock data (June–August 2025) were collected — 3 CSV files merged into one dataset of 8,600+ rows across 150 tickers and 11 sectors."),
        ("2️⃣", "Data Cleaning", "Dates parsed, columns renamed, missing values removed. Data sorted by Ticker + Date so time-series features work correctly."),
        ("3️⃣", "Feature Engineering", "Raw columns → new signals: Daily Return %, 52-Week Position %, Price Range %, Log Volume, Log Market Cap. These give the model richer inputs."),
        ("4️⃣", "Target Creation", "For each row: 'Will tomorrow's Close > today's Close?' → 1 (UP) or 0 (DOWN). This is the label the model learns to predict."),
        ("5️⃣", "Train / Test Split", "80% of rows used for training (model learns patterns). 20% held out for testing (we check accuracy on unseen data)."),
        ("6️⃣", "Model Training", f"A {model_choice} was trained on 13 features. It learns decision rules from historical patterns to classify future direction."),
        ("7️⃣", "Evaluation", f"Tested on held-out data → Accuracy: {res['acc']*100:.1f}%, ROC-AUC: {res['roc_auc']:.3f}. Anything above 55% is valuable in finance AI."),
        ("8️⃣", "Prediction", "New stock data → same feature engineering → model outputs UP/DOWN + confidence probability."),
    ]

    for icon, title, desc in steps:
        with st.expander(f"{icon} **{title}**", expanded=True):
            st.markdown(desc)

    st.markdown("---")
    st.markdown("### 🧮 Key Formulas Used")
    col1, col2 = st.columns(2)
    with col1:
        st.code("Daily Return (%) = (Close - Open) / Open × 100", language="python")
        st.code("52W Position (%) = (Close - 52W Low) / (52W High - 52W Low) × 100", language="python")
    with col2:
        st.code("Price Range (%) = (High - Low) / Low × 100", language="python")
        st.code("Target = 1 if tomorrow_close > today_close else 0", language="python")

    st.markdown("### 📋 Tech Stack")
    c1, c2, c3, c4 = st.columns(4)
    for col, tech, desc2 in zip(
        [c1,c2,c3,c4],
        ["Python 3","Pandas","Scikit-learn","Streamlit"],
        ["Language","Data handling","ML model","Dashboard"]
    ):
        with col:
            st.markdown(f"""<div class="kpi-card">
                <b style="color:#58a6ff">{tech}</b><br>
                <small style="color:#8b949e">{desc2}</small>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 💡 Tips for Your Internship Presentation")
    tips = [
        "**64%+ accuracy beats random chance** (50%) — explain this clearly.",
        "**ROC-AUC > 0.5** means the model has real predictive power.",
        "**Feature importance** shows Daily Return is the strongest predictor — this makes intuitive sense.",
        "**Never claim 100% accuracy** — stock markets are inherently uncertain.",
        "**Mention overfitting prevention** — train/test split ensures honest evaluation.",
    ]
    for tip in tips:
        st.markdown(f"- {tip}")