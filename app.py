import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
import io

st.set_page_config(page_title="Sales & Demand Forecasting", page_icon="📈", layout="wide")

st.title("📈 Sales & Demand Forecasting")
st.markdown("**XGBoost + Linear Regression on Superstore Sales Dataset**")
st.markdown("---")

@st.cache_data
def load_and_train(file):
    df = pd.read_csv(file, encoding="latin1")
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df['Ship Date'] = pd.to_datetime(df['Ship Date'])

    monthly_sales = df.set_index('Order Date').resample('ME')['Sales'].sum().reset_index()
    monthly_sales.columns = ['Order Date', 'Sales']

    monthly_sales['Month'] = monthly_sales['Order Date'].dt.month
    monthly_sales['Quarter'] = monthly_sales['Order Date'].dt.quarter
    monthly_sales['IsHolidySeason'] = monthly_sales['Month'].isin([11, 12]).astype(int)
    monthly_sales['TimeIndex'] = range(len(monthly_sales))

    features = ['Month', 'Quarter', 'IsHolidySeason', 'TimeIndex']

    train = monthly_sales[monthly_sales['Order Date'].dt.year <= 2016]
    test  = monthly_sales[monthly_sales['Order Date'].dt.year == 2017]

    X_train, y_train = train[features], train['Sales']
    X_test,  y_test  = test[features],  test['Sales']

    # Baseline
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)

    # XGBoost tuned
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
    xgb.fit(X_train, y_train)
    xgb_preds = xgb.predict(X_test)

    mae_xgb  = mean_absolute_error(y_test, xgb_preds)
    rmse_xgb = np.sqrt(mean_squared_error(y_test, xgb_preds))
    mae_lr   = mean_absolute_error(y_test, lr_preds)
    rmse_lr  = np.sqrt(mean_squared_error(y_test, lr_preds))

    # 2018 forecast
    future_dates = pd.date_range(start='2018-01-31', periods=12, freq='ME')
    future_df = pd.DataFrame({'Order Date': future_dates})
    future_df['Month'] = future_df['Order Date'].dt.month
    future_df['Quarter'] = future_df['Order Date'].dt.quarter
    future_df['IsHolidySeason'] = future_df['Month'].isin([11, 12]).astype(int)
    future_df['TimeIndex'] = range(len(monthly_sales), len(monthly_sales) + 12)
    future_preds = xgb.predict(future_df[features])

    return {
        'monthly_sales': monthly_sales,
        'test': test,
        'xgb_preds': xgb_preds,
        'lr_preds': lr_preds,
        'future_df': future_df,
        'future_preds': future_preds,
        'mae_xgb': mae_xgb,
        'rmse_xgb': rmse_xgb,
        'mae_lr': mae_lr,
        'rmse_lr': rmse_lr,
        'xgb': xgb,
        'features': features,
        'y_test': test['Sales']
    }

# --- Sidebar ---
st.sidebar.header("📂 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload Superstore CSV", type=["csv"])

st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset:** Superstore Sales (Kaggle)")
st.sidebar.markdown("**Models:** XGBoost · Linear Regression")
st.sidebar.markdown("**Period:** 2014–2017 → Forecast 2018")

if uploaded_file:
    with st.spinner("Training models..."):
        data = load_and_train(uploaded_file)

    monthly_sales = data['monthly_sales']
    test          = data['test']
    xgb_preds     = data['xgb_preds']
    lr_preds      = data['lr_preds']
    future_df     = data['future_df']
    future_preds  = data['future_preds']
    mae_xgb       = data['mae_xgb']
    rmse_xgb      = data['rmse_xgb']
    mae_lr        = data['mae_lr']
    rmse_lr       = data['rmse_lr']
    xgb_model     = data['xgb']
    y_test        = data['y_test']

    # --- Metrics ---
    st.subheader("📊 Model Performance")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("XGBoost MAE",  f"${mae_xgb:,.0f}")
    col2.metric("XGBoost RMSE", f"${rmse_xgb:,.0f}")
    col3.metric("Baseline MAE",  f"${mae_lr:,.0f}", delta=f"-${mae_lr - mae_xgb:,.0f} vs XGB", delta_color="inverse")
    col4.metric("Baseline RMSE", f"${rmse_lr:,.0f}", delta=f"-${rmse_lr - rmse_xgb:,.0f} vs XGB", delta_color="inverse")

    st.markdown("---")

    # --- Main Forecast Chart ---
    st.subheader("📈 Sales Forecast — Historical + 2018 Prediction")
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(monthly_sales['Order Date'], monthly_sales['Sales'],
            label='Past Sales (2014–2017)', color='steelblue', linewidth=2)
    ax.plot(test['Order Date'], xgb_preds,
            label='Model Test (2017)', color='orange', linewidth=2, marker='o')
    ax.plot(future_df['Order Date'], future_preds,
            label='Predicted Sales (2018)', color='green', linewidth=2, linestyle='--', marker='o')
    ax.fill_between(future_df['Order Date'],
                    future_preds - mae_xgb, future_preds + mae_xgb,
                    alpha=0.15, color='green', label=f'Forecast Range (±${mae_xgb:,.0f})')
    ax.axvline(x=pd.Timestamp('2018-01-01'), color='red', linestyle=':', linewidth=1.5, label='Forecast Start')
    ax.set_title('Monthly Sales Forecast — Superstore (2014–2018)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Month / Year', fontsize=12)
    ax.set_ylabel('Total Monthly Sales (USD)', fontsize=12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")
    col_left, col_right = st.columns(2)

    # --- Actual vs Predicted 2017 ---
    with col_left:
        st.subheader("🔍 Actual vs Predicted (2017)")
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        ax2.plot(test['Order Date'], y_test.values, marker='o', label='Actual Sales', color='steelblue')
        ax2.plot(test['Order Date'], xgb_preds, marker='o', label='XGBoost', color='orange')
        ax2.plot(test['Order Date'], lr_preds, marker='o', linestyle='--', label='Linear Regression', color='gray')
        ax2.set_title('2017 — Actual vs Predicted')
        ax2.set_ylabel('Sales ($)')
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
        ax2.legend()
        ax2.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2)

    # --- Feature Importance ---
    with col_right:
        st.subheader("⚙️ Feature Importance (XGBoost)")
        feature_names = ['Month', 'Quarter', 'Holiday Season', 'Trend (TimeIndex)']
        importances = xgb_model.feature_importances_
        fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values('Importance')
        fig3, ax3 = plt.subplots(figsize=(7, 4))
        bars = ax3.barh(fi_df['Feature'], fi_df['Importance'], color='steelblue')
        ax3.set_title('What Drives the Forecast?')
        ax3.set_xlabel('Importance Score')
        ax3.bar_label(bars, fmt='%.3f', padding=3)
        ax3.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig3)

    st.markdown("---")

    # --- 2018 Forecast Table ---
    st.subheader("📅 2018 Monthly Forecast")
    forecast_table = pd.DataFrame({
        'Month': future_df['Order Date'].dt.strftime('%B %Y'),
        'Predicted Sales': [f"${v:,.0f}" for v in future_preds],
        'Lower Bound': [f"${v - mae_xgb:,.0f}" for v in future_preds],
        'Upper Bound': [f"${v + mae_xgb:,.0f}" for v in future_preds],
    })
    st.dataframe(forecast_table, use_container_width=True)

    st.markdown("---")

    # --- Business Summary ---
    st.subheader("💼 Business Summary")
    col_a, col_b, col_c = st.columns(3)
    col_a.info("📦 **Inventory**\nStock up before October for the Q4 sales surge.")
    col_b.warning("💰 **Cash Flow**\nExpect slower months in Jan–Feb. Plan expenses accordingly.")
    col_c.success("👥 **Staffing**\nHire seasonal staff in October before the holiday rush.")

else:
    st.info("👈 Upload your **Superstore Sales CSV** from the sidebar to get started.")
    st.markdown("""
    ### What this app does:
    - Loads and cleans the Superstore Sales dataset
    - Engineers time-based features (Month, Quarter, Holiday Season, Trend)
    - Trains **XGBoost** and **Linear Regression** models
    - Evaluates and compares both on 2017 test data
    - Forecasts **2018 monthly sales** with confidence bands
    - Shows feature importance and business recommendations
    """)
