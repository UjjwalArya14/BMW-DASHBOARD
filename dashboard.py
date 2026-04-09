import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression

# ----------------------------------------
# PAGE SETTINGS
# ----------------------------------------
st.set_page_config(page_title="BMW Sales Dashboard", layout="wide")
st.title("🚗 BMW Sales Dashboard")
st.write("Upload your BMW sales CSV and explore insights with filters, KPIs, graphs & forecasting.")

# ----------------------------------------
# FILE UPLOADER
# ----------------------------------------
uploaded_file = st.file_uploader("📂 Upload BMW Sales CSV File", type=["csv"])

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)

    # Clean data
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
    df.dropna(subset=["Year", "Sales"], inplace=True)

    st.success("✅ File uploaded successfully!")

    # ----------------------------------------
    # SIDEBAR FILTERS
    # ----------------------------------------
    st.sidebar.header("🔎 Filters")

    min_year, max_year = int(df["Year"].min()), int(df["Year"].max())
    year_range = st.sidebar.slider(
        "Select Year Range", min_year, max_year, (min_year, max_year)
    )

    model_filter = "All"
    if "Model" in df.columns:
        models = ["All"] + sorted(df["Model"].dropna().unique().tolist())
        model_filter = st.sidebar.selectbox("Select Model", models)

    sales_min, sales_max = int(df["Sales"].min()), int(df["Sales"].max())
    sales_range = st.sidebar.slider(
        "Filter by Sales Range", sales_min, sales_max, (sales_min, sales_max)
    )

    # ----------------------------------------
    # APPLY FILTERS
    # ----------------------------------------
    filtered_df = df[
        (df["Year"] >= year_range[0]) &
        (df["Year"] <= year_range[1]) &
        (df["Sales"] >= sales_range[0]) &
        (df["Sales"] <= sales_range[1])
    ]

    if model_filter != "All" and "Model" in df.columns:
        filtered_df = filtered_df[filtered_df["Model"] == model_filter]

    st.subheader("📄 Filtered Data Preview")
    st.dataframe(filtered_df.head())

    # ----------------------------------------
    # KPI CARDS
    # ----------------------------------------
    st.subheader("📌 Key Performance Indicators")

    if len(filtered_df) > 0:
        filtered_df = filtered_df.sort_values("Year")

        total_sales = int(filtered_df["Sales"].sum())
        filtered_df["YoY_Growth_%"] = filtered_df["Sales"].pct_change() * 100
        avg_growth = filtered_df["YoY_Growth_%"].mean()

        best_row = filtered_df.loc[filtered_df["Sales"].idxmax()]
        best_year = int(best_row["Year"])
        best_sales = int(best_row["Sales"])
    else:
        total_sales = 0
        avg_growth = 0
        best_year = "N/A"
        best_sales = 0

    col1, col2, col3 = st.columns(3)

    col1.metric("💰 Total Sales", f"{total_sales:,}")
    col2.metric("📊 Avg Growth (%)", f"{avg_growth:.2f}%" if not np.isnan(avg_growth) else "0%")
    col3.metric("🏆 Best Year", f"{best_year} ({best_sales:,})")

    # ----------------------------------------
    # SALES TREND
    # ----------------------------------------
    st.subheader("📈 Sales Trend")

    if len(filtered_df) > 0:
        fig1 = px.line(filtered_df, x="Year", y="Sales", markers=True)
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.warning("No data available for selected filters.")

    # ----------------------------------------
    # YOY GROWTH GRAPH
    # ----------------------------------------
    st.subheader("📊 Year-over-Year Growth (%)")

    if len(filtered_df) > 1:
        fig2 = px.bar(filtered_df, x="Year", y="YoY_Growth_%")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("Not enough data to calculate growth.")

    # ----------------------------------------
    # FORECASTING MODEL
    # ----------------------------------------
    st.subheader("🔮 Sales Forecast (Next 5 Years)")

    if len(df) > 1:
        model = LinearRegression()
        X = df[["Year"]]
        y = df[["Sales"]]
        model.fit(X, y)

        future_years = np.arange(max_year + 1, max_year + 6).reshape(-1, 1)
        future_sales = model.predict(future_years)

        forecast_df = pd.DataFrame({
            "Year": future_years.flatten(),
            "Predicted_Sales": future_sales.flatten()
        })

        fig3 = px.line(forecast_df, x="Year", y="Predicted_Sales", markers=True)
        st.plotly_chart(fig3, use_container_width=True)
        st.dataframe(forecast_df)
    else:
        st.warning("Not enough data for forecasting.")

    # ----------------------------------------
    # TOP MODELS
    # ----------------------------------------
    if "Model" in df.columns and len(filtered_df) > 0:
        st.subheader("🏆 Top BMW Models")

        top_models = (
            filtered_df.groupby("Model")["Sales"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )

        fig4 = px.bar(top_models)
        st.plotly_chart(fig4, use_container_width=True)

    # ----------------------------------------
    # SEARCH MODEL
    # ----------------------------------------
    if "Model" in df.columns:
        st.subheader("🔍 Search BMW Model")

        search = st.text_input("Enter model name:")
        if search:
            result = df[df["Model"].str.contains(search, case=False, na=False)]

            if result.empty:
                st.warning("Model not found.")
            else:
                st.success("Model found!")
                st.dataframe(result)

                fig5 = px.line(result, x="Year", y="Sales", markers=True)
                st.plotly_chart(fig5, use_container_width=True)

else:
    st.info("📥 Please upload a CSV file to start.")