import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

st.set_page_config(page_title="Weather & Electricity Price Analysis", layout="wide")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("fmi_weather_and_price.csv")
    
    # Rename columns to match expected names
    df.rename(columns={"Time": "timestamp", "Wind": "wind_speed", "Temp": "temperature", "Price": "price"}, inplace=True)
    
    # Convert timestamp column to datetime format
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Extract additional time-related features
    df["hour"] = df["timestamp"].dt.hour
    df["date_only"] = df["timestamp"].dt.date
    
    return df

df = load_data()





df = load_data()

st.title("🌤️ Weather Conditions and Nordpool Spot Electricity Prices in Finland")

# Page layout
tab1, tab2 = st.tabs(["📊 Overview & Visuals", "📈 Correlation & Regression"])

with tab1:
    st.subheader("Descriptive Overview")
    st.write(df.describe())

    st.subheader("Line Plot: Price, Temperature, Wind Speed Over Time")
    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(df['timestamp'], df['price'], label="Price (€/MWh)", color="blue")
    ax2 = ax1.twinx()
    ax2.plot(df['timestamp'], df['temperature'], label="Temp (°C)", color="red", alpha=0.4)
    ax2.plot(df['timestamp'], df['wind_speed'], label="Wind (m/s)", color="green", alpha=0.4)
    ax1.set_ylabel("Price")
    ax2.set_ylabel("Weather")
    ax1.set_xlabel("Timestamp")
    fig.legend(loc="upper right")
    st.pyplot(fig)

    st.subheader("Distributions")
    col1, col2, col3 = st.columns(3)
    with col1:
        sns.histplot(df['price'], kde=True)
        plt.title("Price Distribution")
        st.pyplot(plt.gcf())
        plt.clf()
    with col2:
        sns.histplot(df['temperature'], kde=True, color='red')
        plt.title("Temperature Distribution")
        st.pyplot(plt.gcf())
        plt.clf()
    with col3:
        sns.histplot(df['wind_speed'], kde=True, color='green')
        plt.title("Wind Speed Distribution")
        st.pyplot(plt.gcf())
        plt.clf()

with tab2:
    st.subheader("Correlation Matrix")
    corr = df[['price', 'temperature', 'wind_speed']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    st.pyplot(plt.gcf())
    plt.clf()

    st.subheader("Regression Model: Predict Price from Weather")
    X = df[['temperature', 'wind_speed']]
    y = df['price']
    
    model = LinearRegression()
    model.fit(X, y)
    df['predicted_price'] = model.predict(X)

    r2 = r2_score(y, df['predicted_price'])
    rmse = mean_squared_error(y, df['predicted_price'], squared=False)

    st.markdown(f"**Model Performance**  \nR²: `{r2:.3f}`  \nRMSE: `{rmse:.2f}`")

    fig, ax = plt.subplots()
    ax.scatter(y, df['predicted_price'], alpha=0.3)
    ax.set_xlabel("Actual Price")
    ax.set_ylabel("Predicted Price")
    ax.set_title("Actual vs Predicted Electricity Prices")
    st.pyplot(fig)
