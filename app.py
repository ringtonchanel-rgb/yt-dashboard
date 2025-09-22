import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Настройки страницы
st.set_page_config(page_title="YouTube Dashboard", layout="wide")

st.title("📊 YouTube Analytics Dashboard")

# Загрузка CSV
uploaded_file = st.file_uploader("Загрузите CSV из YouTube Studio", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Убираем строку "Итоговое значение"
    if "Название видео" in df.columns:
        df = df[df["Название видео"].notna()]

    # Приведение типов
    df["Просмотры"] = pd.to_numeric(df["Просмотры"], errors="coerce")
    df["Показы"] = pd.to_numeric(df["Показы"], errors="coerce")
    df["CTR для значков видео (%)"] = pd.to_numeric(df["CTR для значков видео (%)"], errors="coerce")
    df["Расчетный доход (USD)"] = pd.to_numeric(df["Расчетный доход (USD)"], errors="coerce")

    # Дата публикации
    if "Время публикации видео" in df.columns:
        df["Время публикации видео"] = pd.to_datetime(df["Время публикации видео"], errors="coerce")
        df = df.sort_values("Время публикации видео", ascending=False)

    # Фильтр: количество последних видео
    n = st.slider("Сколько последних видео показать:", 3, 15, 8)
    subset = df.head(n)

    # Таблица
    st.subheader("📌 Сводная таблица")
    st.dataframe(subset[[
        "Название видео", "Время публикации видео", "Просмотры", "Показы",
        "CTR для значков видео (%)", "Средняя продолжительность просмотра",
        "Расчетный доход (USD)"
    ]])

    # График: Просмотры и CTR
    st.subheader("📈 Просмотры и CTR")
    fig, ax1 = plt.subplots(figsize=(10,5))
    ax2 = ax1.twinx()

    ax1.bar(subset["Название видео"], subset["Просмотры"], color="skyblue", label="Просмотры")
    ax2.plot(subset["Название видео"], subset["CTR для значков видео (%)"], color="red", marker="o", label="CTR (%)")

    ax1.set_ylabel("Просмотры")
    ax2.set_ylabel("CTR (%)")
    ax1.set_xticklabels(subset["Название видео"], rotation=20, ha="right")
    ax1.set_title("Сравнение: Просмотры и CTR")
    fig.tight_layout()
    st.pyplot(fig)

    # График: Доход
    st.subheader("💰 Доход по видео")
    fig2, ax = plt.subplots(figsize=(10,5))
    ax.bar(subset["Название видео"], subset["Расчетный доход (USD)"], color="green")
    ax.set_ylabel("USD")
    ax.set_title("Расчетный доход по видео")
    ax.set_xticklabels(subset["Название видео"], rotation=20, ha="right")
    st.pyplot(fig2)

else:
    st.info("👆 Загрузите CSV, выгруженный из YouTube Studio, чтобы построить дашборд.")

