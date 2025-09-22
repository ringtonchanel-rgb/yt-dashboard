import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="YouTube Dashboard", layout="wide")

st.title("📊 YouTube Analytics Dashboard")

# Загрузка CSV
uploaded_file = st.file_uploader("Загрузите CSV из YouTube Studio", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Убираем строку "Итоговое значение"
    if "Название видео" in df.columns:
        df = df[df["Название видео"].notna()]

    # Преобразуем дату публикации
    if "Время публикации видео" in df.columns:
        df["Время публикации видео"] = pd.to_datetime(df["Время публикации видео"], errors="coerce")
        df = df.sort_values("Время публикации видео", ascending=False)

    # Сколько последних видео показывать
    n = st.slider("Сколько последних видео показать:", 3, 20, 8)
    subset = df.head(n).copy()

    # Отображаем таблицу
    st.subheader("📋 Последние видео")
    st.dataframe(subset)

    # Выбор метрик
    available_metrics = ["Просмотры", "Показы", "CTR для значков видео (%)", "Средняя продолжительность просмотра"]
    selected_metrics = st.multiselect("Выберите метрики для анализа", available_metrics, default=["Просмотры", "CTR для значков видео (%)"])

    if selected_metrics:
        st.subheader("📈 Сравнение метрик")
        fig = px.line(subset, x="Название видео", y=selected_metrics, markers=True)
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👆 Загрузите CSV-файл, чтобы увидеть аналитику")
