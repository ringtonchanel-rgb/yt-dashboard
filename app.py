import streamlit as st
import pandas as pd
import plotly.express as px

# Настройки страницы
st.set_page_config(page_title="YouTube Dashboard 🚀", layout="wide")

# Заголовок
st.title("📊 YouTube Dashboard 🚀")
st.markdown("Аналитика YouTube-канала: просмотры, CTR, удержание и другие ключевые метрики")

# Боковая панель
st.sidebar.header("⚙️ Настройки")
uploaded_file = st.sidebar.file_uploader("Загрузите CSV из YouTube Studio", type=["csv"])
n = st.sidebar.slider("Сколько последних видео показывать:", 3, 20, 8)

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Убираем пустые строки
    if "Название видео" in df.columns:
        df = df[df["Название видео"].notna()]

    # Преобразуем дату
    if "Время публикации видео" in df.columns:
        df["Время публикации видео"] = pd.to_datetime(df["Время публикации видео"], errors="coerce")
        df = df.sort_values("Время публикации видео", ascending=False)

    # Берём последние n видео
    subset = df.head(n).copy()

    # --- Карточки с метриками ---
    st.subheader("✨ Основные показатели")
    col1, col2, col3, col4 = st.columns(4)

    def safe_metric(col, label, column):
        try:
            value = subset[column].iloc[0]
            col.metric(label, f"{value:,}")
        except:
            col.metric(label, "—")

    safe_metric(col1, "Просмотры", "Просмотры")
    safe_metric(col2, "Показы", "Показы")
    safe_metric(col3, "CTR (%)", "CTR для значков видео (%)")
    safe_metric(col4, "Сред. время просмотра", "Средняя продолжительность просмотра")

    # --- Таблица ---
    st.subheader("📋 Последние видео")
    st.dataframe(subset)

    # --- Графики ---
    st.subheader("📈 Сравнение метрик")
    available_metrics = ["Просмотры", "Показы", "CTR для значков видео (%)", "Средняя продолжительность просмотра"]
    selected_metrics = st.multiselect("Выберите метрики для анализа", available_metrics, default=["Просмотры", "CTR для значков видео (%)"])

    if selected_metrics:
        fig = px.bar(
            subset,
            x="Название видео",
            y=selected_metrics,
            barmode="group",
            title="Сравнение по выбранным метрикам",
            text_auto=True
        )
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👆 Загрузите CSV-файл, чтобы увидеть аналитику")
