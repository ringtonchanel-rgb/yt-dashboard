import streamlit as st
import pandas as pd
import plotly.express as px

# ---------- НАСТРОЙКИ ----------
st.set_page_config(page_title="YouTube Dashboard 🚀", layout="wide")

st.sidebar.header("⚙️ Настройки")
uploaded_file = st.sidebar.file_uploader("Загрузите CSV из YouTube Studio", type=["csv"])
num_videos = st.sidebar.slider("Сколько последних видео показывать:", 3, 30, 7)

# Метрики для отображения
metrics_options = {
    "👁️ Просмотры": "Views",
    "📈 CTR": "Impressions click-through rate",
    "⏱️ AVD (средняя продолжительность просмотра)": "Average view duration",
    "🎬 Длительность видео": "Duration",
    "📊 Подписчики": "Subscribers",
    "💰 Доход": "Estimated partner revenue"
}
selected_metrics = st.sidebar.multiselect("Выберите метрики:", list(metrics_options.keys()), default=["👁️ Просмотры"])

# ---------- ОСНОВНОЙ КОНТЕНТ ----------
st.markdown("<h1 style='text-align: center;'>📊 YouTube Dashboard 🚀</h1>", unsafe_allow_html=True)
st.write("Аналитика YouTube-канала: просмотры, CTR, удержание, доход и другие ключевые метрики.")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Попробуем угадать основные колонки (под разные CSV)
    rename_map = {
        "Video ID": "Video ID",
        "External Video ID": "Video ID",
        "Название видео": "Title",
        "Video title": "Title",
        "Название": "Title",
        "Impressions click-through rate": "Impressions click-through rate",
        "CTR": "Impressions click-through rate",
        "Average view duration": "Average view duration",
        "Average Percentage Viewed": "Average Percentage Viewed",
        "Views": "Views",
        "Estimated partner revenue": "Estimated partner revenue",
        "Subscribers": "Subscribers",
        "Duration": "Duration"
    }
    df = df.rename(columns=rename_map)

    # Оставляем только последние N видео
    if "Video publish time" in df.columns:
        df = df.sort_values("Video publish time", ascending=False)
    df = df.head(num_videos)

    # Создаем кликабельные ссылки
    if "Video ID" in df.columns:
        df["YouTube Link"] = df["Video ID"].apply(lambda x: f"[🔗 Открыть](https://www.youtube.com/watch?v={x})")

    # ---- ВИЗУАЛИЗАЦИЯ ----
    st.subheader("📋 Таблица всех метрик")
    st.dataframe(df[["Title", "Video ID", "YouTube Link"] + list(metrics_options.values()) if "Title" in df.columns else df])

    # ---- ГРАФИКИ ----
    if selected_metrics:
        st.subheader("📊 Визуализация выбранных метрик")
        for metric in selected_metrics:
            col = metrics_options[metric]
            if col in df.columns:
                fig = px.line(
                    df,
                    x="Title",
                    y=col,
                    markers=True,
                    text=col,
                    title=f"{metric} по видео"
                )
                fig.update_traces(textposition="top center")
                fig.update_layout(xaxis_tickangle=-30)
                st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👆 Загрузите CSV-файл, чтобы увидеть аналитику")
