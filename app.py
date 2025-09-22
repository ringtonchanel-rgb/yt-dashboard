import streamlit as st
import pandas as pd
import plotly.express as px

# Настройка страницы
st.set_page_config(page_title="YouTube Dashboard", layout="wide")
st.markdown(
    "<h1 style='text-align: center;'>📊 YouTube Dashboard 🚀</h1>",
    unsafe_allow_html=True
)
st.write("Аналитика YouTube-канала: просмотры, CTR, удержание, доход и другие ключевые метрики")

# Боковое меню
st.sidebar.header("⚙️ Настройки")
uploaded_file = st.sidebar.file_uploader("Загрузите CSV из YouTube Studio", type="csv")

num_videos = st.sidebar.slider("Сколько последних видео показывать:", 3, 30, 7)

# Возможные метрики
metrics_options = {
    "Просмотры": ["Views", "Просмотры"],
    "CTR": ["Impressions click-through rate", "CTR"],
    "AVD (средняя продолжительность просмотра)": ["Average view duration", "AVD"],
    "Длительность видео": ["Duration", "Длительность видео"],
    "Доход": ["Estimated revenue", "Revenue", "Доход"],
    "Подписчики": ["Subscribers", "Подписчики"]
}

selected_metrics = st.sidebar.multiselect("Выберите метрики:", list(metrics_options.keys()), default=["Просмотры"])

# Обработка CSV
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Определяем название и ID
    title_col = next((c for c in ["Название видео", "Title"] if c in df.columns), None)
    id_col = next((c for c in ["Контент", "Video ID"] if c in df.columns), None)

    # Ограничение по последним видео
    df = df.tail(num_videos)

    # Делаем ссылку на YouTube
    if id_col:
        df["YouTube Link"] = df[id_col].apply(lambda x: f"https://www.youtube.com/watch?v={x}")

    # Визуализация метрик
    for metric in selected_metrics:
        found_col = next((c for c in metrics_options[metric] if c in df.columns), None)
        if found_col:
            st.subheader(f"{metric} по видео")
            fig = px.bar(
                df,
                x=title_col if title_col else id_col,
                y=found_col,
                text=found_col,
                hover_data=[id_col] if id_col else None,
                title=metric
            )
            fig.update_traces(texttemplate='%{text}', textposition='outside')
            fig.update_layout(xaxis_tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"⚠️ Метрика «{metric}» не найдена в CSV")

    # Таблица
    st.subheader("📋 Таблица всех метрик")
    show_cols = []
    if title_col: show_cols.append(title_col)
    if id_col: show_cols.append(id_col)
    if "YouTube Link" in df.columns: show_cols.append("YouTube Link")

    for m in selected_metrics:
        found_col = next((c for c in metrics_options[m] if c in df.columns), None)
        if found_col:
            show_cols.append(found_col)

    st.dataframe(df[show_cols])
else:
    st.info("👆 Загрузите CSV-файл, чтобы увидеть аналитику")
