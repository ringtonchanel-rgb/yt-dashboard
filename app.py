import streamlit as st
import pandas as pd
import plotly.express as px

# Заголовок
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

metrics_options = {
    "Просмотры": "Views",
    "CTR": "Impressions click-through rate",
    "AVD (средняя продолжительность просмотра)": "Average view duration",
    "Длительность видео": "Duration",
    "Доход": "Estimated revenue",
    "Подписчики": "Subscribers"
}
selected_metrics = st.sidebar.multiselect("Выберите метрики:", list(metrics_options.keys()), default=["Просмотры"])

# Если файл загружен
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Универсальные названия колонок
    if "Название видео" in df.columns:
        title_col = "Название видео"
    elif "Title" in df.columns:
        title_col = "Title"
    else:
        title_col = None

    if "Контент" in df.columns:
        id_col = "Контент"
    elif "Video ID" in df.columns:
        id_col = "Video ID"
    else:
        id_col = None

    # Оставляем только последние N видео
    df = df.tail(num_videos)

    # Если ID есть — делаем кликабельные ссылки
    if id_col:
        df["YouTube Link"] = df[id_col].apply(lambda x: f"https://www.youtube.com/watch?v={x}")

    # Визуализация выбранных метрик
    for metric in selected_metrics:
        col_name = metrics_options[metric]
        if col_name in df.columns:
            st.subheader(f"{metric} по видео")
            fig = px.bar(
                df,
                x=title_col if title_col else id_col,
                y=col_name,
                text=col_name,
                hover_data=[id_col] if id_col else None,
                title=metric
            )
            fig.update_traces(texttemplate='%{text}', textposition='outside')
            fig.update_layout(xaxis_tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)

    # Таблица с метриками
    st.subheader("📋 Таблица всех метрик")
    show_cols = []
    if title_col: show_cols.append(title_col)
    if id_col: show_cols.append(id_col)
    if "YouTube Link" in df.columns: show_cols.append("YouTube Link")
    for m in selected_metrics:
        if metrics_options[m] in df.columns:
            show_cols.append(metrics_options[m])

    st.dataframe(df[show_cols])
else:
    st.info("👆 Загрузите CSV-файл, чтобы увидеть аналитику")
