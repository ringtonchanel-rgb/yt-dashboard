import streamlit as st
import pandas as pd
import plotly.express as px

# --- Настройка страницы ---
st.set_page_config(page_title="YouTube Dashboard 🚀", layout="wide")

st.title("📊 YouTube Dashboard 🚀")
st.markdown("Аналитика YouTube-канала: просмотры, CTR, удержание, доход и другие ключевые метрики.")

# --- Боковое меню ---
st.sidebar.header("⚙️ Настройки")
uploaded_file = st.sidebar.file_uploader("Загрузите CSV из YouTube Studio", type=["csv"])

n_videos = st.sidebar.slider("Сколько последних видео показывать:", 3, 50, 10)

# Возможные метрики
metrics_options = {
    "Просмотры": ["Views", "Просмотры"],
    "CTR": ["Impressions click-through rate", "CTR", "Impressions click-through rate (%)"],
    "AVD (средняя продолжительность просмотра)": ["Average view duration", "Средняя продолжительность просмотра"],
    "Длительность видео": ["Duration", "Длительность"],
    "Доход": ["Estimated partner revenue", "Расчетный доход"],
    "Подписчики": ["Subscribers", "Подписчики"],
    "Показы": ["Impressions", "Показы"],
    "RPM (доход за 1000 просмотров)": ["RPM", "Доход за 1000 показов"]
}

selected_metrics = st.sidebar.multiselect("Выберите метрики:", list(metrics_options.keys()))
show_top = st.sidebar.checkbox("Показать ТОП-5 видео", value=True)
show_scatter = st.sidebar.checkbox("Scatter-графики (сравнение метрик)", value=True)

# --- Загрузка данных ---
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = [c.strip() for c in df.columns]

    # Определяем название и ID
    title_col = next((c for c in ["Название видео", "Title", "Video title"] if c in df.columns), None)
    id_col = next((c for c in ["Video ID", "ID видео", "Content"] if c in df.columns), None)

    # Сортируем по дате (если есть)
    if "Время публикации видео" in df.columns:
        df["Время публикации видео"] = pd.to_datetime(df["Время публикации видео"], errors="coerce")
        df = df.sort_values("Время публикации видео", ascending=False)

    df = df.head(n_videos)

    # Добавляем ссылки
    if id_col:
        df["YouTube Link"] = df[id_col].apply(lambda x: f"https://www.youtube.com/watch?v={x}")

    # Функция для поиска колонки (гибкий поиск)
    def find_col(possible_names):
        for name in possible_names:
            for col in df.columns:
                if name.lower() in col.lower():
                    return col
        return None

    # Таблица
    st.subheader("📋 Таблица всех метрик")
    base_cols = [c for c in [title_col, id_col, "YouTube Link"] if c]
    metric_cols = []
    for metric in selected_metrics:
        col_name = find_col(metrics_options[metric])
        if col_name:
            metric_cols.append(col_name)

    if base_cols + metric_cols:
        st.dataframe(df[base_cols + metric_cols])

    # Графики для выбранных метрик
    for metric in selected_metrics:
        col_name = find_col(metrics_options[metric])
        if col_name:
            st.subheader(f"{metric} по видео")
            fig = px.bar(
                df,
                x=title_col if title_col else id_col,
                y=col_name,
                text=col_name,
                hover_data=[id_col] if id_col else None
            )
            fig.update_traces(texttemplate='%{text}', textposition='outside')
            fig.update_layout(xaxis_tickangle=-45, height=500)
            st.plotly_chart(fig, use_container_width=True)

    # ТОП-5
    if show_top and selected_metrics:
        st.subheader("🏆 ТОП-5 видео по выбранным метрикам")
        col_name = find_col(metrics_options[selected_metrics[0]])
        if col_name:
            top5 = df.sort_values(col_name, ascending=False).head(5)
            st.table(top5[[title_col, col_name, "YouTube Link"]] if title_col else top5[[id_col, col_name, "YouTube Link"]])

    # Scatter
    if show_scatter and len(selected_metrics) >= 2:
        st.subheader("🔗 Сравнение метрик (Scatter)")
        col_x = find_col(metrics_options[selected_metrics[0]])
        col_y = find_col(metrics_options[selected_metrics[1]])
        if col_x and col_y:
            fig = px.scatter(
                df,
                x=col_x,
                y=col_y,
                size=find_col(metrics_options["Просмотры"]),
                color=title_col if title_col else id_col,
                hover_data=[id_col] if id_col else None
            )
            st.plotly_chart(fig, use_container_width=True)
