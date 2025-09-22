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

# Метрики, которые пользователь может включать
metrics_options = {
    "Просмотры": "Views",
    "CTR": "Impressions click-through rate",
    "AVD (средняя продолжительность просмотра)": "Average view duration",
    "Длительность видео": "Duration",
    "Доход": "Estimated partner revenue",
    "Подписчики": "Subscribers",
    "Показы": "Impressions",
    "RPM (доход за 1000 просмотров)": "RPM"
}
selected_metrics = st.sidebar.multiselect("Выберите метрики:", list(metrics_options.keys()))

show_top = st.sidebar.checkbox("Показать ТОП-5 видео", value=True)
show_scatter = st.sidebar.checkbox("Scatter-графики (сравнение метрик)", value=True)

# --- Загрузка и обработка данных ---
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Унифицируем названия столбцов (чтобы работало на разных выгрузках)
    df.columns = [c.strip() for c in df.columns]

    # Определяем ключевые поля (гибко, через if)
    title_col = None
    for col in ["Название видео", "Title", "Video title"]:
        if col in df.columns:
            title_col = col
            break

    id_col = None
    for col in ["Video ID", "ID видео", "Content"]:
        if col in df.columns:
            id_col = col
            break

    # Ограничение по числу последних видео
    if "Время публикации видео" in df.columns:
        df["Время публикации видео"] = pd.to_datetime(df["Время публикации видео"], errors="coerce")
        df = df.sort_values("Время публикации видео", ascending=False)
    df = df.head(n_videos)

    # Добавляем кликабельные ссылки
    if id_col:
        df["YouTube Link"] = df[id_col].apply(lambda x: f"https://www.youtube.com/watch?v={x}")

    st.subheader("📋 Таблица всех метрик")
    base_cols = []
    if title_col: base_cols.append(title_col)
    if id_col: base_cols.append(id_col)
    if "YouTube Link" in df.columns: base_cols.append("YouTube Link")

    st.dataframe(df[base_cols + [metrics_options[m] for m in selected_metrics if metrics_options[m] in df.columns]])

    # --- Визуализация выбранных метрик ---
    for metric in selected_metrics:
        col_name = metrics_options[metric]
        if col_name in df.columns:
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

    # --- ТОП-5 видео ---
    if show_top and selected_metrics:
        st.subheader("🏆 ТОП-5 видео по выбранным метрикам")
        metric = metrics_options[selected_metrics[0]]  # первая выбранная метрика
        if metric in df.columns:
            top5 = df.sort_values(metric, ascending=False).head(5)
            st.table(top5[[title_col, metric, "YouTube Link"]] if title_col else top5[[id_col, metric, "YouTube Link"]])

    # --- Scatter-плоты для анализа связей ---
    if show_scatter and len(selected_metrics) >= 2:
        st.subheader("🔗 Сравнение метрик (Scatter)")
        metric_x = metrics_options[selected_metrics[0]]
        metric_y = metrics_options[selected_metrics[1]]
        if metric_x in df.columns and metric_y in df.columns:
            fig = px.scatter(
                df,
                x=metric_x,
                y=metric_y,
                size=metrics_options["Просмотры"] if "Просмотры" in selected_metrics and metrics_options["Просмотры"] in df.columns else None,
                color=title_col if title_col else id_col,
                hover_data=[id_col] if id_col else None
            )
            st.plotly_chart(fig, use_container_width=True)

    # --- Сравнение с медианой ---
    st.subheader("📈 Сравнение с медианой канала")
    for metric in selected_metrics:
        col_name = metrics_options[metric]
        if col_name in df.columns:
            median_val = df[col_name].median()
            st.markdown(f"**{metric} (медиана по выборке):** {median_val:.2f}")
