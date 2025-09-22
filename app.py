import streamlit as st
import pandas as pd
import plotly.express as px

# ======================
# НАСТРОЙКИ СТРАНИЦЫ
# ======================
st.set_page_config(page_title="YouTube Dashboard 🚀", layout="wide")

st.title("📊 YouTube Dashboard 🚀")
st.markdown("Аналитика YouTube-канала: просмотры, CTR, удержание, доход и другие метрики")

# ======================
# БОКОВАЯ ПАНЕЛЬ
# ======================
st.sidebar.header("⚙️ Настройки")

uploaded_file = st.sidebar.file_uploader("Загрузите CSV из YouTube Studio", type=["csv"])

n_videos = st.sidebar.slider("Сколько последних видео показывать:", 3, 50, 10)

# ======================
# ЗАГРУЗКА ДАННЫХ
# ======================
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Очищаем от пустых строк
    df = df.dropna(subset=["Название видео"])
    
    # Дата публикации
    if "Время публикации видео" in df.columns:
        df["Время публикации видео"] = pd.to_datetime(df["Время публикации видео"], errors="coerce")
        df = df.sort_values("Время публикации видео", ascending=False)

    df = df.head(n_videos)

    # ======================
    # ВЫБОР МЕТРИК
    # ======================
    st.sidebar.subheader("📌 Выберите метрики для анализа")

    metrics_blocks = {
        "Общее": ["Просмотры", "Подписчики", "Уникальные зрители", "Время просмотра (часы)"],
        "Удержание": ["Средняя продолжительность просмотра", "Средний % просмотра", 
                      "Продолжили смотреть (%)", "Заинтересованные просмотры",
                      "Клики по элементам конечной заставки", "CTR конечных заставок"],
        "Трафик": ["Показы", "CTR для значков видео (%)", "Цена за тысячу показов (CPM)", 
                   "Монетизируемые воспроизведения"],
        "Доход": ["Расчетный доход (USD)", "Доход на тысячу просмотров (RPM)", 
                  "Доход от рекламы на YouTube (USD)", "YouTube Premium (USD)", 
                  "Доход от транзакций"]
    }

    selected_metrics = []
    for block, metrics in metrics_blocks.items():
        with st.sidebar.expander(block, expanded=True):
            for metric in metrics:
                if metric in df.columns:
                    if st.checkbox(metric, True):
                        selected_metrics.append(metric)

    # ======================
    # ВЫВОД МЕТРИК
    # ======================
    st.subheader("📈 Таблица по выбранным метрикам")
    st.dataframe(df[["Название видео", "Время публикации видео"] + selected_metrics])

    # ======================
    # ГРАФИКИ
    # ======================
    st.subheader("📊 Визуализация метрик")

    for metric in selected_metrics:
        if metric in df.columns:
            fig = px.bar(df, x="Название видео", y=metric, 
                         title=f"{metric} по видео", 
                         text_auto=True)
            st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👆 Загрузите CSV-файл, чтобы увидеть аналитику")
