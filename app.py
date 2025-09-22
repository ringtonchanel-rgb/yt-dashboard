import streamlit as st
import pandas as pd
import plotly.express as px

# --- Конфигурация страницы ---
st.set_page_config(page_title="YouTube Dashboard 🚀", layout="wide")

# --- Заголовок ---
st.markdown(
    """
    <h1 style="display: flex; align-items: center;">
        📊 YouTube Dashboard 🚀
    </h1>
    <p>Аналитика YouTube-канала: просмотры, CTR, удержание и другие ключевые метрики</p>
    """,
    unsafe_allow_html=True
)

# --- Боковое меню настроек ---
st.sidebar.header("⚙️ Настройки")
uploaded_file = st.sidebar.file_uploader("Загрузите CSV из YouTube Studio", type=["csv"])
n = st.sidebar.slider("Сколько последних видео показывать:", 3, 20, 7)

# Чекбоксы для выбора метрик
show_views = st.sidebar.checkbox("👁️ Просмотры", value=True)
show_ctr = st.sidebar.checkbox("📈 CTR", value=False)
show_avd = st.sidebar.checkbox("⏱️ AVD (средняя продолжительность просмотра)", value=False)
show_duration = st.sidebar.checkbox("🕒 Длительность видео", value=False)
show_table = st.sidebar.checkbox("📊 Таблица всех метрик", value=True)

# --- Если файл загружен ---
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # убираем строку "Итоговое значение", если есть
    if "Название видео" in df.columns:
        df = df[df["Название видео"].notna()]

    # сортировка по дате публикации (если есть)
    if "Время публикации видео" in df.columns:
        df["Время публикации видео"] = pd.to_datetime(df["Время публикации видео"], errors="coerce")
        df = df.sort_values("Время публикации видео", ascending=False)

    # оставляем только последние N видео
    df = df.head(n).copy()

    # делаем сокращённые названия для графиков
    df["Название (сокр.)"] = df["Название видео"].apply(
        lambda x: x[:40] + "..." if len(str(x)) > 40 else x
    )

    # --- Метрики ---
    if show_views and "Просмотры" in df.columns:
        st.subheader("👁️ Просмотры по видео")
        fig = px.bar(
            df,
            x="Название (сокр.)",
            y="Просмотры",
            text="Просмотры",
            hover_data={"Название видео": True, "Просмотры": True},
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(xaxis_tickangle=-30, height=500)
        st.plotly_chart(fig, use_container_width=True)

    if show_ctr and "CTR для значков видео (%)" in df.columns:
        st.subheader("📈 CTR по видео")
        fig = px.bar(
            df,
            x="Название (сокр.)",
            y="CTR для значков видео (%)",
            text="CTR для значков видео (%)",
            hover_data={"Название видео": True},
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(xaxis_tickangle=-30, height=500)
        st.plotly_chart(fig, use_container_width=True)

    if show_avd and "Средняя продолжительность просмотра" in df.columns:
        st.subheader("⏱️ Средняя продолжительность просмотра")
        fig = px.bar(
            df,
            x="Название (сокр.)",
            y="Средняя продолжительность просмотра",
            text="Средняя продолжительность просмотра",
            hover_data={"Название видео": True},
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(xaxis_tickangle=-30, height=500)
        st.plotly_chart(fig, use_container_width=True)

    if show_duration and "Расчётная длительность (секунды)" in df.columns:
        st.subheader("🕒 Длительность видео (секунды)")
        fig = px.bar(
            df,
            x="Название (сокр.)",
            y="Расчётная длительность (секунды)",
            text="Расчётная длительность (секунды)",
            hover_data={"Название видео": True},
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(xaxis_tickangle=-30, height=500)
        st.plotly_chart(fig, use_container_width=True)

    if show_table:
        st.subheader("📊 Таблица всех метрик")
        st.dataframe(df, use_container_width=True)

# --- Если файл не загружен ---
else:
    st.info("👆 Загрузите CSV-файл, чтобы увидеть аналитику")
