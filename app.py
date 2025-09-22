import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="YouTube Dashboard", layout="wide")

st.title("📊 YouTube Analytics Dashboard")

uploaded_file = st.file_uploader("Загрузите CSV из YouTube Studio", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Убираем строку "Итоговое значение"
    if "Название видео" in df.columns:
        df = df[df["Название видео"].notna()]

    # Дата публикации
    if "Время публикации видео" in df.columns:
        df["Время публикации видео"] = pd.to_datetime(df["Время публикации видео"], errors="coerce")
        df = df.sort_values("Время публикации видео", ascending=False)

    # Последние N видео
    n = st.slider("Сколько последних видео показать:", 3, 20, 8)
    subset = df.head(n).copy()

    # Укорачиваем названия для графиков
    subset["Короткое название"] = subset["Название видео"].str.slice(0, 40) + "..."

    # === Выбор метрик ===
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    chosen_metrics = st.multiselect(
        "Выберите метрики для отображения:",
        options=numeric_cols,
        default=["Просмотры", "CTR для значков видео (%)"]
    )

    # Таблица
    st.subheader("📋 Сводная таблица")
    st.dataframe(
        subset[["Название видео", "Время публикации видео"] + chosen_metrics].fillna("—")
    )

    # Графики
    if chosen_metrics:
        st.subheader("📈 Графики по выбранным метрикам")

        for metric in chosen_metrics:
            fig = px.bar(
                subset,
                x="Короткое название",
                y=metric,
                title=f"{metric} по видео",
                text=metric,
                color=metric,
                color_continuous_scale="Blues"
            )
            st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👆 Загрузите CSV из YouTube Studio, чтобы построить дашборд.")
