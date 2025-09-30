import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="YouTube Yearly Analytics", layout="wide")

st.title("📊 YouTube Analytics — просмотры по годам публикации")

# Загрузка CSV
uploaded_file = st.file_uploader("Загрузите CSV (YouTube экспорт)", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Ошибка чтения файла: {e}")
        st.stop()

    st.write("### Пара первых строк таблицы")
    st.dataframe(df.head())

    # Ищем колонку даты публикации
    publish_cols = [c for c in df.columns if "publish" in c.lower() or "дата" in c.lower()]
    if not publish_cols:
        st.error("Не найдена колонка с датой публикации (publish time/date).")
        st.stop()

    pub_col = publish_cols[0]
    df["publish_time"] = pd.to_datetime(df[pub_col], errors="coerce")
    df = df.dropna(subset=["publish_time"])
    df["year"] = df["publish_time"].dt.year

    # Ищем колонку просмотров
    view_cols = [c for c in df.columns if "view" in c.lower() or "просмотр" in c.lower()]
    if not view_cols:
        st.error("Не найдена колонка с просмотрами (views).")
        st.stop()

    views_col = view_cols[0]
    df[views_col] = pd.to_numeric(df[views_col], errors="coerce")

    # Агрегируем по годам
    agg = df.groupby("year")[views_col].sum().reset_index().sort_values("year")

    st.write("### Сумма просмотров по годам")
    st.dataframe(agg)

    # Строим график
    fig = px.bar(
        agg, x="year", y=views_col,
        labels={"year": "Год публикации", views_col: "Просмотры"},
        title="Просмотры по годам публикации",
        text_auto=True
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("⬆️ Загрузите CSV, чтобы увидеть аналитику.")
