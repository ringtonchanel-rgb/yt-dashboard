# app.py — Year Mix Only (две метрики по годам: суммарные просмотры и кол-во видео)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ---------- Page ----------
st.set_page_config(page_title="YouTube Dashboard — Year Mix", layout="wide")
st.markdown("<h1 style='text-align:center'>📊 YouTube Dashboard — Годовой микс</h1>", unsafe_allow_html=True)
st.caption("Один экран • Годовой микс: суммарные просмотры по годам + количество видео по годам.")

# ---------- Sidebar ----------
st.sidebar.header("⚙️ Данные")
file = st.sidebar.file_uploader(
    "Загрузите CSV из YouTube Studio (как «Новая таблица - Jan 23 - Aug 25.csv»)", type=["csv"]
)
show_table = st.sidebar.checkbox("Показать таблицу с цифрами", value=False)

# ---------- Helpers ----------
def _norm(s: str) -> str:
    return s.strip().lower()

# Карта возможных названий колонок (рус/англ/варианты)
MAP = {
    "publish_time": [
        "video publish time", "publish time", "время публикации видео", "дата публикации", "publish date",
    ],
    "views": [
        "views", "просмотры"
    ],
}

def find_col(df: pd.DataFrame, names) -> str | None:
    """Находит колонку в df по списку возможных названий (учитывает регистр, пробелы и вхождения)."""
    if isinstance(names, str):
        names = [names]
    by_norm = {_norm(c): c for c in df.columns}
    # 1) точное совпадение (по нормализованному имени)
    for n in names:
        nn = _norm(n)
        if nn in by_norm:
            return by_norm[nn]
    # 2) подстрока
    for n in names:
        nn = _norm(n)
        for c in df.columns:
            if nn in _norm(c):
                return c
    return None

def detect_columns(df: pd.DataFrame):
    """Определяет ключевые колонки: дата публикации, просмотры."""
    return {
        "publish_time": find_col(df, MAP["publish_time"]),
        "views": find_col(df, MAP["views"]),
    }

# ---------- Main ----------
if not file:
    st.info("👆 Загрузите CSV, и я построю два графика по годам. Подходит выгрузка из YouTube Studio.")
    st.stop()

# Читаем CSV
df = pd.read_csv(file)
# Чиним заголовки
df.columns = [c.strip() for c in df.columns]

# В некоторых выгрузках в конце бывает строка «ИТОГО» — уберём её аккуратно
try:
    df = df[~df.apply(lambda r: r.astype(str).str.contains("итог", case=False).any(), axis=1)]
except Exception:
    pass

# Ищем колонки
C = detect_columns(df)
pub_col  = C["publish_time"]
views_col = C["views"]

# Проверка наличия обязательных колонок
missing = []
if not (pub_col and pub_col in df.columns):
    missing.append("Дата публикации")
if not (views_col and views_col in df.columns):
    missing.append("Просмотры")

if missing:
    st.error("Не хватает колонок в файле: " + ", ".join(missing))
    st.stop()

# Приводим дату публикации к datetime
df[pub_col] = pd.to_datetime(df[pub_col], errors="coerce")
# Фильтруем явные NaT
df = df[df[pub_col].notna()].copy()

# Числовое представление просмотров
df["_views_num"] = pd.to_numeric(df[views_col], errors="coerce")

# Год публикации
df["_year"] = df[pub_col].dt.year

# Агрегации
views_year = (
    df.groupby("_year", as_index=False)["_views_num"]
      .sum()
      .rename(columns={"_year": "Год", "_views_num": "Суммарное количество просмотров"})
      .sort_values("Год")
)

count_year = (
    df.groupby("_year", as_index=False)
      .size()
      .rename(columns={"_year": "Год", "size": "Количество видео"})
      .sort_values("Год")
)

# Сообщение, если пусто
if views_year.empty or count_year.empty:
    st.info("Недостаточно данных для построения графиков по годам.")
    st.stop()

# ---------- Графики ----------
c1, c2 = st.columns(2)

# ЛЕВЫЙ: суммарные просмотры по годам
fig1 = px.bar(
    views_year,
    x="Год",
    y="Суммарное количество просмотров",
    text="Суммарное количество просмотров",
    template="simple_white",
)
fig1.update_traces(marker_color="#4e79a7", texttemplate="%{text:,}", textposition="outside")
fig1.update_layout(
    title="Суммарное количество просмотров по годам",
    xaxis_title="Год публикации",
    yaxis_title="Суммарное количество просмотров",
    showlegend=False,
    margin=dict(l=10, r=10, t=50, b=10),
    height=430,
)
fig1.update_xaxes(type="category", categoryorder="category ascending")
c1.plotly_chart(fig1, use_container_width=True)

# ПРАВЫЙ: количество видео по годам
fig2 = px.bar(
    count_year,
    x="Год",
    y="Количество видео",
    text="Количество видео",
    template="simple_white",
)
fig2.update_traces(marker_color="#4e79a7", texttemplate="%{text}", textposition="outside")
fig2.update_layout(
    title="Количество видео по годам",
    xaxis_title="Год публикации",
    yaxis_title="Количество видео",
    showlegend=False,
    margin=dict(l=10, r=10, t=50, b=10),
    height=430,
)
fig2.update_xaxes(type="category", categoryorder="category ascending")
c2.plotly_chart(fig2, use_container_width=True)

# (опционально) Таблица с цифрами
if show_table:
    st.markdown("### Таблица")
    tbl = pd.merge(views_year, count_year, on="Год", how="outer").sort_values("Год")
    st.dataframe(tbl, use_container_width=True)
