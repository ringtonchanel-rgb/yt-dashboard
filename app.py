import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io, re
from datetime import datetime

st.set_page_config(page_title="YouTube Cohort Analytics", layout="wide")

# -----------------------------
# Помощники парсинга/поиска колонок
# -----------------------------
def _norm(s: str) -> str:
    return str(s).strip().lower()

COLMAP = {
    "publish_time": [
        "video publish time","publish time","publish date","upload date",
        "время публикации видео","дата публикации","дата"
    ],
    "views": ["views","просмотры","просмторы","просмотры (views)"],
    "impressions": ["impressions","показы","показы значков","показы для значков","показы для значков видео"],
    "ctr": ["impressions click-through rate","ctr","ctr (%)","ctr для значков","ctr для значков видео (%)","ctr видео"],
    "watch_hours": ["watch time (hours)","watch time hours","время просмотра (часы)","время просмотра (часов)"],
    "watch_minutes":["watch time (minutes)","watch time (mins)","время просмотра (мин)","время просмотра (минуты)"],
    "engaged_views":["engaged views","вовлеченные просмотры","просмотры с вовлечением"],
    "title": ["title","название видео","video title","видео","название","content","контент"]
}

def find_col(df, names):
    if isinstance(names,str): names=[names]
    pool = {_norm(c): c for c in df.columns}
    for n in names:
        nn=_norm(n)
        if nn in pool: return pool[nn]
    # частичное совпадение
    for n in names:
        nn=_norm(n)
        for c in df.columns:
            if nn in _norm(c): return c
    return None

def detect_columns(df):
    return {k: find_col(df, v) for k,v in COLMAP.items()}

def to_number(x):
    if x is None: return np.nan
    if isinstance(x,(int,float,np.number)): return float(x)
    s = str(x).strip()
    if s=="" or s.lower() in {"nan","none"}: return np.nan
    s = s.replace(" ", "").replace("\u202f","").replace("\xa0","")
    if s.endswith("%"): s = s[:-1]
    if "," in s and "." not in s:
        s = s.replace(",", ".")
    try: return float(s)
    except: return np.nan

# -----------------------------
# Загрузка CSV (мультифайл)
# -----------------------------
st.title("YouTube Analytics — Cohort Growth by Publish Year")

files = st.file_uploader("Загрузите один или несколько CSV (YouTube Studio export или ваш отчёт)", type=["csv"], accept_multiple_files=True)
if not files:
    st.info("Загрузите хотя бы один CSV, чтобы построить аналитику.")
    st.stop()

dfs=[]
meta=[]
for uf in files:
    raw = uf.getvalue()
    read_ok=False
    for enc in (None,"utf-8-sig","cp1251"):
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=enc) if enc else pd.read_csv(io.BytesIO(raw))
            read_ok=True
            break
        except Exception:
            pass
    if not read_ok or df is None or df.empty:
        meta.append(f"❌ {uf.name}: не удалось прочитать CSV.")
        continue
    df.columns=[c.strip() for c in df.columns]
    meta.append(f"✅ {uf.name}: {df.shape[0]} строк, {df.shape[1]} колонок.")
    dfs.append(df)

st.caption("Загрузка:")
for m in meta: st.write(m)

if not dfs:
    st.error("Ни один файл не прочитан. Проверьте формат.")
    st.stop()

df = pd.concat(dfs, ignore_index=True)

# -----------------------------
# Нормализация к ядру колонок
# -----------------------------
C = detect_columns(df)

if not C["publish_time"]:
    st.error("Не нашёл колонку даты публикации видео (Publish time). Без неё когортный анализ невозможен.")
    st.stop()

df["_publish_time"] = pd.to_datetime(df[C["publish_time"]], errors="coerce")
df = df.dropna(subset=["_publish_time"]).copy()
df["_pub_year"]  = df["_publish_time"].dt.year
df["_pub_month"] = df["_publish_time"].dt.month

# Метрики, которыми реально можно оперировать
metrics_pool = []
if C["views"]:         metrics_pool.append(("Просмотры","views","sum"))
if C["impressions"]:   metrics_pool.append(("Показы","impressions","sum"))
if C["engaged_views"]: metrics_pool.append(("Вовлечённые просмотры","engaged_views","sum"))
if C["watch_hours"] or C["watch_minutes"]:
    # приведём к часам
    if C["watch_hours"]:
        df["_watch_hours"] = pd.to_numeric(df[C["watch_hours"]].apply(to_number), errors="coerce")
    else:
        df["_watch_hours"] = pd.to_numeric(df[C["watch_minutes"]].apply(to_number), errors="coerce")/60.0
    metrics_pool.append(("Часы просмотра","_watch_hours","sum"))

if C["ctr"]:
    df["_ctr"] = pd.to_numeric(df[C["ctr"]].apply(to_number), errors="coerce")
    # CTR корректнее усреднять по видео (или взвешенно по показам — позже можно добавить)
    metrics_pool.append(("CTR, %","_ctr","mean"))

if not metrics_pool:
    st.error("Не нашёл ни одной метрики (Просмотры/Показы/AVD/CTR/…): нечего агрегировать.")
    st.stop()

# Приведём числовые колонки
for _, col, _agg in metrics_pool:
    if col not in df.columns:
        continue
    if col.startswith("_"):  # уже подготовлено
        continue
    df[col] = pd.to_numeric(df[col].apply(to_number), errors="coerce")

# -----------------------------
# Панель параметров
# -----------------------------
st.subheader("Параметры анализа")
c1,c2,c3,c4 = st.columns([2,2,2,2])

metric_label = c1.selectbox(
    "Метрика",
    [m[0] for m in metrics_pool],
    index=0
)
metric_col  = [m[1] for m in metrics_pool if m[0]==metric_label][0]
metric_agg  = [m[2] for m in metrics_pool if m[0]==metric_label][0]

years_sorted = sorted(df["_pub_year"].unique())
yr_min, yr_max = years_sorted[0], years_sorted[-1]
yr_from, yr_to = c2.select_slider("Диапазон лет выпуска", options=years_sorted,
                                  value=(yr_min, yr_max))
mask_year = (df["_pub_year"]>=yr_from) & (df["_pub_year"]<=yr_to)
dff = df.loc[mask_year].copy()

top_k = int(c3.number_input("Сколько лет (Top-K) показывать на кривых", min_value=1, max_value=len(years_sorted), value=min(6,len(years_sorted))))

mode = c4.radio("Режим кривой", ["Кумулятивный рост","Помесячно"], horizontal=True)
normalize = st.checkbox("Нормализовать кривые до 0–100% по годам (для сравнения формы)", value=False)
log_y      = st.checkbox("Логарифмическая шкала Y", value=False)

st.divider()

# -----------------------------
# 1) Бар-чарт по годам выпуска (Total)
# -----------------------------
agg_total = (dff.groupby("_pub_year")
             .agg(value=(metric_col, metric_agg))
             .reset_index()
             .rename(columns={"_pub_year":"Год","value":metric_label})
             .sort_values("Год"))

cA,cB = st.columns([1,2])
with cA:
    st.markdown("#### Сумма по годам выпуска")
with cB:
    st.caption("Это «суммарный вклад» каждого года выпуска по выбранной метрике.")

fig_year = px.bar(
    agg_total, x="Год", y=metric_label,
    template="simple_white", color_discrete_sequence=["#4e79a7"]
)
fig_year.update_layout(
    height=420, xaxis_title="Год выпуска", yaxis_title=metric_label
)
if log_y:
    fig_year.update_yaxes(type="log")
st.plotly_chart(fig_year, use_container_width=True)

st.download_button(
    "Скачать таблицу (сумма по годам, CSV)",
    data=agg_total.to_csv(index=False).encode("utf-8"),
    file_name="year_totals.csv",
    mime="text/csv"
)

st.divider()

# -----------------------------
# 2) Когортные «линии роста» по месяцам
# -----------------------------
st.markdown("### Когортные кривые по месяцам (год выпуска → янв…дек)")

# агрегируем по (year, month)
month_agg = (dff.groupby(["_pub_year","_pub_month"])
             .agg(v=(metric_col, metric_agg))
             .reset_index()
             .rename(columns={"_pub_year":"Год","_pub_month":"Месяц"}))

# вычислим итог по году, чтобы определить Top-K лет
tot_by_year = month_agg.groupby("Год")["v"].sum().sort_values(ascending=False)
top_years = tot_by_year.index.tolist()[:top_k]
month_agg_top = month_agg[month_agg["Год"].isin(top_years)].copy()

# разворот на месяца 1..12, fillna=0
pivot = month_agg_top.pivot_table(index="Год", columns="Месяц", values="v", aggfunc="sum").fillna(0.0)
# гарантируем все месяцы
for m in range(1,13):
    if m not in pivot.columns:
        pivot[m]=0.0
pivot = pivot[sorted(pivot.columns)]

# кумулятив или помесячно
if mode.startswith("Кумулятив"):
    pivot = pivot.cumsum(axis=1)

# нормализация 0..1 (или 0..100%)
if normalize:
    totals = pivot.max(axis=1) if mode.startswith("Кумулятив") else pivot.sum(axis=1)
    pivot = pivot.div(totals.where(totals>0, np.nan), axis=0) * 100.0
    y_label = f"{metric_label} — норм., %"
else:
    y_label = metric_label

# в «длинный» формат
curve = pivot.reset_index().melt(id_vars="Год", var_name="Месяц", value_name="value")
curve["Месяц"] = curve["Месяц"].astype(int)
# для красивых подписей
month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
curve["Месяц/label"] = curve["Месяц"].apply(lambda m: month_names[m-1])

fig_curve = px.line(
    curve, x="Месяц/label", y="value", color="Год",
    markers=True, template="simple_white"
)
fig_curve.update_layout(
    height=480,
    xaxis_title="Месяц публикации",
    yaxis_title=y_label,
    legend=dict(orientation="h", y=1.02, yanchor="bottom"),
    margin=dict(l=60,r=30,t=40,b=60)
)
if log_y and not normalize:
    fig_curve.update_yaxes(type="log")

st.plotly_chart(fig_curve, use_container_width=True)

# экспорт подложки
st.download_button(
    "Скачать таблицу (кривые, CSV)",
    data=curve.rename(columns={"value": y_label}).to_csv(index=False).encode("utf-8"),
    file_name="cohort_curves.csv",
    mime="text/csv"
)

st.caption("""
**Как читать график:**
- Каждая линия — это *когорта по году выпуска* (2020, 2021, 2022…).
- По оси X — месяцы **в году выпуска** (Jan…Dec).  
- В режиме *Кумулятивный рост* — линия показывает накопление метрики в рамках календаря публикаций.
- *Нормализация* до 0–100% помогает сравнить **форму** кривых между годами (кто растёт быстрее/дольше), даже если объёмы разные.
""")
