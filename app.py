# app.py — v1 "Amigos Core": одна вкладка, строго по документу
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import timedelta
import re, html

# -------------------- Page --------------------
st.set_page_config(page_title="YouTube Dashboard — Core", layout="wide")
st.markdown("<h1 style='text-align:center'>📊 YouTube Dashboard — Core</h1>", unsafe_allow_html=True)
st.caption("Минимум функций: быстрые KPI, кликабельная таблица и 2 ключевых графика (вклад и качество).")

# -------------------- Sidebar --------------------
st.sidebar.header("⚙️ Данные")
file = st.sidebar.file_uploader("Загрузите CSV из YouTube Studio (вкладка Content/Видео)", type=["csv"])
n_videos = st.sidebar.slider("Сколько последних видео показывать:", 3, 300, 30)
search_q = st.sidebar.text_input("Поиск по названию/ID")
only_shorts = st.sidebar.checkbox("Только Shorts (<60 сек)", value=False)

st.sidebar.header("📈 Блоки")
show_kpi   = st.sidebar.checkbox("KPI-плашки", value=True)
show_table = st.sidebar.checkbox("Таблица с кликами", value=True)
show_bar   = st.sidebar.checkbox("Гистограмма: Просмотры по видео", value=True)
show_quality = st.sidebar.checkbox("Качество: AVD/Duration & CTR/Impressions", value=True)

# -------------------- Helpers --------------------
def norm(s:str)->str:
    return s.strip().lower()

METRICS_MAP = {
    "title": ["название видео","title","video title","название"],
    "video_id": ["video id","external video id","контент","content","id видео","ид видео"],
    "publish_time": ["video publish time","publish time","время публикации видео","дата публикации"],
    "views": ["views","просмотры"],
    "impressions": ["impressions","показы"],
    "ctr": ["impressions click-through rate","ctr","impressions click-through rate (%)","ctr для значков видео"],
    "avd": ["average view duration","средняя продолжительность просмотра"],
    "duration": ["duration","продолжительность","длительность"],
    "revenue": ["estimated partner revenue","расчетный доход","расчётный доход"],
}

def find_col(df: pd.DataFrame, keys) -> str | None:
    if isinstance(keys,str): keys=[keys]
    # точные совпадения
    by_norm = {norm(c): c for c in df.columns}
    for k in keys:
        nk = norm(k)
        if nk in by_norm: return by_norm[nk]
    # частичные
    for k in keys:
        nk = norm(k)
        for c in df.columns:
            if nk in norm(c): return c
    return None

def detect_cols(df: pd.DataFrame):
    return {k: find_col(df, v) for k,v in METRICS_MAP.items()}

def parse_duration_to_seconds(x):
    if pd.isna(x): return np.nan
    s = str(x).strip()
    # число секунд
    if re.fullmatch(r"\d+(\.\d+)?", s): 
        try: return float(s)
        except: return np.nan
    parts = s.split(":")
    try:
        parts = [int(p) for p in parts]
        if len(parts)==2: m,s = parts; return m*60+s
        if len(parts)==3: h,m,s = parts; return h*3600+m*60+s
        return np.nan
    except:
        return np.nan

def seconds_to_hhmmss(x):
    try:
        x = int(round(float(x))); 
        return str(timedelta(seconds=x))
    except: 
        return ""

def shorten(text, n=40):
    t = "" if text is None else str(text)
    return (t[:n]+"…") if len(t)>n else t

def add_clickable_title(df, title_col, id_col, new_col="Видео"):
    out = df.copy()
    if id_col is None or id_col not in out.columns:
        out[new_col] = out[title_col] if title_col in out.columns else out.index.astype(str)
        return out
    titles = out[title_col] if (title_col and title_col in out.columns) else out[id_col].astype(str)
    urls = "https://www.youtube.com/watch?v=" + out[id_col].astype(str)
    out[new_col] = [f"<a href='{u}' target='_blank'>{html.escape(str(t))}</a>" for t,u in zip(titles, urls)]
    return out

def safe_table(df, cols, escape=False):
    use = [c for c in cols if c in df.columns]
    if not use:
        st.info("Нет доступных колонок для отображения.")
        return
    st.markdown(df[use].to_html(index=False, escape=escape), unsafe_allow_html=True)

# -------------------- Main --------------------
if not file:
    st.info("👆 Загрузите CSV. Я ожидаю выгрузку из Studio (таблица Видео/Content).")
    st.stop()

df = pd.read_csv(file)
df.columns = [c.strip() for c in df.columns]
C = detect_cols(df)

title_col  = C["title"]
id_col     = C["video_id"]
pub_col    = C["publish_time"]
views_col  = C["views"]
imp_col    = C["impressions"]
ctr_col    = C["ctr"]
avd_col    = C["avd"]
dur_col    = C["duration"]
rev_col    = C["revenue"]

# дата + сортировка
if pub_col:
    df[pub_col] = pd.to_datetime(df[pub_col], errors="coerce")
    df = df.sort_values(pub_col, ascending=False)

# длительности в секунды
df["__duration_sec__"] = df[dur_col].apply(parse_duration_to_seconds) if dur_col else np.nan
df["__avd_sec__"]      = df[avd_col].apply(parse_duration_to_seconds) if avd_col else np.nan

# подсечки
if id_col: 
    df["YouTube Link"] = "https://www.youtube.com/watch?v=" + df[id_col].astype(str)
if title_col:
    df["__title_short__"] = df[title_col].apply(lambda x: shorten(x, 36))
x_axis = "__title_short__" if title_col else (id_col if id_col else None)

# фильтры
df = df.head(n_videos).copy()
if search_q:
    cols_for_search = [c for c in [title_col, id_col] if c]
    if cols_for_search:
        df = df[df[cols_for_search].astype(str).apply(lambda r: search_q.lower() in " ".join(r).lower(), axis=1)]
if only_shorts:
    df = df[df["__duration_sec__"] < 60]

# ================== UI (одна вкладка) ==================
# KPI
if show_kpi:
    st.subheader("KPI по выборке")
    c = st.columns(5)
    if views_col: c[0].metric("Views", f"{df[views_col].sum():,.0f}")
    if ctr_col:
        ctr_vals = pd.to_numeric(df[ctr_col], errors="coerce")
        if ctr_vals.notna().any(): c[1].metric("CTR avg", f"{ctr_vals.mean():.2f}%")
    if df["__avd_sec__"].notna().any(): c[2].metric("AVD avg", seconds_to_hhmmss(df["__avd_sec__"].mean()))
    if rev_col: c[3].metric("Revenue", f"{df[rev_col].sum():,.2f}")
    if pub_col: c[4].metric("Videos", f"{df.shape[0]:,}")

# Таблица
if show_table:
    st.subheader("Таблица (кликабельные названия)")
    view = add_clickable_title(df, title_col, id_col, new_col="Видео")
    cols = ["Видео"] + [x for x in [id_col, views_col, ctr_col, imp_col, "__avd_sec__", "__duration_sec__", "YouTube Link"] if (x in view.columns) or (x=="YouTube Link")]
    human = {"__avd_sec__":"AVD (сек)","__duration_sec__":"Длительность (сек)"}
    safe_table(view.rename(columns=human), cols, escape=False)

# Просмотры по видео
if show_bar and x_axis and views_col:
    st.subheader("Просмотры по видео")
    fig = px.bar(df, x=x_axis, y=views_col, text=views_col)
    fig.update_traces(textposition="outside")
    fig.update_layout(xaxis_tickangle=-35, height=440, margin=dict(l=8,r=8,t=30,b=20))
    st.plotly_chart(fig, use_container_width=True)

# Качество: AVD/Duration и CTR/Impressions
if show_quality:
    st.subheader("Качество")
    c1, c2 = st.columns(2)

    # AVD vs Duration
    if df["__duration_sec__"].notna().any() and df["__avd_sec__"].notna().any():
        fig1 = px.scatter(
            df, x="__duration_sec__", y="__avd_sec__",
            hover_name=title_col if title_col else id_col
        )
        fig1.update_layout(height=420, xaxis_title="Duration (сек)", yaxis_title="AVD (сек)")
        c1.plotly_chart(fig1, use_container_width=True)
    else:
        c1.info("Нет AVD/Duration в распознаваемом формате.")

    # CTR vs Impressions
    if ctr_col and imp_col:
        fig2 = px.scatter(
            df, x=imp_col, y=pd.to_numeric(df[ctr_col], errors="coerce"),
            hover_name=title_col if title_col else id_col
        )
        fig2.update_layout(height=420, xaxis_title="Impressions", yaxis_title="CTR (%)")
        c2.plotly_chart(fig2, use_container_width=True)
    else:
        c2.info("Нет CTR/Impressions для сравнения.")
