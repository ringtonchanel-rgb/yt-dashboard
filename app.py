# app.py — Zero Clean Core (одна страница, под отчет из документа)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import timedelta
import re, html

st.set_page_config(page_title="YouTube Dashboard — Zero Core", layout="wide")
st.markdown("<h1 style='text-align:center'>📊 YouTube Dashboard — Zero Core</h1>", unsafe_allow_html=True)
st.caption("Один экран. Загружаешь CSV — получаешь таблицы как в анализе (годовой микс, локомотивы, качество).")

# ------------------ Sidebar ------------------
st.sidebar.header("⚙️ Данные")
file = st.sidebar.file_uploader("Загрузите CSV (как «Новая таблица - Jan 23 - Aug 25.csv»)", type=["csv"])

st.sidebar.header("🎛 Параметры выборки")
n_videos = st.sidebar.slider("Сколько последних видео брать (по дате публикации):", 10, 1000, 500, step=10)
top_k    = st.sidebar.slider("ТОП «локомотивов» по просмотрам:", 5, 100, 20)
only_year = st.sidebar.selectbox("Фильтр по году публикации (опц.)", ["Все годы"] + [str(y) for y in range(2018, 2031)], index=0)

st.sidebar.header("🧩 Что показывать")
show_year_mix  = st.sidebar.checkbox("Таблица «Микс по годам»", value=True)
show_locom     = st.sidebar.checkbox("Таблица «Локомотивы (ТОП по просмотрам)»", value=True)
show_quality   = st.sidebar.checkbox("Таблица «Качество (AVD / %досмотра)»", value=True)
show_ctr       = st.sidebar.checkbox("Таблица «CTR & Показы (если есть)»", value=True)
show_underperf = st.sidebar.checkbox("Таблица «Проседают (ниже медиан)»", value=True)
show_charts    = st.sidebar.checkbox("Включить графики к таблицам (минимум)", value=False)

# ------------------ Helpers ------------------
def norm(s:str)->str: return s.strip().lower()

MAP = {
    "title": ["название видео","title","video title","название"],
    "video_id": ["video id","external video id","контент","content","id видео","ид видео"],
    "publish_time": ["video publish time","publish time","время публикации видео","дата публикации"],
    "views": ["views","просмотры"],
    "impressions": ["impressions","показы"],
    "ctr": ["impressions click-through rate","ctr","impressions click-through rate (%)","ctr для значков видео"],
    "avd": ["average view duration","средняя продолжительность просмотра"],
    "duration": ["duration","продолжительность","длительность"],
    "revenue": ["estimated partner revenue","расчетный доход","расчётный доход"],
    "watch_time_hours": ["watch time (hours)","время просмотра (часы)"],
    "unique_viewers": ["unique viewers","уникальные зрители"],
    "engaged_views": ["engaged views","заинтересованные просмотры"],
}

def find_col(df: pd.DataFrame, keys) -> str | None:
    if isinstance(keys, str): keys=[keys]
    by_norm = {norm(c): c for c in df.columns}
    for k in keys:
        if norm(k) in by_norm: return by_norm[norm(k)]
    for k in keys:
        nk = norm(k)
        for c in df.columns:
            if nk in norm(c): return c
    return None

def detect_cols(df: pd.DataFrame):
    return {k: find_col(df, v) for k,v in MAP.items()}

def parse_duration_to_seconds(x):
    if pd.isna(x): return np.nan
    s = str(x).strip()
    if re.fullmatch(r"\d+(\.\d+)?", s):
        try: return float(s)
        except: return np.nan
    parts = s.split(":")
    try:
        parts = [int(p) for p in parts]
        if len(parts)==2: m, ss = parts; return m*60 + ss
        if len(parts)==3: h,m,ss = parts; return h*3600 + m*60 + ss
        return np.nan
    except: return np.nan

def seconds_to_hhmmss(x):
    try:
        x = int(round(float(x))); return str(timedelta(seconds=x))
    except: return ""

def shorten(text, n=60):
    t = "" if text is None else str(text)
    return (t[:n]+"…") if len(t)>n else t

def add_clickable(df, title_col, id_col, new_col="Видео"):
    out = df.copy()
    if id_col is None or id_col not in out.columns:
        out[new_col] = out[title_col] if title_col in out.columns else out.index.astype(str)
        return out
    titles = out[title_col] if (title_col and title_col in out.columns) else out[id_col].astype(str)
    urls = "https://www.youtube.com/watch?v=" + out[id_col].astype(str)
    out[new_col] = [f"<a href='{u}' target='_blank'>{html.escape(str(t))}</a>" for t,u in zip(titles, urls)]
    return out

def html_table(df, cols, escape=False):
    use = [c for c in cols if c in df.columns]
    if not use:
        st.info("Нет колонок для отображения."); return
    st.markdown(df[use].to_html(index=False, escape=escape), unsafe_allow_html=True)

# ------------------ Main ------------------
if not file:
    st.info("👆 Загрузите отчёт CSV. Подходит формат «Новая таблица - Jan 23 - Aug 25.csv».")
    st.stop()

df = pd.read_csv(file)
# Убираем возможную строку «ИТОГО», тримим пробелы в заголовках
df = df[~df.apply(lambda r: r.astype(str).str.contains("итог", case=False).any(), axis=1)]
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
wth_col    = C["watch_time_hours"]

# Приводим дату и сортируем по ней (новые сверху)
if pub_col:
    df[pub_col] = pd.to_datetime(df[pub_col], errors="coerce")
    df = df.sort_values(pub_col, ascending=False)

# Приводим длительности в секунды
df["AVD_sec"] = df[avd_col].apply(parse_duration_to_seconds) if avd_col else np.nan
df["Dur_sec"] = df[dur_col].apply(parse_duration_to_seconds) if dur_col else np.nan
df["Avg_%_viewed"] = np.where(
    (df["AVD_sec"].notna()) & (df["Dur_sec"].replace(0,np.nan).notna()),
    df["AVD_sec"]/df["Dur_sec"]*100.0, np.nan
)

# Ограничение по последним N видео
if pub_col:
    df = df.head(n_videos).copy()

# Фильтр по году (если задан)
if pub_col and only_year != "Все годы":
    year = int(only_year)
    df = df[df[pub_col].dt.year == year]

# Колонка с ссылкой
if id_col: df["YouTube Link"] = "https://www.youtube.com/watch?v=" + df[id_col].astype(str)

# ------------------ KPI (минимум, без графиков) ------------------
k = st.columns(5)
if views_col: k[0].metric("Views (sum)", f"{pd.to_numeric(df[views_col], errors='coerce').sum():,.0f}")
if wth_col:   k[1].metric("Watch time (h)", f"{pd.to_numeric(df[wth_col], errors='coerce').sum():,.1f}")
if df["AVD_sec"].notna().any(): k[2].metric("Avg AVD", seconds_to_hhmmss(df["AVD_sec"].mean()))
if ctr_col:
    ctr_vals = pd.to_numeric(df[ctr_col], errors="coerce")
    if ctr_vals.notna().any(): k[3].metric("CTR avg", f"{ctr_vals.mean():.2f}%")
if pub_col: k[4].metric("Videos", f"{len(df):,}")

st.markdown("---")

# ===================== ТАБЛИЦЫ =====================

# 1) МИКС ПО ГОДАМ (как в анализе: сколько видео + сколько просмотров по годам)
if show_year_mix and pub_col and views_col:
    st.subheader("Микс по годам публикации (Count & Views)")
    tmp = df[[pub_col, views_col]].copy()
    tmp["Год"] = tmp[pub_col].dt.year
    by_year = tmp.groupby("Год", as_index=False).agg(
        Видео=("Год", "count"),
        Просмотры=(views_col, lambda s: pd.to_numeric(s, errors="coerce").sum())
    ).sort_values("Просмотры", ascending=False)
    html_table(by_year, ["Год","Видео","Просмотры"])
    if show_charts:
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.bar(by_year, x="Год", y="Видео", title="Кол-во видео по годам"), use_container_width=True)
        c2.plotly_chart(px.bar(by_year, x="Год", y="Просмотры", title="Просмотры по годам"), use_container_width=True)
    st.caption("Идея как в документе: оценить вклад старых/новых годов публикации в текущие просмотры.")

# 2) ЛОКОМОТИВЫ (ТОП по просмотрам) — с кликабельными названиями
if show_locom and views_col:
    st.subheader(f"Локомотивы канала — ТОП-{top_k} по просмотрам")
    top_df = df.sort_values(pd.to_numeric(df[views_col], errors="coerce"), ascending=False).head(top_k).copy()
    top_df = add_clickable(top_df, title_col, id_col, new_col="Видео")
    cols = ["Видео"] + [c for c in [id_col, views_col, "AVD_sec", "Avg_%_viewed", "YouTube Link"] if c in top_df.columns or c=="YouTube Link"]
    human = {"AVD_sec":"AVD (сек)"}
    html_table(top_df.rename(columns=human), cols, escape=False)
    if show_charts:
        fig = px.bar(top_df, x=top_df[title_col] if title_col else top_df[id_col], y=views_col, text=views_col)
        fig.update_traces(textposition="outside"); fig.update_layout(xaxis_tickangle=-35, height=420)
        st.plotly_chart(fig, use_container_width=True)
    st.caption("Ровно как в документе: видно, что часть топов «умирает», если подключить ряд — сделаем decay отдельно.")

# 3) КАЧЕСТВО (AVD / %досмотра) — концентрат по удержанию
if show_quality and df["AVD_sec"].notna().any():
    st.subheader("Качество: удержание")
    q = df[[c for c in [title_col, id_col, "AVD_sec", "Dur_sec", "Avg_%_viewed", "YouTube Link"] if c]].copy()
    q = add_clickable(q, title_col, id_col, new_col="Видео")
    q["AVD (чч:мм:сс)"] = q["AVD_sec"].apply(seconds_to_hhmmss)
    q["Dur (чч:мм:сс)"] = q["Dur_sec"].apply(seconds_to_hhmmss)
    cols = ["Видео"] + [c for c in [id_col, "AVD (чч:мм:сс)", "Dur (чч:мм:сс)", "Avg_%_viewed", "YouTube Link"] if c in q.columns or c=="YouTube Link"]
    html_table(q, cols, escape=False)
    if show_charts and "Dur_sec" in q.columns:
        fig = px.scatter(df, x="Dur_sec", y="AVD_sec", hover_name=title_col if title_col else id_col,
                         labels={"Dur_sec":"Duration (сек)", "AVD_sec":"AVD (сек)"})
        st.plotly_chart(fig, use_container_width=True)
    st.caption("Смотри средний AVD и % досмотра — в документе это ключевой фактор падения ранжирования.")

# 4) CTR & ПОКАЗЫ (если есть в отчёте)
if show_ctr and ctr_col and imp_col:
    st.subheader("CTR & Показы (если есть в отчёте)")
    t = df[[c for c in [title_col, id_col, imp_col, ctr_col, views_col, "YouTube Link"] if c]].copy()
    t = add_clickable(t, title_col, id_col, new_col="Видео")
    cols = ["Видео"] + [c for c in [id_col, imp_col, ctr_col, views_col, "YouTube Link"] if c in t.columns or c=="YouTube Link"]
    html_table(t, cols, escape=False)
    if show_charts:
        fig = px.scatter(df, x=imp_col, y=pd.to_numeric(df[ctr_col], errors="coerce"),
                         hover_name=title_col if title_col else id_col,
                         labels={imp_col:"Impressions", "y":"CTR (%)"})
        st.plotly_chart(fig, use_container_width=True)
    st.caption("По документу: резкого падения CTR по каналу нет, но стартовый CTR у новых роликов слабый — видно по этой паре.")

# 5) ПРОСЕДАЮТ (ниже медиан по ключевым метрикам)
if show_underperf:
    st.subheader("Проседают относительно медианы (быстрый фильтр)")
    meds = {}
    if views_col: meds["views"] = np.nanmedian(pd.to_numeric(df[views_col], errors="coerce"))
    if ctr_col:   meds["ctr"]   = np.nanmedian(pd.to_numeric(df[ctr_col], errors="coerce"))
    if df["AVD_sec"].notna().any(): meds["avd"] = np.nanmedian(df["AVD_sec"])
    # флаги
    bad = df.copy()
    if "avd" in meds:   bad["flag_avd"] = bad["AVD_sec"] < meds["avd"]
    if "ctr" in meds and ctr_col: bad["flag_ctr"] = pd.to_numeric(bad[ctr_col], errors="coerce") < meds["ctr"]
    if "views" in meds and views_col: bad["flag_views"] = pd.to_numeric(bad[views_col], errors="coerce") < meds["views"]
    mask = False
    for c in ["flag_avd","flag_ctr","flag_views"]:
        if c in bad.columns:
            mask = mask | bad[c].fillna(False)
    under = bad[mask] if isinstance(mask, pd.Series) else bad.iloc[0:0]
    if under.empty:
        st.success("⛳ Всё выше или около медиан по выбранной выборке.")
    else:
        u = add_clickable(under, title_col, id_col, new_col="Видео")
        cols = ["Видео"] + [x for x in [id_col, views_col, ctr_col, "AVD_sec", "Avg_%_viewed", "YouTube Link"] if x in u.columns or x=="YouTube Link"]
        human = {"AVD_sec":"AVD (сек)"}
        html_table(u.rename(columns=human), cols, escape=False)
        st.caption("Таблица подсвечивает видео, которые ниже медианных значений (Views/CTR/AVD) — это «подозреваемые» в документе.")

# ------------------ Footer ------------------
st.markdown("---")
st.caption("Формат гибкий: если в файле нет Impressions/CTR — соответствующий блок скрывается. Все названия видео кликабельны.")
