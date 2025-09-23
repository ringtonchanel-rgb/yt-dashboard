# app.py — Zero Clean Core (robust)
# Один экран под отчёты формата "Новая таблица - Jan 23 - Aug 25.csv"
# Все блоки проверяют наличие нужных колонок и НЕ падают.

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import timedelta
import re, html

# ---------- Page ----------
st.set_page_config(page_title="YouTube Dashboard — Zero Core", layout="wide")
st.markdown("<h1 style='text-align:center'>📊 YouTube Dashboard — Zero Core</h1>", unsafe_allow_html=True)
st.caption("Один экран. Загружаешь CSV — получаешь таблицы как в анализе (микс по годам, локомотивы, качество).")

# ---------- Sidebar ----------
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

# ---------- Helpers ----------
def norm(s:str)->str: 
    return s.strip().lower()

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
    # уже число (сек)
    if re.fullmatch(r"\d+(\.\d+)?", s):
        try: return float(s)
        except: return np.nan
    # форматы mm:ss / hh:mm:ss
    parts = s.split(":")
    try:
        parts = [int(p) for p in parts]
        if len(parts)==2:
            m, ss = parts; return m*60 + ss
        if len(parts)==3:
            h, m, ss = parts; return h*3600 + m*60 + ss
        return np.nan
    except:
        return np.nan

def seconds_to_hhmmss(x):
    try:
        x = int(round(float(x)))
        return str(timedelta(seconds=x))
    except:
        return ""

def shorten(text, n=60):
    t = "" if text is None else str(text)
    return (t[:n]+"…") if len(t)>n else t

def add_clickable(df, title_col, id_col, new_col="Видео"):
    out = df.copy()
    if id_col is None or id_col not in out.columns:
        if title_col and title_col in out.columns:
            out[new_col] = out[title_col]
        else:
            out[new_col] = out.index.astype(str)
        return out
    titles = out[title_col] if (title_col and title_col in out.columns) else out[id_col].astype(str)
    urls = "https://www.youtube.com/watch?v=" + out[id_col].astype(str)
    out[new_col] = [f"<a href='{u}' target='_blank'>{html.escape(str(t))}</a>" for t,u in zip(titles, urls)]
    return out

def html_table(df, cols, escape=False):
    use = [c for c in cols if c in df.columns]
    if not use:
        st.info("Нет колонок для отображения.")
        return
    st.markdown(df[use].to_html(index=False, escape=escape), unsafe_allow_html=True)

def num(series):
    return pd.to_numeric(series, errors="coerce")

def warn_missing(block_title, need_cols):
    st.warning(f"«{block_title}» пропущен: в отчёте нет нужных колонок → {', '.join(need_cols)}")

# ---------- Main ----------
if not file:
    st.info("👆 Загрузите отчёт CSV. Подходит формат «Новая таблица - Jan 23 - Aug 25.csv».")
    st.stop()

df = pd.read_csv(file)

# Удаляем строку «ИТОГО», если вдруг есть
try:
    df = df[~df.apply(lambda r: r.astype(str).str.contains("итог", case=False).any(), axis=1)]
except Exception:
    pass

# Чиним заголовки
df.columns = [c.strip() for c in df.columns]

# Обнаруживаем колонки
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

# Дата + сортировка по дате (новые сверху)
if pub_col and pub_col in df.columns:
    df[pub_col] = pd.to_datetime(df[pub_col], errors="coerce")
    df = df.sort_values(pub_col, ascending=False)

# Приводим длительности в секунды
df["AVD_sec"] = df[avd_col].apply(parse_duration_to_seconds) if (avd_col and avd_col in df.columns) else np.nan
df["Dur_sec"] = df[dur_col].apply(parse_duration_to_seconds) if (dur_col and dur_col in df.columns) else np.nan
df["Avg_%_viewed"] = np.where(
    (pd.to_numeric(df["AVD_sec"], errors="coerce").notna()) & 
    (pd.to_numeric(df["Dur_sec"], errors="coerce").replace(0,np.nan).notna()),
    pd.to_numeric(df["AVD_sec"], errors="coerce")/pd.to_numeric(df["Dur_sec"], errors="coerce")*100.0, 
    np.nan
)

# Ограничение по последним N видео
if pub_col and pub_col in df.columns:
    df = df.head(n_videos).copy()
else:
    # если нет даты — просто берем верхние n строк файла
    df = df.head(n_videos).copy()

# Фильтр по году (если задан и есть дата)
if pub_col and pub_col in df.columns and only_year != "Все годы":
    year = int(only_year)
    df = df[df[pub_col].dt.year == year]

# Ссылка на YouTube
if id_col and id_col in df.columns:
    df["YouTube Link"] = "https://www.youtube.com/watch?v=" + df[id_col].astype(str)

# ---------- KPI ----------
k = st.columns(5)
if views_col and views_col in df.columns:
    k[0].metric("Views (sum)", f"{num(df[views_col]).sum():,.0f}")
if wth_col and wth_col in df.columns:
    k[1].metric("Watch time (h)", f"{num(df[wth_col]).sum():,.1f}")
if "AVD_sec" in df.columns and num(df["AVD_sec"]).notna().any():
    k[2].metric("Avg AVD", seconds_to_hhmmss(num(df["AVD_sec"]).mean()))
if ctr_col and ctr_col in df.columns:
    ctr_vals = num(df[ctr_col])
    if ctr_vals.notna().any():
        k[3].metric("CTR avg", f"{ctr_vals.mean():.2f}%")
if pub_col and pub_col in df.columns:
    k[4].metric("Videos", f"{len(df):,}")

st.markdown("---")

# ===================== TABLES (with guards) =====================

# 1) МИКС ПО ГОДАМ
if show_year_mix:
    if pub_col and pub_col in df.columns and views_col and views_col in df.columns:
        st.subheader("Микс по годам публикации (Count & Views)")
        tmp = df[[pub_col, views_col]].copy()
        tmp["Год"] = tmp[pub_col].dt.year
        by_year = tmp.groupby("Год", as_index=False).agg(
            Видео=("Год", "count"),
            Просмотры=(views_col, lambda s: num(s).sum())
        ).sort_values("Просмотры", ascending=False)
        if by_year.empty:
            st.info("Нет данных для построения «Микс по годам».")
        else:
            html_table(by_year, ["Год","Видео","Просмотры"])
            if show_charts:
                c1, c2 = st.columns(2)
                c1.plotly_chart(px.bar(by_year, x="Год", y="Видео", title="Кол-во видео по годам"), use_container_width=True)
                c2.plotly_chart(px.bar(by_year, x="Год", y="Просмотры", title="Просмотры по годам"), use_container_width=True)
        st.caption("Как в документе: вклад старых/новых годов публикации в текущие просмотры.")
    else:
        need = []
        if not (pub_col and pub_col in df.columns): need.append("Дата публикации")
        if not (views_col and views_col in df.columns): need.append("Просмотры")
        warn_missing("Микс по годам", need)

# 2) ЛОКОМОТИВЫ (ТОП по просмотрам)
if show_locom:
    if views_col and views_col in df.columns:
        st.subheader(f"Локомотивы канала — ТОП-{top_k} по просмотрам")
        # безопасная сортировка по числовому представлению
        sort_series = num(df[views_col])
        top_df = df.loc[sort_series.sort_values(ascending=False).index].head(top_k).copy()
        if top_df.empty:
            st.info("Нет данных для построения «Локомотивы».")
        else:
            top_df = add_clickable(top_df, title_col, id_col, new_col="Видео")
            cols = ["Видео"] + [c for c in [id_col, views_col, "AVD_sec", "Avg_%_viewed", "YouTube Link"] 
                                if (c in top_df.columns) or (c=="YouTube Link")]
            human = {"AVD_sec":"AVD (сек)"}
            html_table(top_df.rename(columns=human), cols, escape=False)
            if show_charts:
                x_name = title_col if (title_col and title_col in top_df.columns) else (id_col if id_col in top_df.columns else None)
                if x_name:
                    fig = px.bar(top_df, x=top_df[x_name], y=views_col, text=views_col)
                    fig.update_traces(textposition="outside")
                    fig.update_layout(xaxis_tickangle=-35, height=420)
                    st.plotly_chart(fig, use_container_width=True)
        st.caption("Как в документе: быстрый список «локомотивов» по просмотрам.")
    else:
        warn_missing("Локомотивы", ["Просмотры"])

# 3) КАЧЕСТВО (AVD / %досмотра)
if show_quality:
    if "AVD_sec" in df.columns and num(df["AVD_sec"]).notna().any():
        st.subheader("Качество: удержание")
        cols_src = [c for c in [title_col, id_col, "AVD_sec", "Dur_sec", "Avg_%_viewed", "YouTube Link"] if c]
        q = df[cols_src].copy()
        q = add_clickable(q, title_col, id_col, new_col="Видео")
        q["AVD (чч:мм:сс)"] = q["AVD_sec"].apply(seconds_to_hhmmss)
        if "Dur_sec" in q.columns:
            q["Dur (чч:мм:сс)"] = q["Dur_sec"].apply(seconds_to_hhmmss)
        cols_show = ["Видео"] + [c for c in [id_col, "AVD (чч:мм:сс)", "Dur (чч:мм:сс)", "Avg_%_viewed", "YouTube Link"]
                                 if (c in q.columns) or (c=="YouTube Link")]
        html_table(q, cols_show, escape=False)

        if show_charts and "Dur_sec" in df.columns:
            fig = px.scatter(df, x="Dur_sec", y="AVD_sec",
                             hover_name=title_col if (title_col and title_col in df.columns) else (id_col if id_col in df.columns else None),
                             labels={"Dur_sec":"Duration (сек)", "AVD_sec":"AVD (сек)"})
            st.plotly_chart(fig, use_container_width=True)
        st.caption("Ключевой блок анализа: средний AVD и % досмотра.")
    else:
        warn_missing("Качество (AVD / %досмотра)", ["Average View Duration", "Duration"])

# 4) CTR & ПОКАЗЫ
if show_ctr:
    if ctr_col and ctr_col in df.columns and imp_col and imp_col in df.columns:
        st.subheader("CTR & Показы (если есть в отчёте)")
        t_cols = [c for c in [title_col, id_col, imp_col, ctr_col, views_col, "YouTube Link"] if c]
        t = df[t_cols].copy()
        t = add_clickable(t, title_col, id_col, new_col="Видео")
        cols = ["Видео"] + [c for c in [id_col, imp_col, ctr_col, views_col, "YouTube Link"] if (c in t.columns) or (c=="YouTube Link")]
        html_table(t, cols, escape=False)
        if show_charts:
            fig = px.scatter(df, x=imp_col, y=num(df[ctr_col]),
                             hover_name=title_col if (title_col and title_col in df.columns) else (id_col if id_col in df.columns else None),
                             labels={imp_col:"Impressions", "y":"CTR (%)"})
            st.plotly_chart(fig, use_container_width=True)
        st.caption("По документу: следить за стартовым CTR новых роликов.")
    else:
        need = []
        if not (imp_col and imp_col in df.columns): need.append("Impressions/Показы")
        if not (ctr_col and ctr_col in df.columns): need.append("CTR")
        warn_missing("CTR & Показы", need)

# 5) ПРОСЕДАЮТ (ниже медиан)
if show_underperf:
    st.subheader("Проседают относительно медианы (быстрый фильтр)")
    meds = {}
    if views_col and views_col in df.columns: meds["views"] = np.nanmedian(num(df[views_col]))
    if ctr_col and ctr_col in df.columns:     meds["ctr"]   = np.nanmedian(num(df[ctr_col]))
    if "AVD_sec" in df.columns and num(df["AVD_sec"]).notna().any(): 
        meds["avd"] = np.nanmedian(num(df["AVD_sec"]))

    if not meds:
        st.info("Недостаточно метрик для расчёта медиан.")
    else:
        bad = df.copy()
        if "avd" in meds:   bad["flag_avd"]   = num(bad["AVD_sec"]) < meds["avd"]
        if "ctr" in meds and ctr_col and ctr_col in bad.columns:
            bad["flag_ctr"] = num(bad[ctr_col]) < meds["ctr"]
        if "views" in meds and views_col and views_col in bad.columns:
            bad["flag_views"] = num(bad[views_col]) < meds["views"]

        mask = pd.Series(False, index=bad.index)
        for c in ["flag_avd","flag_ctr","flag_views"]:
            if c in bad.columns: mask = mask | bad[c].fillna(False)

        under = bad[mask]
        if under.empty:
            st.success("⛳ Всё выше или около медиан по выбранной выборке.")
        else:
            u = add_clickable(under, title_col, id_col, new_col="Видео")
            cols = ["Видео"] + [x for x in [id_col, views_col, ctr_col, "AVD_sec", "Avg_%_viewed", "YouTube Link"] 
                                if (x in u.columns) or (x=="YouTube Link")]
            human = {"AVD_sec":"AVD (сек)"}
            html_table(u.rename(columns=human), cols, escape=False)
            st.caption("Подсвечивает видео, которые ниже медианных значений (Views/CTR/AVD) — «подозреваемые» из документа.")

# ---------- Footer ----------
st.markdown("---")
st.caption("Если нужной колонки нет — соответствующий блок скрывается и показывает, что требуется. "
           "Названия видео кликабельны и ведут на YouTube.")
