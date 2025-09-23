# app.py — "Amigos Core" (один экран по твоему анализу)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import timedelta
import re, html

# -------------------- Страница --------------------
st.set_page_config(page_title="YouTube Dashboard — Amigos Core", layout="wide")
st.markdown("<h1 style='text-align:center'>📊 YouTube Dashboard — Amigos Core</h1>", unsafe_allow_html=True)
st.caption("Один экран: KPI, таблица, вклад по видео, качество (AVD/CTR), и воронка — строго по твоему анализу.")

# -------------------- Сайдбар --------------------
st.sidebar.header("⚙️ Данные")
file = st.sidebar.file_uploader("Загрузите CSV из YouTube Studio (Видео/Content)", type=["csv"])

st.sidebar.header("🎛 Фильтры")
n_videos = st.sidebar.slider("Сколько последних видео показывать:", 3, 300, 40)
search_q = st.sidebar.text_input("Поиск по названию/ID")
only_shorts = st.sidebar.checkbox("Только Shorts (<60 сек)", value=False)

st.sidebar.header("🧩 Блоки (вкл/выкл)")
show_kpi    = st.sidebar.checkbox("KPI-плашки", value=True)
show_table  = st.sidebar.checkbox("Таблица (кликабельные названия)", value=True)
show_bar    = st.sidebar.checkbox("Гистограмма: Просмотры по видео", value=True)
show_quality= st.sidebar.checkbox("Качество: AVD/Duration и CTR/Impressions", value=True)
show_funnel = st.sidebar.checkbox("Воронка (Impr → CTR → Views)", value=True)
show_insights = st.sidebar.checkbox("Мини-инсайты (Top/Under vs медиана)", value=True)

# -------------------- Хелперы --------------------
def norm(s:str)->str: return s.strip().lower()

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
    "watch_time_hours": ["watch time (hours)","время просмотра (часы)"],
    "unique_viewers": ["unique viewers","уникальные зрители"],
    "engaged_views": ["engaged views","заинтересованные просмотры"],
}

def find_col(df: pd.DataFrame, keys) -> str | None:
    if isinstance(keys,str): keys=[keys]
    by_norm = {norm(c): c for c in df.columns}
    for k in keys:
        nk = norm(k)
        if nk in by_norm: return by_norm[nk]
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
    if re.fullmatch(r"\d+(\.\d+)?", s):
        try: return float(s)
        except: return np.nan
    parts = s.split(":")
    try:
        parts = [int(p) for p in parts]
        if len(parts)==2: m,s = parts; return m*60+s
        if len(parts)==3: h,m,s = parts; return h*3600+m*60+s
        return np.nan
    except: return np.nan

def seconds_to_hhmmss(x):
    try:
        x = int(round(float(x)))
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
        st.info("Нет доступных колонок для отображения."); return
    st.markdown(df[use].to_html(index=False, escape=escape), unsafe_allow_html=True)

# -------------------- Основной поток --------------------
if not file:
    st.info("👆 Загрузите CSV. Подходит выгрузка «Видео/Content» из YouTube Studio (RU/EN — неважно).")
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
wth_col    = C["watch_time_hours"]
uv_col     = C["unique_viewers"]
eng_col    = C["engaged_views"]

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

# -------------------- UI: Один экран --------------------
# KPI
if show_kpi:
    st.subheader("KPI по выборке")
    k = st.columns(6)
    if views_col: k[0].metric("Views", f"{df[views_col].sum():,.0f}")
    if wth_col:   k[1].metric("Watch time (h)", f"{df[wth_col].sum():,.1f}")
    if rev_col:   k[2].metric("Revenue ($)", f"{df[rev_col].sum():,.2f}")
    if uv_col:    k[3].metric("Unique viewers", f"{df[uv_col].sum():,.0f}")
    if df["__avd_sec__"].notna().any(): k[4].metric("Avg AVD", seconds_to_hhmmss(df["__avd_sec__"].mean()))
    # Средний CTR (если есть)
    if ctr_col:
        ctr_vals = pd.to_numeric(df[ctr_col], errors="coerce")
        if ctr_vals.notna().any(): k[5].metric("CTR avg", f"{ctr_vals.mean():.2f}%")

# Таблица (кликабельные названия)
if show_table:
    st.subheader("Таблица видео (кликабельные названия → YouTube)")
    view = add_clickable_title(df, title_col, id_col, new_col="Видео")
    cols = ["Видео"] + [x for x in [
        id_col, views_col, imp_col, ctr_col, "__avd_sec__", "__duration_sec__", wth_col, rev_col, eng_col, uv_col, "YouTube Link"
    ] if (x in view.columns) or (x=="YouTube Link")]
    human = {"__avd_sec__":"AVD (сек)","__duration_sec__":"Длительность (сек)"}
    safe_table(view.rename(columns=human), cols, escape=False)

# Гистограмма «Просмотры по видео»
if show_bar and x_axis and views_col:
    st.subheader("Вклад по видео: Просмотры")
    fig = px.bar(df, x=x_axis, y=views_col, text=views_col)
    fig.update_traces(textposition="outside")
    fig.update_layout(xaxis_tickangle=-35, height=440, margin=dict(l=8,r=8,t=30,b=20))
    st.plotly_chart(fig, use_container_width=True)

# Качество: AVD/Duration и CTR/Impressions
if show_quality:
    st.subheader("Качество контента")
    c1, c2 = st.columns(2)

    # AVD vs Duration — всегда доступно, если есть обе метрики
    if df["__duration_sec__"].notna().any() and df["__avd_sec__"].notna().any():
        fig1 = px.scatter(
            df, x="__duration_sec__", y="__avd_sec__",
            hover_name=title_col if title_col else id_col,
            labels={"__duration_sec__":"Duration (сек)", "__avd_sec__":"AVD (сек)"}
        )
        fig1.update_layout(height=420)
        c1.plotly_chart(fig1, use_container_width=True)
    else:
        c1.info("Нет AVD/Duration в распознаваемом формате.")

    # CTR vs Impressions — только если есть Показы и CTR
    if ctr_col and imp_col:
        fig2 = px.scatter(
            df, x=imp_col, y=pd.to_numeric(df[ctr_col], errors="coerce"),
            hover_name=title_col if title_col else id_col,
            labels={imp_col:"Impressions", "y":"CTR (%)"}
        )
        fig2.update_layout(height=420)
        c2.plotly_chart(fig2, use_container_width=True)
    else:
        c2.info("Нет CTR/Impressions для сравнения.")

# Воронка (Impressions → CTR → Views)
if show_funnel and imp_col and ctr_col and views_col:
    st.subheader("Воронка: Показы → Кликов (CTR) → Просмотров")
    funnel_df = df[[x for x in [title_col, id_col, imp_col, ctr_col, views_col] if x]].copy()
    # приблизительная оценка кликов по CTR (если Studio не даёт клики напрямую)
    funnel_df["__clicks__"] = pd.to_numeric(funnel_df[ctr_col], errors="coerce").fillna(0) / 100.0 * pd.to_numeric(funnel_df[imp_col], errors="coerce").fillna(0)
    # нормировка для графика
    melt = funnel_df.melt(
        id_vars=[c for c in [title_col, id_col] if c],
        value_vars=[imp_col, "__clicks__", views_col],
        var_name="stage", value_name="value"
    )
    stage_names = {imp_col:"Impressions", "__clicks__":"Clicks (≈ Impr×CTR)", views_col:"Views"}
    melt["stage"] = melt["stage"].map(stage_names)
    # показываем «среднюю» воронку по выборке
    agg = melt.groupby("stage", as_index=False)["value"].sum()
    fig = px.funnel(agg, x="value", y="stage", title="Cуммарная воронка по выборке")
    st.plotly_chart(fig, use_container_width=True)

    # и мини-таблица по топ-роликам (по просмотрам)
    topN = st.slider("Сколько роликов показать в таблице воронки:", 5, 30, 10)
    top_f = funnel_df.sort_values(views_col, ascending=False).head(topN)
    top_f_view = add_clickable_title(top_f, title_col, id_col, new_col="Видео")
    cols_f = ["Видео"] + [c for c in [imp_col, ctr_col, "__clicks__", views_col, "YouTube Link"] if (c in top_f_view.columns) or (c=="YouTube Link")]
    safe_table(top_f_view.rename(columns={"__clicks__":"Clicks (≈)"}), cols_f, escape=False)
elif show_funnel:
    st.info("Для воронки нужны колонки «Показы/Impressions» и «CTR» + «Просмотры/Views» в файле.")

# Мини-инсайты (относительно медианы)
if show_insights:
    st.subheader("Мини-инсайты (vs медиана)")
    bullets = []
    # медианы
    med = {}
    if views_col: med["views"] = np.nanmedian(pd.to_numeric(df[views_col], errors="coerce"))
    if ctr_col:   med["ctr"]   = np.nanmedian(pd.to_numeric(df[ctr_col], errors="coerce"))
    if "__avd_sec__" in df.columns: med["avd"] = np.nanmedian(df["__avd_sec__"])
    # топ/низы по каждой метрике, если есть
    def top_under(series, n=3, largest=True):
        s = pd.to_numeric(series, errors="coerce")
        s = s.dropna()
        if s.empty: return pd.Index([])
        return s.nlargest(n).index if largest else s.nsmallest(n).index

    if views_col:
        idx_top = top_under(df[views_col], 3, True)
        idx_low = top_under(df[views_col], 3, False)
        if len(idx_top)>0:
            names = [shorten(df.iloc[i][title_col] if title_col else df.iloc[i][id_col], 40) for i in idx_top]
            bullets.append(f"🔼 Views: лидеры — {', '.join(names)}")
        if len(idx_low)>0:
            names = [shorten(df.iloc[i][title_col] if title_col else df.iloc[i][id_col], 40) for i in idx_low]
            bullets.append(f"🔽 Views: проседают — {', '.join(names)}")
    if ctr_col:
        s = pd.to_numeric(df[ctr_col], errors="coerce")
        idx_top = top_under(s, 3, True); idx_low = top_under(s, 3, False)
        if len(idx_top)>0:
            names = [shorten(df.iloc[i][title_col] if title_col else df.iloc[i][id_col], 40) for i in idx_top]
            bullets.append(f"✨ CTR: топ — {', '.join(names)}")
        if len(idx_low)>0:
            names = [shorten(df.iloc[i][title_col] if title_col else df.iloc[i][id_col], 40) for i in idx_low]
            bullets.append(f"⚠️ CTR: низ — {', '.join(names)}")
    if "__avd_sec__" in df.columns and df["__avd_sec__"].notna().any():
        s = df["__avd_sec__"]
        idx_top = top_under(s, 3, True); idx_low = top_under(s, 3, False)
        if len(idx_top)>0:
            names = [shorten(df.iloc[i][title_col] if title_col else df.iloc[i][id_col], 40) for i in idx_top]
            bullets.append(f"🕒 AVD: высокий — {', '.join(names)}")
        if len(idx_low)>0:
            names = [shorten(df.iloc[i][title_col] if title_col else df.iloc[i][id_col], 40) for i in idx_low]
            bullets.append(f"⏳ AVD: низкий — {', '.join(names)}")
    if bullets:
        for b in bullets: st.write("- " + b)
    else:
        st.info("Пока не удалось вычислить быстрые инсайты (нет подходящих колонок).")
