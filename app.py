# app.py — YouTube Dashboard (Amigos Edition)
# Клик по названиям в таблицах + (опционально) клик по графикам
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import timedelta
import re, html

# ====== опциональные зависимости ======
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

try:
    from streamlit_plotly_events import plotly_events
    PLOTLY_EVENTS_OK = True
except Exception:
    PLOTLY_EVENTS_OK = False

# -------------------- Настройки страницы --------------------
st.set_page_config(page_title="YouTube Dashboard 🚀", layout="wide")
st.markdown("<h1 style='text-align:center'>📊 YouTube Dashboard 🚀</h1>", unsafe_allow_html=True)
st.caption("Инструмент для быстрых ответов: где и почему падает, какие ролики тянут/умирают, как меняются AVD/CTR, и что делать дальше.")

# -------------------- Сайдбар --------------------
st.sidebar.header("⚙️ Данные")
file_main = st.sidebar.file_uploader("Загрузите основной CSV из YouTube Studio (Content/Видео)", type=["csv"])
file_ts   = st.sidebar.file_uploader("(Опционально) CSV с временными рядами (Views by Date)", type=["csv"])
file_queries = st.sidebar.file_uploader("(Опционально) CSV с запросами/vidIQ", type=["csv"])
n_videos = st.sidebar.slider("Сколько последних видео показывать:", 3, 500, 60)

st.sidebar.header("🎛 Фильтры (Content)")
search_q = st.sidebar.text_input("Поиск по названию/ID")
min_dur = st.sidebar.number_input("Мин. длительность (сек)", 0, 24*3600, 0)
max_dur = st.sidebar.number_input("Макс. длительность (сек)", 0, 24*3600, 24*3600)
flag_shorts_only = st.sidebar.checkbox("Только Shorts (<60 сек)", value=False)

st.sidebar.header("⚠️ Алерты")
thr_ctr = st.sidebar.number_input("CTR ниже (%) → флаг", 0.0, 100.0, 3.0, step=0.1)
thr_avd = st.sidebar.number_input("AVD ниже (сек) → флаг", 0.0, 24*3600.0, 600.0, step=10.0)

# -------------------- Утилиты --------------------
METRICS_MAP = {
    "views": ["views", "просмотры"],
    "impressions": ["impressions", "показы"],
    "ctr": ["impressions click-through rate", "ctr", "impressions click-through rate (%)", "ctr для значков видео"],
    "avd": ["average view duration", "средняя продолжительность просмотра"],
    "duration": ["duration", "продолжительность", "длительность"],
    "revenue": ["estimated partner revenue", "расчетный доход", "расчётный доход"],
    "rpm": ["rpm", "доход за 1000 показов", "доход на тысячу просмотров"],
    "subs": ["subscribers", "подписчики"],
    "watch_time_hours": ["watch time (hours)", "время просмотра (часы)"],
    "publish_time": ["video publish time", "publish time", "время публикации видео", "дата публикации"],
    "title": ["название видео", "title", "video title", "название"],
    "video_id": ["video id", "external video id", "контент", "content", "id видео", "ид видео"]
}

def norm(s:str)->str: return s.strip().lower()

def find_col(df: pd.DataFrame, keys) -> str | None:
    if df is None or df.empty: return None
    if isinstance(keys, str): keys = [keys]
    cols_norm = {norm(c): c for c in df.columns}
    for k in keys:
        nk = norm(k)
        if nk in cols_norm: return cols_norm[nk]
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
    if re.fullmatch(r"\d+(\.\d+)?", s): return float(s)
    parts = s.split(":")
    try:
        parts = [int(p) for p in parts]
        if len(parts)==2:
            m, s = parts; return m*60 + s
        if len(parts)==3:
            h, m, s = parts; return h*3600 + m*60 + s
        return np.nan
    except Exception:
        return np.nan

def seconds_to_hhmmss(x):
    try:
        x = int(round(float(x))); return str(timedelta(seconds=x))
    except Exception:
        return ""

def shorten(text:str, n:int=38)->str:
    t = "" if text is None else str(text)
    return (t[:n]+"…") if len(t)>n else t

def add_clickable_title_column(df: pd.DataFrame, title_col: str | None, id_col: str | None, new_col="Видео"):
    out = df.copy()
    if id_col is None or id_col not in out.columns: return out
    titles = out[title_col] if (title_col and title_col in out.columns) else out[id_col].astype(str)
    urls = "https://www.youtube.com/watch?v=" + out[id_col].astype(str)
    out[new_col] = [f"<a href='{u}' target='_blank'>{html.escape(str(t))}</a>" for t,u in zip(titles, urls)]
    return out

def render_html_table(df: pd.DataFrame, columns: list[str], escape: bool=False):
    safe_cols = [c for c in columns if c in df.columns]
    if not safe_cols:
        st.info("Нет колонок для отображения."); return
    st.markdown(df[safe_cols].to_html(index=False, escape=escape), unsafe_allow_html=True)

def plot_bar_clickable(df: pd.DataFrame, x: str, y: str, id_col: str | None):
    fig = px.bar(df, x=x, y=y, text=y)
    if id_col and id_col in df.columns:
        urls = "https://www.youtube.com/watch?v=" + df[id_col].astype(str)
        fig.update_traces(customdata=np.stack([urls], axis=-1),
                          hovertemplate="<b>%{x}</b><br>"+y+": %{y}<br>URL: %{customdata[0]}<extra></extra>")
    fig.update_traces(textposition="outside")
    fig.update_layout(xaxis_tickangle=-35, height=420, margin=dict(l=8, r=8, t=30, b=10))

    if PLOTLY_EVENTS_OK:
        selected = plotly_events(fig, click_event=True, hover_event=False, select_event=False,
                                 override_height=420, override_width="100%")
        st.plotly_chart(fig, use_container_width=True)
        if selected and id_col in df.columns:
            idx = selected[0].get("pointIndex")
            if idx is not None and 0<=idx<len(df):
                url = f"https://www.youtube.com/watch?v={df.iloc[idx][id_col]}"
                st.link_button("🔗 Открыть выбранное видео", url, use_container_width=True)
    else:
        st.plotly_chart(fig, use_container_width=True)
        if id_col and id_col in df.columns:
            st.caption("Подсказка: установи `streamlit-plotly-events`, чтобы открывать ролик кликом по столбику.")

# -------------------- Загрузка --------------------
if not file_main:
    st.info("👆 Загрузите основной CSV из YouTube Studio, затем появятся вкладки анализа.")
    st.stop()

df = pd.read_csv(file_main)
df.columns = [c.strip() for c in df.columns]

C = detect_cols(df)
title_col = C["title"]; id_col = C["video_id"]; views_col = C["views"]; imp_col = C["impressions"]
ctr_col = C["ctr"]; avd_col = C["avd"]; dur_col = C["duration"]; rev_col = C["revenue"]; rpm_col = C["rpm"]
subs_col = C["subs"]; wth_col = C["watch_time_hours"]; pub_col = C["publish_time"]

# даты + сортировка
if pub_col:
    df[pub_col] = pd.to_datetime(df[pub_col], errors="coerce")
    df = df.sort_values(pub_col, ascending=False)

# длительность/AVD
df["__duration_sec__"] = df[dur_col].apply(parse_duration_to_seconds) if dur_col else np.nan
df["__avd_sec__"] = df[avd_col].apply(parse_duration_to_seconds) if avd_col else np.nan

# производные
if (rpm_col is None) and (rev_col and views_col):
    df["__RPM__"] = df[rev_col] / df[views_col].replace(0,np.nan) * 1000.0
else:
    df["__RPM__"] = df[rpm_col] if rpm_col else np.nan

df["__efficiency__"] = df[views_col] / df[imp_col].replace(0,np.nan) if (imp_col and views_col) else np.nan
df["__avg_percent_viewed__"] = np.where(
    (df["__avd_sec__"].notna()) & (df["__duration_sec__"].replace(0,np.nan).notna()),
    df["__avd_sec__"]/df["__duration_sec__"]*100.0, np.nan
)

# ограничение по N
df = df.head(n_videos).copy()

# ссылки и короткие имена
if id_col:
    df["YouTube Link"] = df[id_col].apply(lambda x: f"https://www.youtube.com/watch?v={x}")
if title_col:
    df["__title_short__"] = df[title_col].apply(lambda x: shorten(x, 36))
x_axis = "__title_short__" if title_col else (id_col if id_col else None)

# Подготовка TS (опционально)
ts_df = None
if file_ts is not None:
    ts_df = pd.read_csv(file_ts)
    ts_df.columns = [c.strip() for c in ts_df.columns]
    ts_vid = find_col(ts_df, ["video id","external video id","content","контент"])
    ts_date = find_col(ts_df, ["date","дата"])
    ts_views = find_col(ts_df, ["views","просмотры"])
    if ts_vid and ts_date and ts_views:
        ts_df[ts_date] = pd.to_datetime(ts_df[ts_date], errors="coerce")
    else:
        ts_df = None

# ================================== ТАБЫ ==================================
tabs = st.tabs([
    "Overview", "Content", "Year Mix", "Locomotives & Decay",
    "Quality Trends", "Overlay Compare",
    "CTR & Thumbnails", "Retention", "Traffic & SEO", "Monetization", "Cadence", "Shorts vs Longs", "Alerts"
])

# -------- Overview --------
with tabs[0]:
    st.subheader("KPI за выборку")
    cols = st.columns(6)
    if views_col: cols[0].metric("Views", f"{df[views_col].sum():,.0f}")
    if wth_col:   cols[1].metric("Watch time (h)", f"{df[wth_col].sum():,.1f}")
    if subs_col:  cols[2].metric("Subs", f"{df[subs_col].sum():,.0f}")
    if rev_col:   cols[3].metric("Revenue ($)", f"{df[rev_col].sum():,.2f}")
    if df["__RPM__"].notna().any(): cols[4].metric("RPM", f"{df['__RPM__'].mean():,.2f}")
    if df["__avd_sec__"].notna().any(): cols[5].metric("Avg AVD", seconds_to_hhmmss(df['__avd_sec__'].mean()))

    st.markdown("### Просмотры по видео")
    if x_axis and views_col:
        plot_bar_clickable(df, x=x_axis, y=views_col, id_col=id_col)

# -------- Content --------
with tabs[1]:
    st.subheader("Таблица контента + фильтры")
    view_df = df.copy()
    if search_q:
        pool = [c for c in [title_col, id_col] if c]
        if pool:
            view_df = view_df[view_df[pool].astype(str).apply(lambda r: search_q.lower() in " ".join(r).lower(), axis=1)]
    if flag_shorts_only:
        view_df = view_df[view_df["__duration_sec__"] < 60]
    else:
        view_df = view_df[(view_df["__duration_sec__"].isna()) | ((view_df["__duration_sec__"] >= min_dur) & (view_df["__duration_sec__"] <= max_dur))]

    view_df_click = add_clickable_title_column(view_df, title_col, id_col, new_col="Видео")
    base_cols = [id_col, views_col, imp_col, ctr_col, subs_col, rev_col, "__RPM__", "__avd_sec__", "__duration_sec__", "__avg_percent_viewed__", "__efficiency__", "YouTube Link"]
    show_cols = ["Видео"] + [c for c in base_cols if (c in view_df_click.columns) or (c == "YouTube Link")]

    human = {"__RPM__": "RPM", "__avd_sec__":"AVD (сек)", "__duration_sec__":"Длительность (сек)", "__avg_percent_viewed__":"Avg % viewed", "__efficiency__":"Views/Impr."}
    df_print = view_df_click.rename(columns=human).copy()
    render_html_table(df_print, show_cols, escape=False)

    st.markdown("### Кластера тем (по названию)")
    if SKLEARN_OK and title_col:
        k = st.slider("Количество кластеров", 2, 12, 6)
        try:
            X = TfidfVectorizer(max_features=2000, ngram_range=(1,2)).fit_transform(view_df[title_col].fillna(""))
            km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
            view_df["__cluster__"] = km.labels__
            fig = px.scatter(view_df.reset_index(drop=True), x=view_df.index, y="__cluster__", hover_name=title_col, color="__cluster__")
            fig.update_layout(height=360, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.info(f"Кластеризация не удалась: {e}")
    else:
        st.caption("Для кластеров установи scikit-learn.")

# -------- Year Mix --------
with tabs[2]:
    st.subheader("Микс по годам: вклад старых/новых")
    if pub_col and views_col:
        tmp = df[[pub_col, views_col, id_col, title_col]].copy()
        tmp["year"] = tmp[pub_col].dt.year
        by_year_count = tmp.groupby("year", dropna=True)[id_col].count().rename("Количество видео")
        by_year_views = tmp.groupby("year", dropna=True)[views_col].sum().rename("Просмотры")

        c1, c2 = st.columns(2)
        c1.plotly_chart(px.bar(by_year_count.reset_index(), x="year", y="Количество видео", title="Количество видео по годам"), use_container_width=True)
        c2.plotly_chart(px.bar(by_year_views.reset_index(), x="year", y="Просмотры", title="Просмотры по годам"), use_container_width=True)

        st.caption("Сравнивай долю по количеству и по просмотрам — видно, когда старые роли дают непропорционально много трафика.")
    else:
        st.info("Нужны дата публикации и просмотры.")

# -------- Locomotives & Decay --------
with tabs[3]:
    st.subheader("Локомотивы и «умирание» трафика")
    if views_col:
        top_k = st.slider("Сколько топ-роликов показать:", 5, 50, 15)
        top_df = df.sort_values(views_col, ascending=False).head(top_k).copy()
        top_click = add_clickable_title_column(top_df, title_col, id_col, new_col="Видео")
        cols_top = ["Видео"] + [c for c in [id_col, views_col, "__avg_percent_viewed__", ctr_col, "__efficiency__", "YouTube Link"] if (c in top_click.columns) or (c=="YouTube Link")]
        st.write("**ТОП ролики по просмотрам**")
        render_html_table(top_click, cols_top, escape=False)

        # Если есть временной ряд — покажем decay-кривые
        if ts_df is not None:
            ts_vid = find_col(ts_df, ["video id","external video id","content","контент"])
            ts_date = find_col(ts_df, ["date","дата"])
            ts_views = find_col(ts_df, ["views","просмотры"])
            vids = st.multiselect("Выбери ролики для наложения (до 5):",
                                  options=top_df[id_col].tolist() if id_col else [],
                                  default=(top_df[id_col].tolist()[:3] if id_col else []))
            if vids and ts_vid and ts_date and ts_views:
                for v in vids[:5]:
                    one = ts_df[ts_df[ts_vid]==v].sort_values(ts_date)
                    name = df.loc[df[id_col]==v, title_col].iloc[0] if (id_col and title_col and v in df[id_col].values) else v
                    fig = px.line(one, x=ts_date, y=ts_views, title=shorten(name, 70))
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Чтобы увидеть decay-кривые, нужен файл с временным рядом (Date, Views, Video ID).")
        else:
            st.caption("Загрузи CSV с дневными/месячными просмотрами, чтобы рисовать decay-кривые во времени.")
    else:
        st.info("Нужна колонка с просмотрами.")

# -------- Quality Trends (AVD/CTR) --------
with tabs[4]:
    st.subheader("Тренды качества: AVD и CTR")
    if pub_col and (df["__avd_sec__"].notna().any() or ctr_col):
        tmp = df.copy()
        tmp["date"] = tmp[pub_col].dt.date
        tmp = tmp.sort_values(pub_col)
        win = st.slider("Окно скользящего среднего (публикаций):", 3, 50, 9)

        c1, c2 = st.columns(2)
        if df["__avd_sec__"].notna().any():
            s = pd.Series(tmp["__avd_sec__"].values).rolling(win, min_periods=1).mean()
            fig = px.line(x=list(range(len(s))), y=s, labels={"x":"публикации (порядок)", "y":"AVD (сек)"}, title="Скользящее среднее AVD")
            c1.plotly_chart(fig, use_container_width=True)
        if ctr_col:
            s = pd.Series(pd.to_numeric(tmp[ctr_col], errors="coerce").values).rolling(win, min_periods=1).mean()
            fig = px.line(x=list(range(len(s))), y=s, labels={"x":"публикации (порядок)", "y":"CTR (%)"}, title="Скользящее среднее CTR")
            c2.plotly_chart(fig, use_container_width=True)
        st.caption("Падает ли AVD в новых релизах? Как ведёт себя стартовый CTR — можно быстро отследить тренд.")
    else:
        st.info("Нужна дата публикации и AVD/CTR.")

# -------- Overlay Compare --------
with tabs[5]:
    st.subheader("Сравнение роликов (наложение)")
    if id_col:
        pick = st.multiselect("Выбери видео:", options=df[id_col].tolist(), default=df[id_col].tolist()[:3])
        metric = st.selectbox("Метрика для сравнения:", [m for m in [views_col, ctr_col, "__avd_sec__", "__avg_percent_viewed__", "__RPM__"] if m], index=0)
        if pick and metric:
            sub = df[df[id_col].isin(pick)].copy()
            sub["name"] = sub[title_col].apply(lambda x: shorten(x, 50)) if title_col else sub[id_col].astype(str)
            fig = px.bar(sub, x="name", y=metric, text=metric)
            fig.update_traces(textposition="outside"); fig.update_layout(xaxis_tickangle=-35, height=420)
            st.plotly_chart(fig, use_container_width=True)
        st.caption("Для временного наложения загрузите timeseries CSV (Date, Views, Video ID) и используйте вкладку Locomotives & Decay.")
    else:
        st.info("Нужна колонка Video ID.")

# -------- CTR & Thumbnails --------
with tabs[6]:
    st.subheader("CTR & Thumbnails")
    if ctr_col and imp_col:
        fig = px.scatter(df, x=imp_col, y=ctr_col, size=views_col if views_col else None,
                         color=title_col if title_col else id_col, hover_data=[id_col] if id_col else None)
        if id_col and id_col in df.columns:
            urls = "https://www.youtube.com/watch?v=" + df[id_col].astype(str)
            fig.update_traces(customdata=np.stack([urls], axis=-1),
                              hovertemplate="<b>%{y}% CTR</b><br>Impr: %{x}<br>URL: %{customdata[0]}<extra></extra>")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Нет CTR/Impressions.")

# -------- Retention --------
with tabs[7]:
    st.subheader("Retention / Удержание")
    if df["__duration_sec__"].notna().any() and df["__avd_sec__"].notna().any():
        c1, c2 = st.columns(2)
        fig = px.scatter(df, x="__duration_sec__", y="__avd_sec__", color=title_col if title_col else id_col,
                         hover_data=[id_col] if id_col else None)
        if id_col and id_col in df.columns:
            urls = "https://www.youtube.com/watch?v=" + df[id_col].astype(str)
            fig.update_traces(customdata=np.stack([urls], axis=-1),
                              hovertemplate="Dur: %{x}s<br>AVD: %{y}s<br>URL: %{customdata[0]}<extra></extra>")
        fig.update_layout(height=420); c1.plotly_chart(fig, use_container_width=True)

        fig2 = px.scatter(df, x="__duration_sec__", y="__avg_percent_viewed__", color=title_col if title_col else id_col,
                          hover_data=[id_col] if id_col else None)
        if id_col and id_col in df.columns:
            urls = "https://www.youtube.com/watch?v=" + df[id_col].astype(str)
            fig2.update_traces(customdata=np.stack([urls], axis=-1),
                               hovertemplate="Dur: %{x}s<br>Avg %: %{y:.1f}%<br>URL: %{customdata[0]}<extra></extra>")
        fig2.update_layout(height=420); c2.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Нет AVD/Длительности.")

# -------- Traffic & SEO --------
with tabs[8]:
    st.subheader("Traffic & SEO")
    if file_queries is not None:
        qdf = pd.read_csv(file_queries); qdf.columns = [c.strip() for c in qdf.columns]
        q_query = find_col(qdf, ["query","запрос"]); q_views = find_col(qdf, ["views","просмотры"]); q_impr = find_col(qdf, ["impressions","показы"])
        show = [c for c in [q_query, q_views, q_impr] if c]
        if show: st.dataframe(qdf[show].head(200), use_container_width=True)
        else: st.info("Не распознаны колонки Query/Views/Impressions.")
    else:
        st.caption("Загрузи CSV с запросами (или vidIQ-экспорт), чтобы дополнить SEO-картину.")

# -------- Monetization --------
with tabs[9]:
    st.subheader("Monetization")
    if rev_col and x_axis: plot_bar_clickable(df, x=x_axis, y=rev_col, id_col=id_col)
    else: st.info("Нет Revenue или оси X.")
    if df["__RPM__"].notna().any() and x_axis: plot_bar_clickable(df, x=x_axis, y="__RPM__", id_col=id_col)
    else: st.info("Нет RPM.")

# -------- Cadence --------
with tabs[10]:
    st.subheader("Календарь публикаций")
    if pub_col and views_col:
        tmp = df[[pub_col, views_col]].dropna().copy()
        tmp["weekday"] = tmp[pub_col].dt.day_name(); tmp["hour"] = tmp[pub_col].dt.hour
        heat = tmp.pivot_table(index="weekday", columns="hour", values=views_col, aggfunc="mean").fillna(0)
        st.dataframe(heat.style.format("{:,.0f}"), use_container_width=True)
    else:
        st.info("Нужны дата публикации и просмотры.")

# -------- Shorts vs Longs --------
with tabs[11]:
    st.subheader("Shorts vs Longs")
    if "__duration_sec__" in df.columns:
        shorts = df[df["__duration_sec__"] < 60]; longs = df[df["__duration_sec__"] >= 60]
        shorts_click = add_clickable_title_column(shorts, title_col, id_col, new_col="Видео")
        longs_click  = add_clickable_title_column(longs,  title_col, id_col, new_col="Видео")
        base = [id_col, views_col, ctr_col, "__avd_sec__", "__avg_percent_viewed__", "YouTube Link"]
        cols_s = ["Видео"] + [c for c in base if (c in shorts_click.columns) or (c=="YouTube Link")]
        cols_l = ["Видео"] + [c for c in base if (c in longs_click.columns)  or (c=="YouTube Link")]
        c1, c2 = st.columns(2)
        c1.write(f"Shorts ({len(shorts)})"); render_html_table(shorts_click, cols_s, escape=False)
        c2.write(f"Longs ({len(longs)})");  render_html_table(longs_click,  cols_l, escape=False)
    else:
        st.info("Нет длительности — нельзя разделить Shorts/Longs.")

# -------- Alerts --------
with tabs[12]:
    st.subheader("Алерты (ниже порога)")
    issues = []
    if ctr_col:
        bad_ctr = df[pd.to_numeric(df[ctr_col], errors="coerce") < thr_ctr]
        if not bad_ctr.empty: issues.append(("CTR ниже порога", bad_ctr))
    if df["__avd_sec__"].notna().any():
        bad_avd = df[df["__avd_sec__"] < thr_avd]
        if not bad_avd.empty: issues.append(("AVD ниже порога", bad_avd))

    if issues:
        for title, d in issues:
            click = add_clickable_title_column(d, title_col, id_col, new_col="Видео")
            cols = ["Видео"] + [c for c in [id_col, ctr_col, "__avd_sec__", "__avg_percent_viewed__", "YouTube Link"] if (c in click.columns) or (c=="YouTube Link")]
            st.write(f"**{title}: {len(d)} видео**")
            render_html_table(click, cols, escape=False)
    else:
        st.success("Проблемных видео не найдено по заданным порогам.")
