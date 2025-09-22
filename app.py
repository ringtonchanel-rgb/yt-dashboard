# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import timedelta
import re

# ======== Безопасный импорт sklearn (не обязателен) ========
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

# -------------------- Страница --------------------
st.set_page_config(page_title="YouTube Dashboard 🚀", layout="wide")
st.markdown("<h1 style='text-align:center'>📊 YouTube Dashboard 🚀</h1>", unsafe_allow_html=True)
st.write("Аналитика YouTube-канала: просмотры, CTR, удержание, трафик, доход и другие ключевые метрики.")

# -------------------- Сайдбар ---------------------
st.sidebar.header("⚙️ Данные")
file_main = st.sidebar.file_uploader("Загрузите основной CSV из YouTube Studio", type=["csv"])
file_queries = st.sidebar.file_uploader("(Опционально) CSV с поисковыми запросами / vidIQ", type=["csv"])
n_videos = st.sidebar.slider("Сколько последних видео показывать:", 3, 200, 30)

st.sidebar.header("🎛 Фильтры (на вкладке Content)")
search_q = st.sidebar.text_input("Поиск по названию/ID")
min_dur = st.sidebar.number_input("Мин. длительность (сек)", 0, 24*3600, 0)
max_dur = st.sidebar.number_input("Макс. длительность (сек)", 0, 24*3600, 24*3600)
flag_shorts_only = st.sidebar.checkbox("Только Shorts (<60 сек)", value=False)

st.sidebar.header("⚠️ Алерты")
thr_ctr = st.sidebar.number_input("CTR ниже (%) → флаг", 0.0, 100.0, 3.0, step=0.1)
thr_avd = st.sidebar.number_input("AVD ниже (сек) → флаг", 0.0, 24*3600.0, 60.0, step=1.0)

# -------------------- Словари и утилиты ---------------------
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

def norm(s: str) -> str:
    return s.strip().lower()

def find_col(df: pd.DataFrame, keys) -> str | None:
    if df is None or df.empty:
        return None
    if isinstance(keys, str):
        keys = [keys]
    # точное совпадение (приведённое к lower)
    cols_norm = {norm(c): c for c in df.columns}
    for k in keys:
        nk = norm(k)
        if nk in cols_norm:
            return cols_norm[nk]
    # contains
    for k in keys:
        nk = norm(k)
        for c in df.columns:
            if nk in norm(c):
                return c
    return None

def detect_cols(df: pd.DataFrame):
    cols = {k: find_col(df, v) for k, v in METRICS_MAP.items()}
    return cols

def parse_duration_to_seconds(x):
    """Поддержка 'MM:SS', 'HH:MM:SS' либо чисел/строк с секундами."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if re.fullmatch(r"\d+(\.\d+)?", s):
        return float(s)
    parts = s.split(":")
    try:
        parts = [int(p) for p in parts]
        if len(parts) == 2:
            m, s = parts
            return m*60 + s
        elif len(parts) == 3:
            h, m, s = parts
            return h*3600 + m*60 + s
        else:
            return np.nan
    except Exception:
        return np.nan

def seconds_to_hhmmss(x):
    try:
        x = int(round(float(x)))
        return str(timedelta(seconds=x))
    except Exception:
        return ""

def shorten(text: str, n: int = 40) -> str:
    t = str(text) if text is not None else ""
    return (t[:n]+"…") if len(t) > n else t

# -------------------- Загрузка данных ---------------------
if file_main:
    df = pd.read_csv(file_main)
    df.columns = [c.strip() for c in df.columns]

    C = detect_cols(df)
    title_col   = C["title"]
    id_col      = C["video_id"]
    views_col   = C["views"]
    imp_col     = C["impressions"]
    ctr_col     = C["ctr"]
    avd_col     = C["avd"]
    dur_col     = C["duration"]
    rev_col     = C["revenue"]
    rpm_col     = C["rpm"]
    subs_col    = C["subs"]
    wth_col     = C["watch_time_hours"]
    pub_col     = C["publish_time"]

    # Приводим дату публикации и сортируем
    if pub_col:
        df[pub_col] = pd.to_datetime(df[pub_col], errors="coerce")
        df = df.sort_values(pub_col, ascending=False)

    # Приведём длительность к сек, AVD тоже (если форматы как HH:MM:SS)
    if dur_col:
        df["__duration_sec__"] = df[dur_col].apply(parse_duration_to_seconds)
    else:
        df["__duration_sec__"] = np.nan
    if avd_col:
        df["__avd_sec__"] = df[avd_col].apply(parse_duration_to_seconds)
    else:
        df["__avd_sec__"] = np.nan

    # Производные метрики
    if (rpm_col is None) and (rev_col and views_col):
        df["__RPM__"] = df[rev_col] / df[views_col].replace(0, np.nan) * 1000.0
    else:
        df["__RPM__"] = df[rpm_col] if rpm_col else np.nan

    if (imp_col and views_col):
        df["__efficiency__"] = df[views_col] / df[imp_col].replace(0, np.nan)  # Views / Impressions
    else:
        df["__efficiency__"] = np.nan

    if (df["__avd_sec__"].notna().any() and df["__duration_sec__"].notna().any()):
        df["__avg_percent_viewed__"] = (df["__avd_sec__"] / df["__duration_sec__"].replace(0, np.nan) * 100.0)
    else:
        df["__avg_percent_viewed__"] = np.nan

    # ограничение по количеству видео
    df = df.head(n_videos).copy()

    # Кликабельные ссылки
    if id_col:
        df["YouTube Link"] = df[id_col].apply(lambda x: f"https://www.youtube.com/watch?v={x}")

    # Короткое имя для оси X
    if title_col:
        df["__title_short__"] = df[title_col].apply(lambda x: shorten(x, 38))
        x_axis = "__title_short__"
    elif id_col:
        x_axis = id_col
    else:
        x_axis = None

    # ==============================================
    #              ВКЛАДКИ / DASH
    # ==============================================
    tab_overview, tab_content, tab_ctr, tab_ret, tab_traffic, tab_money, tab_cadence, tab_split, tab_alerts = st.tabs(
        ["Overview", "Content", "CTR & Thumbnails", "Retention", "Traffic & SEO", "Monetization", "Cadence", "Shorts vs Longs", "Alerts"]
    )

    # -------- Overview --------
    with tab_overview:
        st.subheader("KPI за выборку")
        cols = st.columns(6)
        if views_col: cols[0].metric("Views", f"{df[views_col].sum():,.0f}")
        if wth_col:   cols[1].metric("Watch time (h)", f"{df[wth_col].sum():,.1f}")
        if subs_col:  cols[2].metric("Subs", f"{df[subs_col].sum():,.0f}")
        if rev_col:   cols[3].metric("Revenue ($)", f"{df[rev_col].sum():,.2f}")
        # средние
        if df["__RPM__"].notna().any(): cols[4].metric("RPM", f"{df['__RPM__'].mean():,.2f}")
        if df["__avd_sec__"].notna().any(): cols[5].metric("Avg AVD", seconds_to_hhmmss(df['__avd_sec__'].mean()))

        st.markdown("### ТОП-5 и Андер-5 (по Views / при отсутствии — по Revenue)")
        base_metric = views_col or rev_col
        if base_metric:
            top5 = df.sort_values(base_metric, ascending=False).head(5)
            low5 = df.sort_values(base_metric, ascending=True).head(5)
            c1, c2 = st.columns(2)
            show_cols = [c for c in [title_col, id_col, base_metric, "YouTube Link"] if c in (df.columns.tolist()+['YouTube Link'])]
            c1.write("**ТОП-5**")
            c1.dataframe(top5[[c for c in show_cols if c in top5.columns or c == "YouTube Link"]], use_container_width=True)
            c2.write("**Андер-5**")
            c2.dataframe(low5[[c for c in show_cols if c in low5.columns or c == "YouTube Link"]], use_container_width=True)
        else:
            st.info("Нет базовой метрики для ТОП/Андер.")

        if x_axis and views_col:
            st.markdown("### Просмотры по видео")
            fig = px.bar(df, x=x_axis, y=views_col, text=views_col)
            fig.update_traces(textposition="outside")
            fig.update_layout(xaxis_tickangle=-35, height=460, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)

    # -------- Content (фильтры, таблица, темы) --------
    with tab_content:
        st.subheader("Таблица контента + фильтры")
        view_df = df.copy()
        # фильтры
        if search_q:
            pool = [c for c in [title_col, id_col] if c]
            if pool:
                view_df = view_df[view_df[pool].astype(str).apply(
                    lambda r: search_q.lower() in " ".join(r).lower(), axis=1
                )]
        if flag_shorts_only:
            view_df = view_df[view_df["__duration_sec__"] < 60]
        else:
            view_df = view_df[(view_df["__duration_sec__"].isna()) | ((view_df["__duration_sec__"] >= min_dur) & (view_df["__duration_sec__"] <= max_dur))]

        show_cols = [c for c in [title_col, id_col, views_col, imp_col, ctr_col, subs_col, rev_col, "__RPM__", "__avd_sec__", "__duration_sec__", "__avg_percent_viewed__", "__efficiency__", "YouTube Link"] if (c in view_df.columns) or (c in ["YouTube Link"])]
        # человекочитаемые колонки
        human_names = {
            "__RPM__": "RPM",
            "__avd_sec__": "AVD (сек)",
            "__duration_sec__": "Длительность (сек)",
            "__avg_percent_viewed__": "Avg % viewed",
            "__efficiency__": "Efficiency (Views/Impr.)"
        }
        df_print = view_df[show_cols].rename(columns=human_names)
        st.dataframe(df_print, use_container_width=True)

        # Кластера тем (опционально)
        st.markdown("### Кластера тем (по названию)")
        if SKLEARN_OK and title_col:
            k = st.slider("Количество кластеров", 2, 12, 5)
            try:
                X = TfidfVectorizer(max_features=2000, ngram_range=(1,2)).fit_transform(view_df[title_col].fillna(""))
                km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
                view_df["__cluster__"] = km.labels_
                fig = px.scatter(view_df.reset_index(), x=view_df.index, y="__cluster__", hover_name=title_col, color="__cluster__")
                fig.update_layout(height=380, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Прим.: позиция по оси X — условная (индекс), кластеризация — черновая TF-IDF/KMeans.")
            except Exception as e:
                st.info(f"Кластеризация не удалась: {e}")
        else:
            st.info("Для кластеров установи scikit-learn (или загрузка без кластеризации).")

    # -------- CTR & Thumbnails --------
    with tab_ctr:
        st.subheader("CTR & Thumbnails")
        if ctr_col and imp_col:
            fig = px.scatter(df, x=imp_col, y=ctr_col, size=views_col if views_col else None,
                             color=title_col if title_col else id_col, hover_data=[id_col] if id_col else None)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Нет колонок CTR/Impressions для сравнения.")

        if ctr_col and views_col and x_axis:
            fig = px.bar(df, x=x_axis, y=ctr_col, text=ctr_col, hover_data=[views_col] if views_col else None)
            fig.update_traces(textposition="outside")
            fig.update_layout(xaxis_tickangle=-35, height=420)
            st.plotly_chart(fig, use_container_width=True)

    # -------- Retention --------
    with tab_ret:
        st.subheader("Retention / Удержание")
        if df["__duration_sec__"].notna().any() and df["__avd_sec__"].notna().any():
            c1, c2 = st.columns(2)
            fig = px.scatter(df, x="__duration_sec__", y="__avd_sec__", color=title_col if title_col else id_col,
                             hover_data=[id_col] if id_col else None)
            fig.update_layout(height=420)
            c1.plotly_chart(fig, use_container_width=True)

            fig2 = px.scatter(df, x="__duration_sec__", y="__avg_percent_viewed__", color=title_col if title_col else id_col,
                              hover_data=[id_col] if id_col else None)
            fig2.update_layout(height=420)
            c2.plotly_chart(fig2, use_container_width=True)

            st.caption("Левый график: длительность vs AVD (сек). Правый: длительность vs средний % досмотра.")
        else:
            st.info("Нет данных для AVD/Длительности (или не распознаны форматы).")

    # -------- Traffic & SEO --------
    with tab_traffic:
        st.subheader("Traffic & SEO")
        if file_queries is not None:
            qdf = pd.read_csv(file_queries)
            qdf.columns = [c.strip() for c in qdf.columns]
            # Пробуем найти названия
            q_query = find_col(qdf, ["query", "запрос"])
            q_views = find_col(qdf, ["views", "просмотры"])
            q_impr  = find_col(qdf, ["impressions", "показы"])

            st.write("Данные из загрузки (queries/vidIQ):")
            show_q = [c for c in [q_query, q_views, q_impr] if c]
            if show_q:
                st.dataframe(qdf[show_q].head(100), use_container_width=True)
            else:
                st.info("Не распознаны колонки в queries-файле (ищу: Query/Views/Impressions).")
        else:
            st.info("Загрузи CSV с поисковыми запросами (или vidIQ), чтобы показать SEO-блок.")

        st.markdown("**Идеи:** подключить YouTube Data API + Google Trends для генерации тем/ключей (можно добавить в следующую версию).")

    # -------- Monetization --------
    with tab_money:
        st.subheader("Monetization")
        if rev_col and x_axis:
            fig = px.bar(df, x=x_axis, y=rev_col, text=rev_col)
            fig.update_traces(textposition="outside")
            fig.update_layout(xaxis_tickangle=-35, height=420)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Нет Revenue или оси X.")

        if df["__RPM__"].notna().any() and x_axis:
            fig = px.bar(df, x=x_axis, y="__RPM__", text="__RPM__")
            fig.update_traces(textposition="outside")
            fig.update_layout(xaxis_tickangle=-35, height=420)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Нет RPM (или рассчитать не удалось).")

    # -------- Cadence --------
    with tab_cadence:
        st.subheader("Календарь публикаций / «окна»")
        if pub_col and views_col:
            tmp = df[[pub_col, views_col]].dropna().copy()
            tmp["weekday"] = tmp[pub_col].dt.day_name()
            tmp["hour"] = tmp[pub_col].dt.hour
            heat = tmp.pivot_table(index="weekday", columns="hour", values=views_col, aggfunc="mean").fillna(0)
            st.dataframe(heat.style.format("{:,.0f}"), use_container_width=True)
            st.caption("Средние просмотры по дням/часам публикации (по выборке).")
        else:
            st.info("Нет даты публикации/просмотров для анализа «окна публикаций».")

    # -------- Shorts vs Longs --------
    with tab_split:
        st.subheader("Shorts vs Longs")
        if "__duration_sec__" in df.columns:
            shorts = df[df["__duration_sec__"] < 60]
            longs  = df[df["__duration_sec__"] >= 60]
            c1, c2 = st.columns(2)
            c1.write(f"**Shorts** ({len(shorts)} шт.)")
            c1.dataframe(shorts[[c for c in [title_col, id_col, views_col, ctr_col, "__avd_sec__", "__avg_percent_viewed__", "YouTube Link"] if c in shorts.columns or c=='YouTube Link']], use_container_width=True)
            c2.write(f"**Longs** ({len(longs)} шт.)")
            c2.dataframe(longs[[c for c in [title_col, id_col, views_col, ctr_col, "__avd_sec__", "__avg_percent_viewed__", "YouTube Link"] if c in longs.columns or c=='YouTube Link']], use_container_width=True)
        else:
            st.info("Нет длительности — нельзя разделить Shorts/Longs.")

    # -------- Alerts --------
    with tab_alerts:
        st.subheader("Алерты (ниже порога)")
        issues = []
        if ctr_col:
            issues_ctr = df[(pd.to_numeric(df[ctr_col], errors="coerce") < thr_ctr)]
            if not issues_ctr.empty:
                issues.append(("CTR ниже порога", issues_ctr))
        if df["__avd_sec__"].notna().any():
            issues_avd = df[df["__avd_sec__"] < thr_avd]
            if not issues_avd.empty:
                issues.append(("AVD ниже порога", issues_avd))

        if issues:
            for title, d in issues:
                st.write(f"**{title}: {len(d)} видео**")
                st.dataframe(d[[c for c in [title_col, id_col, ctr_col, "__avd_sec__", "YouTube Link"] if c in d.columns or c=='YouTube Link']], use_container_width=True)
        else:
            st.success("Проблемных видео не найдено (по заданным порогам).")

else:
    st.info("👆 Загрузите основной CSV из YouTube Studio, а затем исследуйте вкладки.")
