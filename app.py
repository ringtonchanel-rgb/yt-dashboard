# app.py — YouTube Analytics Tools
# Dashboard + Group Analytics (Advanced mode + Year compare)
# С нуля: Advanced mode в стиле YouTube Studio
# (c) 2025

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io, re, hashlib
from functools import reduce
from datetime import datetime

# ===========================
#   BASIC CONFIG
# ===========================
st.set_page_config(page_title="YouTube Analytics Tools", layout="wide")
USE_EMOJI = True
ICON_DASH  = "📊 " if USE_EMOJI else ""
ICON_GROUP = "🧩 " if USE_EMOJI else ""
ICON_BRAND = "📺 " if USE_EMOJI else ""

def do_rerun():
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            pass

# ===========================
#   SIDEBAR BRAND + NAV
# ===========================
st.sidebar.markdown(
    f"<div style='font-weight:700;font-size:1.05rem;letter-spacing:.1px;'>{ICON_BRAND}YouTube Analytics Tools</div>",
    unsafe_allow_html=True,
)
st.sidebar.divider()
nav = st.sidebar.radio("Навигация", [f"{ICON_DASH}Dashboard", f"{ICON_GROUP}Group Analytics"])
st.sidebar.divider()

# ===========================
#   HELPERS: column detection / parsing
# ===========================
def _norm(s: str) -> str:
    return str(s).strip().lower()

MAP = {
    "publish_time": [
        "video publish time","publish time","publish date","upload date",
        "время публикации видео","дата публикации","дата"
    ],
    # real daily date columns (not publish date)
    "day": ["date","day","report date","дата отчета","день","дата (день)"],
    "views": ["views","просмотры","просмторы","просмотры (views)"],
    "impressions": ["impressions","показы","показы значков","показы для значков","показы для значков видео"],
    "ctr": ["impressions click-through rate","ctr","ctr (%)","ctr для значков","ctr для значков видео (%)","ctr видео"],
    "avd": ["average view duration","avg view duration","средняя продолжительность просмотра",
            "средняя продолжительность просмотра видео","average view duration (hh:mm:ss)"],
    "watch_hours": ["watch time (hours)","watch time hours","время просмотра (часы)","время просмотра (часов)"],
    "watch_minutes":["watch time (minutes)","watch time (mins)","время просмотра (мин)","время просмотра (минуты)"],
    "unique_viewers":["unique viewers","уникальные зрители","уникальные пользователи"],
    "engaged_views":["engaged views","вовлеченные просмотры","просмотры с вовлечением"],
    "title": ["title","название видео","video title","видео","название","content","контент"],
}

def find_col(df: pd.DataFrame, names) -> str | None:
    if isinstance(names, str):
        names = [names]
    by_norm = {_norm(c): c for c in df.columns}
    for n in names:
        nn = _norm(n)
        if nn in by_norm:
            return by_norm[nn]
    for n in names:
        nn = _norm(n)
        for c in df.columns:
            if nn in _norm(c):
                return c
    return None

def detect_columns(df: pd.DataFrame):
    return {k: find_col(df, v) for k, v in MAP.items()}

def to_number(x):
    if x is None:
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return np.nan
    s = s.replace(" ", "").replace("\u202f","").replace("\xa0","")
    if s.endswith("%"):
        s = s[:-1]
    if "," in s and "." not in s:
        s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return np.nan

def parse_duration_to_seconds(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip()
    if s == "":
        return np.nan
    m = re.match(r"^(\d+):(\d{2}):(\d{2})$", s)
    if m:
        h, m_, s_ = map(int, m.groups())
        return h*3600 + m_*60 + s_
    m = re.match(r"^(\d+):(\d{2})$", s)
    if m:
        m_, s_ = map(int, m.groups())
        return m_*60 + s_
    try:
        return float(s)
    except Exception:
        return np.nan

def seconds_to_hhmmss(sec):
    if pd.isna(sec):
        return "—"
    sec = int(round(sec))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"

# ===========================
#   FILE LOADER
# ===========================
def load_uploaded_file(uploaded_file):
    raw = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
    h = hashlib.md5(raw).hexdigest()
    df = None
    for enc in (None, "utf-8-sig", "cp1251"):
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=enc) if enc else pd.read_csv(io.BytesIO(raw))
            break
        except Exception:
            df = None
    meta = "❌ не удалось прочитать CSV."
    if df is not None and not df.empty:
        df.columns = [c.strip() for c in df.columns]
        meta = f"✅ {uploaded_file.name}: {df.shape[0]} строк, {df.shape[1]} колонок."
    return {"name": uploaded_file.name, "hash": h, "df": df, "meta": meta}

# ===========================
#   SESSION STORE
# ===========================
if "groups" not in st.session_state:
    st.session_state["groups"] = []  # [{name, files:[{name,hash,df,meta}]}]

# ===========================
#   METRIC HELPERS
# ===========================
def kpis_for_group(group):
    total_impr = 0.0
    total_views = 0.0
    ctr_vals, avd_vals = [], []
    for f in group["files"]:
        df = f["df"]
        if df is None or df.empty:
            continue
        C = detect_columns(df)
        if C["impressions"] and C["impressions"] in df.columns:
            total_impr += pd.to_numeric(df[C["impressions"]].apply(to_number), errors="coerce").fillna(0).sum()
        if C["views"] and C["views"] in df.columns:
            total_views += pd.to_numeric(df[C["views"]].apply(to_number), errors="coerce").fillna(0).sum()
        if C["ctr"] and C["ctr"] in df.columns:
            ctr_vals += list(df[C["ctr"]].apply(to_number).dropna().values)
        if C["avd"] and C["avd"] in df.columns:
            avd_vals += list(df[C["avd"]].apply(parse_duration_to_seconds).dropna().values)
    avg_ctr = float(np.nanmean(ctr_vals)) if ctr_vals else np.nan
    avg_avd = float(np.nanmean(avd_vals)) if avd_vals else np.nan
    return dict(impressions=int(total_impr), views=int(total_views), ctr=avg_ctr, avd_sec=avg_avd)

def df_with_core_cols(df: pd.DataFrame) -> pd.DataFrame:
    C = detect_columns(df)
    out = pd.DataFrame(index=range(len(df)))
    # daily date (if exists)
    if C["day"] and C["day"] in df.columns:
        out["day"] = pd.to_datetime(df[C["day"]], errors="coerce")
    # publish date
    if C["publish_time"] and C["publish_time"] in df.columns:
        out["publish_time"] = pd.to_datetime(df[C["publish_time"]], errors="coerce")
    if C["title"] and C["title"] in df.columns:
        out["title"] = df[C["title"]].astype(str)
    if C["impressions"] and C["impressions"] in df.columns:
        out["impressions"] = pd.to_numeric(df[C["impressions"]].apply(to_number), errors="coerce")
    if C["views"] and C["views"] in df.columns:
        out["views"] = pd.to_numeric(df[C["views"]].apply(to_number), errors="coerce")
    if C["ctr"] and C["ctr"] in df.columns:
        out["ctr"] = pd.to_numeric(df[C["ctr"]].apply(to_number), errors="coerce")
    if C["avd"] and C["avd"] in df.columns:
        out["avd_sec"] = df[C["avd"]].apply(parse_duration_to_seconds)
    if C["watch_hours"] and C["watch_hours"] in df.columns:
        out["watch_hours"] = pd.to_numeric(df[C["watch_hours"]].apply(to_number), errors="coerce")
    elif C["watch_minutes"] and C["watch_minutes"] in df.columns:
        out["watch_hours"] = pd.to_numeric(df[C["watch_minutes"]].apply(to_number), errors="coerce")/60.0
    if C["unique_viewers"] and C["unique_viewers"] in df.columns:
        out["unique_viewers"] = pd.to_numeric(df[C["unique_viewers"]].apply(to_number), errors="coerce")
    if C["engaged_views"] and C["engaged_views"] in df.columns:
        out["engaged_views"] = pd.to_numeric(df[C["engaged_views"]].apply(to_number), errors="coerce")
    return out

def timeseries_for_group(group: dict, freq: str = "M") -> pd.DataFrame:
    rows = []
    for f in group["files"]:
        if f["df"] is None or f["df"].empty:
            continue
        base = df_with_core_cols(f["df"])
        # prefer real daily date
        if "day" in base and base["day"].notna().any():
            tmp = base.dropna(subset=["day"]).copy()
            tmp["_period"] = tmp["day"].dt.to_period(freq).dt.to_timestamp()
        elif "publish_time" in base and base["publish_time"].notna().any():
            tmp = base.dropna(subset=["publish_time"]).copy()
            tmp["_period"] = tmp["publish_time"].dt.to_period(freq).dt.to_timestamp()
        else:
            continue
        cols = [c for c in ["impressions","views","ctr","avd_sec","watch_hours","unique_viewers","engaged_views"] if c in tmp.columns]
        rows.append(tmp[["_period"]+cols])
    if not rows:
        return pd.DataFrame(columns=["Date","Impressions","Views","CTR","AVD_sec","Watch_hours","Unique_viewers","Engaged_views"])
    all_df = pd.concat(rows, ignore_index=True)
    ag = (all_df.groupby("_period")
           .agg(**{
               "Impressions":("impressions","sum") if "impressions" in all_df.columns else ("_period","size"),
               "Views":("views","sum") if "views" in all_df.columns else ("_period","size"),
               "CTR":("ctr","mean") if "ctr" in all_df.columns else ("_period","size"),
               "AVD_sec":("avd_sec","mean") if "avd_sec" in all_df.columns else ("_period","size"),
               "Watch_hours":("watch_hours","sum") if "watch_hours" in all_df.columns else ("_period","size"),
               "Unique_viewers":("unique_viewers","sum") if "unique_viewers" in all_df.columns else ("_period","size"),
               "Engaged_views":("engaged_views","sum") if "engaged_views" in all_df.columns else ("_period","size"),
           }).reset_index()
           .rename(columns={"_period":"Date"})
           .sort_values("Date"))
    return ag

def by_year_for_group(group: dict) -> pd.DataFrame:
    rows = []
    for f in group["files"]:
        if f["df"] is None or f["df"].empty:
            continue
        base = df_with_core_cols(f["df"])
        dt_col = "publish_time" if "publish_time" in base else ("day" if "day" in base else None)
        if not dt_col:
            continue
        tmp = base.dropna(subset=[dt_col]).copy()
        tmp["_year"] = tmp[dt_col].dt.year
        rows.append(tmp[["_year","impressions","views","ctr","avd_sec"]])
    if not rows:
        return pd.DataFrame(columns=["Год","Показы","Просмотры","CTR","AVD_sec","Количество_видео"])
    all_df = pd.concat(rows, ignore_index=True)
    out = (all_df.groupby("_year")
               .agg(Показы=("impressions","sum"),
                    Просмотры=("views","sum"),
                    CTR=("ctr","mean"),
                    AVD_sec=("avd_sec","mean"),
                    Количество_видео=("views","count"))
               .reset_index()
               .rename(columns={"_year":"Год"})
               .sort_values("Год"))
    return out

# ===========================
#   DASHBOARD
# ===========================
if nav.endswith("Dashboard"):
    st.header("Dashboard")

    # ----- добавить группу -----
    with st.sidebar.expander("➕ Добавить группу данных", expanded=True):
        group_name = st.text_input("Название группы (канала)", value=f"Group {len(st.session_state['groups'])+1}")
        files = st.file_uploader("Загрузите один или несколько CSV", type=["csv"], accept_multiple_files=True, key="add_group_files")
        if st.button("Добавить группу"):
            if not group_name.strip():
                st.warning("Введите название группы.")
            elif not files:
                st.warning("Загрузите хотя бы один CSV.")
            else:
                new_files = []
                for uf in files:
                    pack = load_uploaded_file(uf)
                    if pack["df"] is None or pack["df"].empty:
                        continue
                    new_files.append(pack)
                if new_files:
                    st.session_state["groups"].append({"name": group_name.strip(), "files": new_files})
                    st.success(f"Группа добавлена. Загружено файлов: {len(new_files)}.")
                    do_rerun()
                else:
                    st.error("Не удалось добавить файлы.")

    if not st.session_state["groups"]:
        st.info("Добавьте хотя бы одну группу в сайдбаре.")
    else:
        st.markdown("### Управление группами")
        for gi, g in enumerate(st.session_state["groups"]):
            with st.expander(f"Группа: {g['name']}", expanded=False):
                new_name = st.text_input("Название", value=g["name"], key=f"rename_{gi}")
                add_more = st.file_uploader("Добавить отчёты в эту группу", type=["csv"], accept_multiple_files=True, key=f"append_files_{gi}")
                if st.button("Сохранить изменения", key=f"save_group_{gi}"):
                    changed = False
                    if new_name.strip() and new_name.strip()!=g["name"]:
                        g["name"] = new_name.strip(); changed=True
                    if add_more:
                        added=0
                        for uf in add_more:
                            pack = load_uploaded_file(uf)
                            if pack["df"] is None or pack["df"].empty: 
                                continue
                            g["files"].append(pack)  # дубликаты допустимы
                            added+=1
                        if added:
                            st.success(f"Добавлено файлов: {added}.")
                            changed=True
                    if changed: do_rerun()
                    else: st.info("Изменений нет.")

                st.markdown("**Файлы группы:**")
                if not g["files"]:
                    st.write("— пока нет файлов.")
                else:
                    for fi, f in enumerate(g["files"]):
                        c1, c2 = st.columns([4,1])
                        with c1: st.write(f["meta"])
                        with c2:
                            if st.button("Удалить", key=f"del_file_{gi}_{fi}"):
                                g["files"].pop(fi); do_rerun()

                st.divider()
                if st.button("Удалить группу", key=f"del_group_{gi}"):
                    st.session_state["groups"].pop(gi); do_rerun()

        st.divider()

        # KPI + помесячные
        st.markdown("### Сводка по группам")
        rows=[]
        for gi, g in enumerate(st.session_state["groups"]):
            kp = kpis_for_group(g)
            st.subheader(f"Группа: {g['name']}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Показы (сумма)", f"{kp['impressions']:,}".replace(",", " "))
            c2.metric("Просмотры (сумма)", f"{kp['views']:,}".replace(",", " "))
            c3.metric("Средний CTR по видео", "—" if np.isnan(kp["ctr"]) else f"{kp['ctr']:.2f}%")
            c4.metric("Средний AVD", seconds_to_hhmmss(kp["avd_sec"]))

            ts = timeseries_for_group(g, freq="M")
            if not ts.empty:
                with st.expander("📆 Просмотры по месяцам", expanded=False):
                    fig = px.line(ts, x="Date", y="Views", markers=True, template="simple_white")
                    fig.update_traces(line_color="#59a14f")
                    st.plotly_chart(fig, use_container_width=True, height=400)
            st.divider()
            rows.append({
                "Группа": g["name"],
                "Показы": kp["impressions"],
                "Просмотры": kp["views"],
                "CTR, % (ср.)": None if np.isnan(kp["ctr"]) else round(kp["ctr"],2),
                "AVD (ср.)": seconds_to_hhmmss(kp["avd_sec"])
            })
        if rows:
            st.markdown("### Сравнение групп")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ===========================
#   GROUP ANALYTICS
# ===========================
else:
    st.header("Group Analytics")
    tool = st.sidebar.selectbox("Инструмент анализа", ["Advanced mode (как в Studio)","Сравнение по годам"])

    # -------------------- ADVANCED MODE --------------------
    if tool.startswith("Advanced"):
        if not st.session_state["groups"]:
            st.info("Нет групп. Добавьте их в Dashboard.")
            st.stop()

        names = [g["name"] for g in st.session_state["groups"]]
        gi = st.selectbox("Группа (Controls)", range(len(names)), format_func=lambda i: names[i])
        group = st.session_state["groups"][gi]

        # соберём «плоский» датасет по роликам
        frames=[]
        daily_available=False
        min_day=None; max_day=None
        for f in group["files"]:
            df = f["df"]
            if df is None or df.empty: 
                continue
            base = df_with_core_cols(df)
            if "title" not in base: 
                continue
            d = base.copy()
            # есть ли дневная колонка?
            if "day" in d and d["day"].notna().any():
                daily_available=True
                if min_day is None: 
                    min_day = d["day"].min()
                    max_day = d["day"].max()
                else:
                    min_day = min(min_day, d["day"].min())
                    max_day = max(max_day, d["day"].max())
            frames.append(d)

        if not frames:
            st.warning("В этой группе не нашлось данных с названиями видео.")
            st.stop()

        data = pd.concat(frames, ignore_index=True)

        # Фильтры (как в Studio)
        st.markdown("#### Конфигуратор")
        colA, colB, colC, colD = st.columns([2,2,2,2])

        # период
        if daily_available:
            start, end = colA.date_input(
                "Период (Date range)",
                value=(min_day.date(), max_day.date()),
                min_value=min_day.date(), max_value=max_day.date()
            )
            # отфильтруем
            mask = data["day"].notna() & (data["day"]>=pd.to_datetime(start)) & (data["day"]<=pd.to_datetime(end))
            data_f = data.loc[mask].copy()
            period_note = ""
        else:
            period_note = "⚠️ В источнике нет дневных дат. Визуализация агрегирована помесячно по дате публикации."
            data_f = data.copy()

        # метрики — только доступные
        metric_candidates = [
            ("Views","views"),
            ("Engaged views","engaged_views"),
            ("Impressions","impressions"),
            ("CTR, %","ctr"),
            ("Watch time (hours)","watch_hours"),
            ("Unique viewers","unique_viewers"),
            ("AVD (sec)","avd_sec"),
        ]
        available_metrics = [ui for ui,col in metric_candidates if col in data_f.columns]
        picked_metrics = colB.multiselect("Metrics (что показывать в таблице)", available_metrics,
                                          default=[m for m in available_metrics if m.startswith("Views") or m.startswith("Impressions")][:2])

        # TopN и поиск
        topN = int(colC.number_input("Top-N (по просмотрам)", min_value=3, max_value=500, value=50, step=1))
        search = colD.text_input("Filter (поиск по названию)")

        st.write(period_note)

        # Breakdown (пока фиксированный — Content)
        st.caption("Breakdown: **Content** (видео)")

        # применим фильтр поиска по названию
        if search.strip():
            mask = data_f["title"].str.contains(search.strip(), case=False, na=False)
            data_f = data_f.loc[mask].copy()

        # таблица по видео (агрегация за период)
        agg_cols = {}
        if "views" in data_f.columns: agg_cols["Views"]=("views","sum")
        if "engaged_views" in data_f.columns: agg_cols["Engaged views"]=("engaged_views","sum")
        if "impressions" in data_f.columns: agg_cols["Impressions"]=("impressions","sum")
        if "ctr" in data_f.columns: agg_cols["CTR, %"]=("ctr","mean")
        if "watch_hours" in data_f.columns: agg_cols["Watch time (hours)"]=("watch_hours","sum")
        if "unique_viewers" in data_f.columns: agg_cols["Unique viewers"]=("unique_viewers","sum")
        if "avd_sec" in data_f.columns: agg_cols["AVD (sec)"]=("avd_sec","mean")

        per_title = (data_f.groupby("title").agg(**agg_cols).reset_index())
        # сортировка для TopN – по Views если есть, иначе по первой доступной метрике
        sort_col = "Views" if "Views" in per_title.columns else (picked_metrics[0] if picked_metrics else per_title.columns[1])
        per_title = per_title.sort_values(sort_col, ascending=False)
        top_titles = per_title["title"].head(topN).tolist()

        # переключатель графика
        show_chart = st.toggle("Показывать график", value=True)

        # построение рядов (по дням если есть, иначе по месяцам публикации)
        if show_chart:
            if daily_available:
                # построим просто линии: дата -> метрика Views, по названиям
                # можно выбрать, какую метрику рисовать в линиях — возьмём первую выбранную, иначе Views
                line_metric_label = picked_metrics[0] if picked_metrics else ("Views" if "views" in data_f.columns else available_metrics[0])
                line_metric_col = [c for (ui,c) in metric_candidates if ui==line_metric_label][0]

                # соберём дата-сет только для Top-N
                df_plot = data_f[data_f["title"].isin(top_titles)].copy()
                df_plot = df_plot.dropna(subset=["day"])
                if df_plot.empty:
                    st.info("Нет дневных точек за выбранный период для выбранных видео.")
                else:
                    df_plot["_date"] = df_plot["day"].dt.floor("D")
                    y_label = line_metric_label
                    y_col = line_metric_col
                    dfp = (df_plot.groupby(["_date","title"])
                                  .agg(val=(y_col,"sum"))
                                  .reset_index()
                                  .rename(columns={"_date":"Date","title":"Content"}))
                    fig = px.line(dfp, x="Date", y="val", color="Content", template="simple_white")
                    fig.update_layout(height=420, xaxis_title="", yaxis_title=y_label, legend=dict(orientation="h", y=1.1))
                    st.plotly_chart(fig, use_container_width=True)
            else:
                # fallback: агрегируем по месяцу публикации и рисуем
                df_plot = data[data["title"].isin(top_titles)].copy()
                if "publish_time" not in df_plot or df_plot["publish_time"].isna().all():
                    st.info("Нет даты публикации — нечего построить.")
                else:
                    df_plot["_period"] = df_plot["publish_time"].dt.to_period("M").dt.to_timestamp()
                    line_metric_label = picked_metrics[0] if picked_metrics else ("Views" if "views" in df_plot.columns else available_metrics[0])
                    line_metric_col = [c for (ui,c) in metric_candidates if ui==line_metric_label][0]
                    dfp = (df_plot.groupby(["_period","title"])
                                   .agg(val=(line_metric_col,"sum"))
                                   .reset_index()
                                   .rename(columns={"_period":"Date","title":"Content"}))
                    fig = px.line(dfp, x="Date", y="val", color="Content", template="simple_white")
                    fig.update_layout(height=420, xaxis_title="", yaxis_title=line_metric_label, legend=dict(orientation="h", y=1.1))
                    st.plotly_chart(fig, use_container_width=True)

        # таблица ниже — только выбранные метрики (как в Studio)
        show_cols = ["title"] + [ui for ui in picked_metrics if ui in per_title.columns]
        table = per_title[show_cols].copy()
        table.rename(columns={"title":"Content"}, inplace=True)

        # строка Total сверху
        totals = {}
        for c in table.columns:
            if c == "Content": continue
            if "CTR" in c or "AVD" in c:  # средние
                totals[c] = round(per_title[c].mean(), 3)
            else:
                totals[c] = per_title[c].sum()
        total_row = pd.DataFrame([{"Content":"Total", **totals}])
        table = pd.concat([total_row, table], ignore_index=True)

        st.dataframe(table, use_container_width=True, hide_index=True)

    # -------------------- YEAR COMPARE --------------------
    else:
        st.subheader("Сравнение по годам")
        if not st.session_state["groups"]:
            st.info("Нет групп. Добавьте их в Dashboard.")
            st.stop()
        names = [g["name"] for g in st.session_state["groups"]]
        gi = st.selectbox("Группа", range(len(names)), format_func=lambda i: names[i])
        g = st.session_state["groups"][gi]
        y = by_year_for_group(g)
        if y.empty:
            st.warning("Нет данных по годам.")
            st.stop()
        y = y.rename(columns={"Просмотры":"Views", "Количество_видео":"Count"})
        c1, c2 = st.columns(2)
        with c1:
            fig1 = px.bar(y, x="Год", y="Views", template="simple_white", color_discrete_sequence=["#4e79a7"])
            fig1.update_layout(height=420, xaxis_title="Год", yaxis_title="Сумма просмотров")
            st.plotly_chart(fig1, use_container_width=True)
        with c2:
            fig2 = px.bar(y, x="Год", y="Count", template="simple_white", color_discrete_sequence=["#59a14f"])
            fig2.update_layout(height=420, xaxis_title="Год", yaxis_title="Кол-во видео")
            st.plotly_chart(fig2, use_container_width=True)
