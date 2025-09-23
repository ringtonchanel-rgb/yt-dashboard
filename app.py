# app.py — YouTube Analytics Tools
# Dashboard + Group Analytics (+ XY-Constructor + Year compare)
# (c) You — build freely :)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io, re, hashlib
from functools import reduce

# --------------------------- CONFIG ---------------------------
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

# --------------------------- SIDEBAR BRAND ---------------------------
st.sidebar.markdown(
    f"<div style='font-weight:700;font-size:1.05rem;letter-spacing:.1px;'>{ICON_BRAND}YouTube Analytics Tools</div>",
    unsafe_allow_html=True,
)
st.sidebar.divider()
nav = st.sidebar.radio("Навигация", [f"{ICON_DASH}Dashboard", f"{ICON_GROUP}Group Analytics"])
st.sidebar.divider()

# --------------------------- HELPERS: column detection & parsing ---------------------------
def _norm(s: str) -> str:
    return str(s).strip().lower()

MAP = {
    "publish_time": ["video publish time","publish time","время публикации видео","дата публикации","publish date","upload date"],
    "views": ["views","просмотры","просмторы","просмотры (views)"],
    "impressions": ["impressions","показы","показы (impressions)","показы значков","показы для значков","показы для значков видео"],
    "ctr": ["impressions click-through rate","ctr","ctr (%)","ctr for thumbnails (%)","ctr для значков",
            "ctr для значков видео (%)","ctr для значков (%)","ctr видео"],
    "avd": ["average view duration","avg view duration","средняя продолжительность просмотра",
            "средняя продолжительность просмотра видео","average view duration (hh:mm:ss)"],
    "title": ["title","название видео","video title","видео","название"],
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
    s = s.replace(" ", "").replace("\u202f", "").replace("\xa0", "")
    is_percent = s.endswith("%")
    if is_percent:
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
        return h * 3600 + m_ * 60 + s_
    m = re.match(r"^(\d+):(\d{2})$", s)
    if m:
        m_, s_ = map(int, m.groups())
        return m_ * 60 + s_
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

# --------------------------- FILE LOADER ---------------------------
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

# --------------------------- SESSION STORE ---------------------------
if "groups" not in st.session_state:
    st.session_state["groups"] = []   # [{name: str, files: [{name,hash,df,meta}, ...]}]

# --------------------------- METRIC HELPERS ---------------------------
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
    return out

def timeseries_for_group(group: dict, freq: str = "M") -> pd.DataFrame:
    rows = []
    for f in group["files"]:
        if f["df"] is None or f["df"].empty: 
            continue
        base = df_with_core_cols(f["df"])
        if "publish_time" not in base:
            continue
        tmp = base.dropna(subset=["publish_time"]).copy()
        if tmp.empty:
            continue
        tmp["_period"] = tmp["publish_time"].dt.to_period(freq).dt.to_timestamp()
        rows.append(tmp[["_period","impressions","views","ctr","avd_sec"]])
    if not rows:
        return pd.DataFrame(columns=["Date","Impressions","Views","CTR","AVD_sec"])
    all_df = pd.concat(rows, ignore_index=True)
    ag = (
        all_df.groupby("_period")
              .agg(Impressions=("impressions","sum"),
                   Views=("views","sum"),
                   CTR=("ctr","mean"),
                   AVD_sec=("avd_sec","mean"))
              .reset_index()
              .rename(columns={"_period":"Date"})
              .sort_values("Date")
    )
    return ag

def monthly_aggregate_for_group(group: dict) -> pd.DataFrame:
    ts = timeseries_for_group(group, freq="M")
    return (ts.rename(columns={"Date":"Месяц","Impressions":"Показы","Views":"Просмотры","AVD_sec":"AVD_sec"})
             [["Месяц","Показы","Просмотры","AVD_sec"]])

def by_year_for_group(group: dict) -> pd.DataFrame:
    rows = []
    for f in group["files"]:
        if f["df"] is None or f["df"].empty: 
            continue
        base = df_with_core_cols(f["df"])
        if "publish_time" not in base: 
            continue
        tmp = base.dropna(subset=["publish_time"]).copy()
        if tmp.empty: 
            continue
        tmp["_year"] = tmp["publish_time"].dt.year
        rows.append(tmp[["_year","impressions","views","ctr","avd_sec"]])
    if not rows:
        return pd.DataFrame(columns=["Год","Показы","Просмотры","CTR","AVD_sec"])
    all_df = pd.concat(rows, ignore_index=True)
    return (all_df.groupby("_year")
                 .agg(Показы=("impressions","sum"),
                      Просмотры=("views","sum"),
                      CTR=("ctr","mean"),
                      AVD_sec=("avd_sec","mean"),
                      Количество_видео=("views","count"))
                 .reset_index()
                 .rename(columns={"_year":"Год"})
                 .sort_values("Год"))

def by_title_for_group(group: dict, topn: int = 20) -> pd.DataFrame:
    rows = []
    for f in group["files"]:
        if f["df"] is None or f["df"].empty: 
            continue
        base = df_with_core_cols(f["df"])
        if "title" not in base: 
            continue
        rows.append(base[["title","impressions","views","ctr","avd_sec"]])
    if not rows:
        return pd.DataFrame(columns=["Название","Показы","Просмотры","CTR","AVD_sec"])
    all_df = pd.concat(rows, ignore_index=True)
    ag = (all_df.groupby("title")
               .agg(Показы=("impressions","sum"),
                    Просмотры=("views","sum"),
                    CTR=("ctr","mean"),
                    AVD_sec=("avd_sec","mean"))
               .reset_index()
               .rename(columns={"title":"Название"}))
    return ag.sort_values("Просмотры", ascending=False).head(topn)

# --------------------------- DASHBOARD ---------------------------
if nav.endswith("Dashboard"):
    st.header("Dashboard")

    # Add group
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
                    if (pack["df"] is None) or (pack["df"].empty):
                        continue
                    new_files.append(pack)
                if new_files:
                    st.session_state["groups"].append({"name": group_name.strip(), "files": new_files})
                    st.success(f"Группа добавлена. Загружено файлов: {len(new_files)}.")
                    do_rerun()
                else:
                    st.error("Не удалось добавить файлы (возможно пустые/повреждены).")

    if not st.session_state["groups"]:
        st.info("Добавьте хотя бы одну группу в сайдбаре.")
    else:
        st.markdown("### Управление группами")
        for gi, g in enumerate(st.session_state["groups"]):
            with st.expander(f"Группа: {g['name']}", expanded=False):
                new_name = st.text_input("Название", value=g["name"], key=f"rename_{gi}")
                add_more = st.file_uploader(
                    "Добавить отчёты в эту группу",
                    type=["csv"], accept_multiple_files=True, key=f"append_files_{gi}"
                )

                if st.button("Сохранить изменения", key=f"save_group_{gi}"):
                    changed = False
                    if new_name.strip() and new_name.strip() != g["name"]:
                        g["name"] = new_name.strip()
                        changed = True
                    if add_more:
                        added = 0
                        for uf in add_more:
                            pack = load_uploaded_file(uf)
                            if (pack["df"] is None) or (pack["df"].empty):
                                continue
                            g["files"].append(pack)   # дубликаты разрешены
                            added += 1
                        if added:
                            st.success(f"Добавлено файлов: {added}.")
                            changed = True
                    if changed:
                        do_rerun()
                    else:
                        st.info("Изменений нет — нечего сохранять.")

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

        # KPI and monthly charts for each group
        st.markdown("### Сводка по группам")
        kpi_rows = []
        for gi, g in enumerate(st.session_state["groups"]):
            kp = kpis_for_group(g)
            st.subheader(f"Группа: {g['name']}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Показы (сумма)", f"{kp['impressions']:,}".replace(",", " "))
            c2.metric("Просмотры (сумма)", f"{kp['views']:,}".replace(",", " "))
            c3.metric("Средний CTR по видео", "—" if np.isnan(kp["ctr"]) else f"{kp['ctr']:.2f}%")
            c4.metric("Средний AVD", seconds_to_hhmmss(kp["avd_sec"]))

            monthly = monthly_aggregate_for_group(g)
            if monthly.empty:
                st.info("Недостаточно данных (нет даты публикации) для помесячных графиков.")
            else:
                with st.expander("📆 Показы по месяцам", expanded=False):
                    fig_imp = px.line(monthly, x="Месяц", y="Показы", markers=True, template="simple_white")
                    fig_imp.update_traces(line_color="#4e79a7")
                    st.plotly_chart(fig_imp, use_container_width=True, height=400)

                with st.expander("👁 Просмотры по месяцам", expanded=False):
                    fig_view = px.line(monthly, x="Месяц", y="Просмотры", markers=True, template="simple_white")
                    fig_view.update_traces(line_color="#59a14f")
                    st.plotly_chart(fig_view, use_container_width=True, height=400)

                with st.expander("⏱ AVD по месяцам", expanded=False):
                    tmp = monthly.copy()
                    tmp["AVD_text"] = tmp["AVD_sec"].apply(seconds_to_hhmmss)
                    fig_avd = px.line(tmp, x="Месяц", y="AVD_sec", markers=True, template="simple_white",
                                      hover_data={"AVD_text": True, "AVD_sec": False})
                    st.plotly_chart(fig_avd, use_container_width=True, height=400)

            st.divider()
            kpi_rows.append({
                "Группа": g["name"],
                "Показы": kp["impressions"],
                "Просмотры": kp["views"],
                "CTR, % (среднее)": None if np.isnan(kp["ctr"]) else round(kp["ctr"], 2),
                "AVD (ср.)": seconds_to_hhmmss(kp["avd_sec"]),
            })
        if kpi_rows:
            st.markdown("### Сравнение групп")
            comp_df = pd.DataFrame(kpi_rows)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

# --------------------------- GROUP ANALYTICS ---------------------------
else:
    st.header("Group Analytics")
    tool = st.sidebar.selectbox(
        "Выберите инструмент анализа",
        [
            "Наложение метрик (Timeseries)",
            "График-конструктор (Chart Builder)",
            "XY-Конструктор (Views vs AVD и др.)",
            "Сравнение по годам (столбики)",
        ]
    )

    # --------- Timeseries Overlay ----------
    if tool.startswith("Наложение"):
        if not st.session_state["groups"]:
            st.info("Нет групп. Добавьте их в Dashboard.")
            st.stop()

        freq_map = {"Месяц":"M", "Неделя":"W", "Квартал":"Q"}
        freq_label = st.selectbox("Частота агрегации", list(freq_map.keys()), index=0)
        freq = freq_map[freq_label]
        mode = st.radio("Режим", ["Метрики одной группы", "Одна метрика по нескольким группам"], horizontal=True)
        smooth = st.slider("Сглаживание (скользящее среднее), периодов", 1, 12, 1)
        index100 = st.checkbox("Индексация к 100 (первый ненулевой период)", value=False)
        avd_minutes = st.checkbox("AVD в минутах", value=False)

        if mode == "Метрики одной группы":
            gi = st.selectbox("Группа", range(len(st.session_state["groups"])),
                              format_func=lambda i: st.session_state["groups"][i]["name"])
            group = st.session_state["groups"][gi]
            ts = timeseries_for_group(group, freq=freq)
            if ts.empty:
                st.warning("Нет данных для рядов.")
                st.stop()
            metrics_all = ["Impressions","Views","CTR","AVD_sec"]
            metrics_show = st.multiselect("Метрики", metrics_all, default=["Impressions","Views","AVD_sec"])
            df = ts.copy()
            if smooth > 1:
                for c in metrics_all:
                    if c in df.columns: df[c] = df[c].rolling(smooth, min_periods=1).mean()
            if index100:
                for c in metrics_show:
                    s = df[c].copy()
                    first = s[s>0].iloc[0] if not s[s>0].empty else np.nan
                    if not pd.isna(first) and first!=0: df[c] = s/first*100
            if avd_minutes and "AVD_sec" in metrics_show and not index100:
                df["AVD_sec"] = df["AVD_sec"]/60.0

            left = [m for m in metrics_show if m in ["Impressions","Views"]]
            right= [m for m in metrics_show if m in ["CTR","AVD_sec"]]

            fig = go.Figure()
            for m in left:  fig.add_trace(go.Scatter(x=df["Date"], y=df[m], mode="lines+markers", name=m, yaxis="y1"))
            for m in right: fig.add_trace(go.Scatter(x=df["Date"], y=df[m], mode="lines+markers", name=m, yaxis="y2"))
            fig.update_layout(template="simple_white", height=480,
                              xaxis_title="Период", yaxis=dict(title="Значение"),
                              yaxis2=dict(title="%", overlaying="y", side="right"))
            st.plotly_chart(fig, use_container_width=True)

        else:
            names = [g["name"] for g in st.session_state["groups"]]
            picked = st.multiselect("Группы", names, default=names[:min(3,len(names))])
            if not picked: st.stop()
            metric = st.selectbox("Метрика", ["Impressions","Views","CTR","AVD_sec"], index=1)
            series=[]
            for name in picked:
                g = st.session_state["groups"][names.index(name)]
                ts = timeseries_for_group(g, freq=freq)
                if ts.empty or metric not in ts.columns: continue
                s = ts[["Date", metric]].rename(columns={metric:name})
                if smooth>1: s[name] = s[name].rolling(smooth, min_periods=1).mean()
                series.append(s)
            if not series: st.warning("Недостаточно данных"); st.stop()
            df = reduce(lambda l,r: pd.merge(l,r,on="Date",how="outer"), series).sort_values("Date")
            y_title = {"Impressions":"Показы","Views":"Просмотры","CTR":"CTR, %","AVD_sec":"AVD, сек"}[metric]
            fig = go.Figure()
            for c in df.columns:
                if c=="Date": continue
                fig.add_trace(go.Scatter(x=df["Date"], y=df[c], mode="lines+markers", name=c))
            fig.update_layout(template="simple_white", height=480, xaxis_title="Период", yaxis_title=y_title)
            st.plotly_chart(fig, use_container_width=True)

    # --------- Chart Builder ----------
    elif tool.startswith("График-конструктор"):
        st.subheader("График-конструктор (Chart Builder)")

        if not st.session_state["groups"]:
            st.info("Нет групп. Добавьте их в Dashboard.")
            st.stop()

        names = [g["name"] for g in st.session_state["groups"]]
        groups_pick = st.multiselect("Группы данных", names, default=[names[0]])
        if not groups_pick: st.stop()
        groups = [st.session_state["groups"][names.index(n)] for n in groups_pick]

        dim = st.selectbox("Измерение", ["Период", "Год публикации", "Название видео (Top-N)"])
        if dim == "Период":
            freq_map = {"Месяц":"M", "Неделя":"W", "Квартал":"Q"}
            freq_label = st.selectbox("Частота", list(freq_map.keys()), index=0)
            freq = freq_map[freq_label]
        else:
            freq = None
        topn = st.slider("Top-N (для названий)", 3, 100, 20) if dim == "Название видео (Top-N)" else None

        chart_type = st.selectbox(
            "Тип графика",
            ["Линия","Область","Столбцы","Горизонтальные столбцы","Точки","Круг (pie)","Кольцо (donut)"]
        )

        metrics_all = ["Impressions","Views","CTR","AVD_sec"]
        if chart_type in ["Круг (pie)","Кольцо (donut)"]:
            metrics = [st.selectbox("Метрика", metrics_all, index=1)]
        else:
            metrics = st.multiselect("Метрики", metrics_all, default=["Views"])

        col1, col2, col3 = st.columns(3)
        with col1:
            stacked = st.checkbox("Стэкинг", value=chart_type in ["Область","Столбцы","Горизонтальные столбцы"])
        with col2:
            markers = st.checkbox("Маркеры", value=chart_type in ["Линия","Точки"])
        with col3:
            avd_minutes = st.checkbox("AVD в минутах", value=False)

        smooth = st.slider("Сглаживание (скользящее среднее), периодов", 1, 12, 1) if chart_type in ["Линия","Область","Точки"] else 1

        sort_mode = st.selectbox("Сортировка", ["Нет","По метрике (возр.)","По метрике (убыв.)","По категории (А→Я)","По категории (Я→А)"])
        sort_metric = st.selectbox("Метрика для сортировки", metrics_all, index=1)

        def build_dataset():
            frames=[]
            if dim == "Период":
                for g in groups:
                    ts = timeseries_for_group(g, freq=freq)
                    if ts.empty: continue
                    ts = ts.rename(columns={"Date":"Категория"})
                    ts["Группа"] = g["name"]
                    frames.append(ts)
            elif dim == "Год публикации":
                for g in groups:
                    df = by_year_for_group(g)
                    if df.empty: continue
                    df = df.rename(columns={"Год":"Категория","Показы":"Impressions","Просмотры":"Views"})
                    df["Группа"] = g["name"]
                    frames.append(df[["Категория","Impressions","Views","CTR","AVD_sec","Группа"]])
            else:
                g = groups[0]
                df = by_title_for_group(g, topn=topn).rename(columns={"Название":"Категория","Показы":"Impressions","Просмотры":"Views"})
                df["Группа"] = g["name"]
                frames.append(df[["Категория","Impressions","Views","CTR","AVD_sec","Группа"]])

            if not frames:
                return pd.DataFrame(columns=["Категория","Группа"]+metrics_all)
            data = pd.concat(frames, ignore_index=True)
            if avd_minutes and "AVD_sec" in data.columns:
                data["AVD_sec"] = data["AVD_sec"]/60.0
            if dim == "Период" and smooth>1 and chart_type in ["Линия","Область","Точки"]:
                data = data.sort_values("Категория")
                for m in metrics_all:
                    if m in data.columns:
                        data[m] = data.groupby("Группа")[m].transform(lambda s: s.rolling(smooth, min_periods=1).mean())
            return data

        data = build_dataset()
        if data.empty:
            st.warning("Недостаточно данных для графика.")
            st.stop()

        if sort_mode != "Нет":
            asc = sort_mode in ["По метрике (возр.)","По категории (А→Я)"]
            if sort_mode.startswith("По метрике"):
                if sort_metric in data.columns:
                    data = data.sort_values(sort_metric, ascending=asc)
            else:
                data = data.sort_values("Категория", ascending=asc)

        def render_chart(df: pd.DataFrame):
            if chart_type in ["Круг (pie)","Кольцо (donut)"]:
                m = metrics[0]
                if len(df["Группа"].unique())>1:
                    pie_df = df.groupby("Категория", as_index=False)[m].sum()
                    fig = px.pie(pie_df, names="Категория", values=m, hole=0.4 if chart_type=="Кольцо (donut)" else 0)
                else:
                    fig = px.pie(df, names="Категория", values=m, color="Категория",
                                 hole=0.4 if chart_type=="Кольцо (donut)" else 0)
                fig.update_layout(template="simple_white", height=520, legend=dict(orientation="h", y=1.07))
                return fig

            multi_groups = len(df["Группа"].unique())>1
            melted = df.melt(id_vars=["Категория","Группа"], value_vars=[m for m in metrics if m in df.columns],
                             var_name="Метрика", value_name="Значение")

            if chart_type == "Линия":
                fig = px.line(melted, x="Категория", y="Значение",
                              color="Метрика" if not multi_groups else "Группа",
                              line_group="Метрика" if not multi_groups else "Метрика",
                              facet_col="Группа" if multi_groups else None,
                              markers=markers, template="simple_white")

            elif chart_type == "Область":
                fig = px.area(melted, x="Категория", y="Значение",
                              color="Метрика" if not multi_groups else "Группа",
                              facet_col="Группа" if multi_groups else None,
                              groupnorm=None, template="simple_white")
                if not stacked:
                    fig.update_traces(fill=None)

            elif chart_type == "Столбцы":
                fig = px.bar(melted, x="Категория", y="Значение",
                             color="Метрика" if not multi_groups else "Группа",
                             facet_col="Группа" if multi_groups else None,
                             barmode="relative" if stacked else "group",
                             template="simple_white")

            elif chart_type == "Горизонтальные столбцы":
                fig = px.bar(melted, y="Категория", x="Значение",
                             color="Метрика" if not multi_groups else "Группа",
                             facet_col="Группа" if multi_groups else None,
                             barmode="relative" if stacked else "group",
                             orientation="h", template="simple_white")

            elif chart_type == "Точки":
                fig = px.scatter(melted, x="Категория", y="Значение",
                                 color="Метрика" if not multi_groups else "Группа",
                                 facet_col="Группа" if multi_groups else None,
                                 template="simple_white")
                if not markers:
                    fig.update_traces(mode="lines")

            else:
                fig = go.Figure()

            fig.update_layout(height=540, margin=dict(l=10, r=10, t=30, b=10),
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            return fig

        fig = render_chart(data)
        st.plotly_chart(fig, use_container_width=True)

        st.caption("Подсказка: для Pie/Donut выберите одну метрику. Для «Название видео» используется первая выбранная группа (Top-N). Сортировку и Top-N применяйте при сравнении категорий.")

    # --------- XY-Constructor ----------
    elif tool.startswith("XY-Конструктор"):
        st.subheader("XY-Конструктор (например, Просмотры vs AVD)")

        if not st.session_state["groups"]:
            st.info("Нет групп. Добавьте их в Dashboard.")
            st.stop()

        names = [g["name"] for g in st.session_state["groups"]]
        groups_pick = st.multiselect("Группы", names, default=[names[0]])
        if not groups_pick: st.stop()
        groups = [st.session_state["groups"][names.index(n)] for n in groups_pick]

        data_source = st.selectbox("Источник данных", ["Видео (Top-N по просмотрам)", "Период (месячный)", "Год"])
        topn = st.slider("Top-N видео", 5, 200, 50) if data_source.startswith("Видео") else None

        metric_map = {"Impressions":"Показы", "Views":"Просмотры", "CTR":"CTR, %", "AVD_sec":"AVD (сек)"}
        x_metric = st.selectbox("Ось X", list(metric_map.keys()), index=1)  # Views по умолчанию
        y_metric = st.selectbox("Ось Y", list(metric_map.keys()), index=3)  # AVD_sec по умолчанию

        chart_type = st.selectbox("Вид графика", ["Scatter","Bubble","Столбцы","Горизонтальные столбцы","Линия"])
        color_by_group = st.checkbox("Цвет по группе", value=True)
        size_metric = st.selectbox("Размер точки (для Bubble)", list(metric_map.keys()), index=0) if chart_type=="Bubble" else None

        trendline = st.checkbox("Тренд-линия (OLS)", value=False)
        avg_lines = st.checkbox("Показать средние линии", value=True)
        log_x = st.checkbox("Log-X", value=False)
        log_y = st.checkbox("Log-Y", value=False)
        show_labels = st.checkbox("Подписи точек (для Scatter/Bubble)", value=False)
        avd_minutes = st.checkbox("AVD в минутах", value=True)

        # собрать данные
        def build_xy():
            frames=[]
            if data_source.startswith("Видео"):
                # по видео — берём первую группу
                g = groups[0]
                df = by_title_for_group(g, topn=topn).rename(columns={"Показы":"Impressions","Просмотры":"Views"})
                df["Категория"] = df["Название"] = df["Название"] if "Название" in df.columns else df.index.astype(str)
                df["Группа"] = g["name"]
                frames.append(df[["Категория","Название","Группа","Impressions","Views","CTR","AVD_sec"]])
            elif data_source.startswith("Период"):
                for g in groups:
                    ts = timeseries_for_group(g, freq="M")
                    if ts.empty: continue
                    ts = ts.rename(columns={"Date":"Категория"})
                    ts["Название"] = ts["Категория"].dt.strftime("%Y-%m")
                    ts["Группа"] = g["name"]
                    frames.append(ts[["Категория","Название","Группа","Impressions","Views","CTR","AVD_sec"]])
            else:
                for g in groups:
                    y = by_year_for_group(g)
                    if y.empty: continue
                    y = y.rename(columns={"Год":"Категория","Показы":"Impressions","Просмотры":"Views"})
                    y["Название"] = y["Категория"].astype(str)
                    y["Группа"] = g["name"]
                    frames.append(y[["Категория","Название","Группа","Impressions","Views","CTR","AVD_sec"]])

            if not frames:
                return pd.DataFrame(columns=["Категория","Название","Группа","Impressions","Views","CTR","AVD_sec"])
            df = pd.concat(frames, ignore_index=True)
            if avd_minutes:
                df["AVD_sec"] = df["AVD_sec"]/60.0
            return df

        data = build_xy()
        if data.empty:
            st.warning("Недостаточно данных.")
            st.stop()

        # построение
        if chart_type in ["Scatter","Bubble"]:
            fig = px.scatter(
                data, x=x_metric, y=y_metric,
                color=("Группа" if color_by_group else None),
                size=(size_metric if chart_type=="Bubble" else None),
                hover_data=["Название","Группа"],
                trendline=("ols" if trendline else None),
                template="simple_white"
            )
            if show_labels:
                fig.update_traces(mode="markers+text", text=data["Название"], textposition="top center",
                                  selector=dict(mode="markers"))
            fig.update_layout(height=560, xaxis_title=metric_map[x_metric], yaxis_title=metric_map[y_metric])
            fig.update_xaxes(type="log" if log_x else "linear")
            fig.update_yaxes(type="log" if log_y else "linear")
            if avg_lines:
                mx = data[x_metric].mean(); my = data[y_metric].mean()
                fig.add_hline(y=my, line_dash="dot", line_color="#999"); fig.add_vline(x=mx, line_dash="dot", line_color="#999")
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type in ["Столбцы","Горизонтальные столбцы","Линия"]:
            # для «столбиков» отображаем категория=Название (или период/год), высота = y_metric, сортируем по X если надо
            df = data.copy()
            # сортировка «внизу по X» — сортируем по x_metric
            df = df.sort_values(x_metric, ascending=True)
            if chart_type == "Столбцы":
                fig = px.bar(df, x="Название", y=y_metric,
                             color=("Группа" if color_by_group else None),
                             hover_data=[x_metric,"Группа"], template="simple_white")
            elif chart_type == "Горизонтальные столбцы":
                fig = px.bar(df, y="Название", x=y_metric,
                             color=("Группа" if color_by_group else None),
                             hover_data=[x_metric,"Группа"], orientation="h", template="simple_white")
            else:  # Линия
                fig = px.line(df, x=x_metric, y=y_metric,
                              color=("Группа" if color_by_group else None),
                              markers=True, hover_data=["Название","Группа"], template="simple_white")
            fig.update_layout(height=560, xaxis_title=(metric_map[x_metric] if chart_type=="Линия" else ""),
                              yaxis_title=metric_map[y_metric])
            st.plotly_chart(fig, use_container_width=True)

        st.caption("Совет: для Scatter выберите X=Просмотры, Y=AVD (мин), включите тренд-линию и средние линии — это быстро выявляет аномалии.")

    # --------- Year compare ----------
    else:  # Сравнение по годам (столбики)
        st.subheader("Сравнение по годам — суммы просмотров и количество видео")

        if not st.session_state["groups"]:
            st.info("Нет групп. Добавьте их в Dashboard.")
            st.stop()

        gi = st.selectbox("Группа", range(len(st.session_state["groups"])),
                          format_func=lambda i: st.session_state["groups"][i]["name"])
        g = st.session_state["groups"][gi]
        y = by_year_for_group(g)
        if y.empty:
            st.warning("Нет данных по годам (похоже, отсутствует «дата публикации»).")
            st.stop()

        y = y.rename(columns={"Просмотры":"Views", "Количество_видео":"Count"})
        c1, c2 = st.columns(2)

        with c1:
            fig1 = px.bar(y, x="Год", y="Views", template="simple_white", color_discrete_sequence=["#4e79a7"])
            fig1.update_layout(height=420, xaxis_title="Год публикации", yaxis_title="Суммарное количество просмотров")
            st.plotly_chart(fig1, use_container_width=True)

        with c2:
            fig2 = px.bar(y, x="Год", y="Count", template="simple_white", color_discrete_sequence=["#59a14f"])
            fig2.update_layout(height=420, xaxis_title="Год публикации", yaxis_title="Количество видео")
            st.plotly_chart(fig2, use_container_width=True)

        st.caption("Это та же пара «столбиков по годам»: слева — сколько просмотров принесли ролики года, справа — сколько роликов опубликовано в этом году.")
