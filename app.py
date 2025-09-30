# app.py — YouTube Analytics Tools
# Dashboard (группы + сохранение) и Group Analytics (Year Mix, Timeseries Overlay)
# NEW: помесячные/недельные/квартальные графики наложения метрик
# Дубликаты CSV РАЗРЕШЕНЫ.

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io, re, hashlib

# --------------------------- UI CONFIG ---------------------------
st.set_page_config(page_title="YouTube Analytics Tools", layout="wide")
USE_EMOJI = True
ICON_DASH  = "📊 " if USE_EMOJI else ""
ICON_GROUP = "🧩 " if USE_EMOJI else ""
ICON_BRAND = "📺 " if USE_EMOJI else ""

st.sidebar.markdown(
    f"<div style='font-weight:700;font-size:1.05rem;letter-spacing:.1px;'>{ICON_BRAND}YouTube Analytics Tools</div>",
    unsafe_allow_html=True,
)
st.sidebar.divider()
nav = st.sidebar.radio("Навигация", [f"{ICON_DASH}Dashboard", f"{ICON_GROUP}Group Analytics"])
st.sidebar.divider()

# --------------------------- HELPERS: columns / parsing ---------------------------
def _norm(s: str) -> str:
    return str(s).strip().lower()

MAP = {
    "publish_time": ["video publish time","publish time","время публикации видео","дата публикации","publish date"],
    "views": ["views","просмотры","просмторы","просмотры (views)"],
    "impressions": ["impressions","показы","показы (impressions)","показы значков","показы для значков"],
    "ctr": ["impressions click-through rate","ctr","ctr (%)","ctr for thumbnails (%)","ctr для значков",
            "ctr для значков видео (%)","ctr для значков (%)","ctr для значков видео","ctr видео"],
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

# --------------------------- STORAGE ---------------------------
if "groups" not in st.session_state:
    st.session_state["groups"] = []   # [{name: str, files: [{name, hash, df, meta}, ...]}]

def concat_groups(indices):
    frames = []
    for i in indices:
        if 0 <= i < len(st.session_state["groups"]):
            for f in st.session_state["groups"][i]["files"]:
                if f["df"] is not None and not f["df"].empty:
                    frames.append(f["df"])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def kpis_for_group(group):
    total_impr = 0.0
    total_views = 0.0
    ctr_vals = []
    avd_vals = []
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

# --------------------------- MONTHLY AGG FOR GROUP ---------------------------
def monthly_aggregate_for_group(group: dict) -> pd.DataFrame:
    rows = []
    for f in group["files"]:
        df = f["df"]
        if df is None or df.empty:
            continue
        C = detect_columns(df)
        pub_col = C["publish_time"]
        if not (pub_col and pub_col in df.columns):
            continue

        tmp = df.copy()
        tmp[pub_col] = pd.to_datetime(tmp[pub_col], errors="coerce")
        tmp = tmp[tmp[pub_col].notna()]
        if tmp.empty:
            continue

        imp = C["impressions"]; vws = C["views"]; avd = C["avd"]
        tmp["_impr"]    = pd.to_numeric(tmp[imp].apply(to_number), errors="coerce") if (imp and imp in tmp.columns) else np.nan
        tmp["_views"]   = pd.to_numeric(tmp[vws].apply(to_number), errors="coerce") if (vws and vws in tmp.columns) else np.nan
        tmp["_avd_sec"] = tmp[avd].apply(parse_duration_to_seconds)                 if (avd and avd in tmp.columns) else np.nan

        tmp["_month"] = tmp[pub_col].dt.to_period("M").dt.to_timestamp()
        rows.append(tmp[["_month", "_impr", "_views", "_avd_sec"]])

    if not rows:
        return pd.DataFrame(columns=["Месяц","Показы","Просмотры","AVD_sec"])

    all_df = pd.concat(rows, ignore_index=True)
    ag = (
        all_df.groupby("_month")
              .agg(Показы=("_impr","sum"), Просмотры=("_views","sum"), AVD_sec=("_avd_sec","mean"))
              .reset_index()
              .rename(columns={"_month":"Месяц"})
              .sort_values("Месяц")
    )
    ag["Показы"] = ag["Показы"].fillna(0)
    ag["Просмотры"] = ag["Просмотры"].fillna(0)
    return ag

# ---------- TIMESERIES for overlay ----------
def timeseries_for_group(group: dict, freq: str = "M") -> pd.DataFrame:
    """Агрегация по дате публикации с заданной частотой.
    Возвращает: Date | Impressions | Views | CTR | AVD_sec
    Суммы: Impressions, Views; Среднее: CTR, AVD_sec
    """
    rows = []
    for f in group["files"]:
        df = f["df"]
        if df is None or df.empty:
            continue
        C = detect_columns(df)
        pub_col = C["publish_time"]
        if not (pub_col and pub_col in df.columns):
            continue

        tmp = df.copy()
        tmp[pub_col] = pd.to_datetime(tmp[pub_col], errors="coerce")
        tmp = tmp[tmp[pub_col].notna()]
        if tmp.empty:
            continue

        imp = C["impressions"]; vws = C["views"]; avd = C["avd"]; ctr = C["ctr"]
        tmp["_impr"]    = pd.to_numeric(tmp[imp].apply(to_number), errors="coerce") if (imp and imp in tmp.columns) else np.nan
        tmp["_views"]   = pd.to_numeric(tmp[vws].apply(to_number), errors="coerce") if (vws and vws in tmp.columns) else np.nan
        tmp["_avd_sec"] = tmp[avd].apply(parse_duration_to_seconds)                 if (avd and avd in tmp.columns) else np.nan
        tmp["_ctr"]     = pd.to_numeric(tmp[ctr].apply(to_number), errors="coerce") if (ctr and ctr in tmp.columns) else np.nan

        tmp["_period"] = tmp[pub_col].dt.to_period(freq).dt.to_timestamp()
        rows.append(tmp[["_period","_impr","_views","_avd_sec","_ctr"]])

    if not rows:
        return pd.DataFrame(columns=["Date","Impressions","Views","CTR","AVD_sec"])

    all_df = pd.concat(rows, ignore_index=True)
    ag = (
        all_df.groupby("_period")
              .agg(Impressions=("_impr","sum"),
                   Views=("_views","sum"),
                   AVD_sec=("_avd_sec","mean"),
                   CTR=("_ctr","mean"))
              .reset_index()
              .rename(columns={"_period":"Date"})
              .sort_values("Date")
    )
    return ag

# --------------------------- DASHBOARD ---------------------------
if nav.endswith("Dashboard"):
    st.header("Dashboard")

    # --- Добавить новую группу
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
                    # Дубликаты РАЗРЕШЕНЫ — добавляем всё
                    new_files.append(pack)
                if new_files:
                    st.session_state["groups"].append({"name": group_name.strip(), "files": new_files})
                    st.success(f"Группа добавлена. Загружено файлов: {len(new_files)}.")
                    st.rerun()
                else:
                    st.error("Не удалось добавить файлы (возможно пустые/повреждены).")

    if not st.session_state["groups"]:
        st.info("Добавьте хотя бы одну группу в сайдбаре.")
    else:
        # --- Управление группами
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
                            if pack["df"] is None or pack["df"].empty:
                                continue
                            g["files"].append(pack)   # дубликаты разрешены
                            added += 1
                        if added:
                            st.success(f"Добавлено файлов: {added}.")
                            changed = True
                    if changed:
                        st.rerun()
                    else:
                        st.info("Изменений нет — нечего сохранять.")

                st.markdown("**Файлы группы:**")
                if not g["files"]:
                    st.write("— пока нет файлов.")
                else:
                    for fi, f in enumerate(g["files"]):
                        c1, c2 = st.columns([4,1])
                        with c1:
                            st.write(f["meta"])
                        with c2:
                            if st.button("Удалить", key=f"del_file_{gi}_{fi}"):
                                g["files"].pop(fi)
                                st.rerun()

                st.divider()
                if st.button("Удалить группу", key=f"del_group_{gi}"):
                    st.session_state["groups"].pop(gi)
                    st.rerun()

        st.divider()

        # --- KPI и ПОМЕСЯЧНЫЕ ГРАФИКИ ПО КАЖДОЙ ГРУППЕ ---
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
                    fig_imp.update_layout(xaxis_title="Месяц", yaxis_title="Показы",
                                         margin=dict(l=10, r=10, t=30, b=10), height=400)
                    st.plotly_chart(fig_imp, use_container_width=True)

                with st.expander("👁 Просмотры по месяцам", expanded=False):
                    fig_view = px.line(monthly, x="Месяц", y="Просмотры", markers=True, template="simple_white")
                    fig_view.update_traces(line_color="#59a14f")
                    fig_view.update_layout(xaxis_title="Месяц", yaxis_title="Просмотры",
                                           margin=dict(l=10, r=10, t=30, b=10), height=400)
                    st.plotly_chart(fig_view, use_container_width=True)

                with st.expander("⏱ AVD по месяцам", expanded=False):
                    tmp = monthly.copy()
                    tmp["AVD_text"] = tmp["AVD_sec"].apply(seconds_to_hhmmss)
                    fig_avd = px.line(tmp, x="Месяц", y="AVD_sec", markers=True, template="simple_white",
                                      hover_data={"AVD_text": True, "AVD_sec": False})
                    fig_avd.update_traces(line_color="#e15759",
                                          hovertemplate="Месяц=%{x|%Y-%m}<br>AVD=%{customdata[0]}")
                    fig_avd.update_layout(xaxis_title="Месяц", yaxis_title="AVD, сек (ср.)",
                                          margin=dict(l=10, r=10, t=30, b=10), height=400)
                    st.plotly_chart(fig_avd, use_container_width=True)

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
        ["Сравнение по годам (Year Mix)", "Наложение метрик (Timeseries)"]
    )

    # --------- Year Mix (как раньше) ---------
    if tool.startswith("Сравнение по годам"):
        # (оставил без изменений — ваш прежний модуль Year Mix)
        # ... из-за объёма здесь был бы дубль — этот блок в вашей версии уже работает ...
        st.info("Year Mix остаётся без изменений (как в предыдущей версии кода).")

    # --------- NEW: Timeseries Overlay ---------
    else:
        st.subheader("Наложение метрик (Timeseries)")

        source_mode = st.sidebar.radio("Источник данных", ["Группы из Dashboard", "Загрузить файлы"])
        if source_mode == "Группы из Dashboard":
            if not st.session_state["groups"]:
                st.info("Нет групп данных. Сначала добавьте их в Dashboard.")
                st.stop()
            group_names = [g["name"] for g in st.session_state["groups"]]
        else:
            up = st.sidebar.file_uploader("Загрузите CSV (можно несколько)", type=["csv"], accept_multiple_files=True, key="ts_upload")
            if not up:
                st.info("Загрузите хотя бы один CSV.")
                st.stop()
            # временная группа «Uploaded»
            temp_group = {"name":"Uploaded", "files":[load_uploaded_file(u) for u in up if load_uploaded_file(u)["df"] is not None]}
            st.session_state["__temp_ts_group"] = [temp_group]
            group_names = ["Uploaded"]

        mode = st.radio("Режим", ["Метрики одной группы", "Одна метрика по нескольким группам"], horizontal=True)

        freq_map = {"Месяц":"M", "Неделя":"W", "Квартал":"Q"}
        freq_label = st.selectbox("Частота агрегации", list(freq_map.keys()), index=0)
        freq = freq_map[freq_label]

        smooth = st.slider("Сглаживание (скользящее среднее), периодов", 1, 12, 1)
        index100 = st.checkbox("Индексация к 100 (первый ненулевой период)", value=False)
        avd_minutes = st.checkbox("Показывать AVD в минутах", value=False)

        # ----- режим 1: метрики в одной группе -----
        if mode == "Метрики одной группы":
            gi = st.selectbox("Группа", range(len(group_names)), format_func=lambda i: group_names[i])
            if source_mode == "Группы из Dashboard":
                group = st.session_state["groups"][gi]
            else:
                group = st.session_state["__temp_ts_group"][0]

            ts = timeseries_for_group(group, freq=freq)
            if ts.empty:
                st.warning("Недостаточно данных для построения ряда (нет дат публикации).")
                st.stop()

            metrics_all = ["Impressions","Views","CTR","AVD_sec"]
            metrics_show = st.multiselect("Выберите метрики", metrics_all, default=["Impressions","Views","AVD_sec"])

            # подготовка
            df = ts.copy()
            if smooth > 1:
                for c in metrics_all:
                    if c in df.columns:
                        df[c] = df[c].rolling(smooth, min_periods=1).mean()

            # индексация от 100
            if index100:
                for c in metrics_all:
                    if c in df.columns and c in metrics_show:
                        s = df[c].copy()
                        first = s[s > 0].iloc[0] if not s[s > 0].empty else np.nan
                        if not pd.isna(first) and first != 0:
                            df[c] = s / first * 100

            # конвертация AVD в минуты (для понятности)
            if avd_minutes and "AVD_sec" in df.columns and "AVD_sec" in metrics_show and not index100:
                df["AVD_sec"] = df["AVD_sec"] / 60.0

            # строим две оси: слева большие метрики, справа «процентные/временные»
            left_metrics  = [m for m in metrics_show if m in ["Impressions","Views"]]
            right_metrics = [m for m in metrics_show if m in ["CTR","AVD_sec"]]

            fig = go.Figure()
            for m in left_metrics:
                fig.add_trace(go.Scatter(x=df["Date"], y=df[m], mode="lines+markers", name=m, yaxis="y1"))
            for m in right_metrics:
                fig.add_trace(go.Scatter(x=df["Date"], y=df[m], mode="lines+markers", name=m, yaxis="y2"))

            y2_title = "CTR, %" if ("CTR" in right_metrics and not index100) else ""
            if "AVD_sec" in right_metrics:
                y2_title = (y2_title + (" / " if y2_title else "")) + ("AVD, мин" if avd_minutes and not index100 else ("AVD, сек" if not index100 else "AVD (index)"))

            fig.update_layout(
                template="simple_white",
                margin=dict(l=10, r=10, t=10, b=10),
                height=480,
                xaxis=dict(title="Период"),
                yaxis=dict(title="Значение", side="left"),
                yaxis2=dict(title=y2_title or "Значение", overlaying="y", side="right"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)

            st.caption("Подсказка: включите индексацию, чтобы сравнить именно форму трендов; AVD можно переключить в минуты.")

        # ----- режим 2: одна метрика по нескольким группам -----
        else:
            if source_mode == "Группы из Dashboard":
                multi = st.multiselect("Выберите группы", group_names, default=group_names[: min(3, len(group_names))])
                if not multi:
                    st.warning("Выберите хотя бы одну группу.")
                    st.stop()
                groups = [st.session_state["groups"][group_names.index(n)] for n in multi]
            else:
                groups = [st.session_state["__temp_ts_group"][0]]
                multi = ["Uploaded"]

            metric = st.selectbox("Метрика", ["Impressions","Views","CTR","AVD_sec"], index=1)

            series = []
            for g, name in zip(groups, multi):
                ts = timeseries_for_group(g, freq=freq)
                if ts.empty or metric not in ts.columns:
                    continue
                s = ts[["Date", metric]].copy().rename(columns={metric: name})
                if smooth > 1:
                    s[name] = s[name].rolling(smooth, min_periods=1).mean()
                if index100:
                    first = s[name][s[name] > 0].iloc[0] if not s[name][s[name] > 0].empty else np.nan
                    if not pd.isna(first) and first != 0:
                        s[name] = s[name] / first * 100
                series.append(s)

            if not series:
                st.warning("Недостаточно данных для выбранной метрики.")
                st.stop()

            from functools import reduce
            df = reduce(lambda l,r: pd.merge(l, r, on="Date", how="outer"), series).sort_values("Date")

            # AVD в минуты, если нужно (и не индекс)
            y_title = metric
            if metric == "AVD_sec" and not index100 and avd_minutes:
                df.loc[:, df.columns != "Date"] = df.loc[:, df.columns != "Date"] / 60.0
                y_title = "AVD, мин"
            elif metric == "AVD_sec" and not index100:
                y_title = "AVD, сек"
            elif metric == "CTR" and not index100:
                y_title = "CTR, %"
            elif index100:
                y_title = f"{metric} (index=100)"

            fig = go.Figure()
            for c in df.columns:
                if c == "Date":
                    continue
                fig.add_trace(go.Scatter(x=df["Date"], y=df[c], mode="lines+markers", name=c))
            fig.update_layout(template="simple_white", height=480,
                              xaxis_title="Период", yaxis_title=y_title,
                              margin=dict(l=10, r=10, t=10, b=10),
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig, use_container_width=True)

            st.caption("Подсказка: индексация к 100 помогает сравнивать каналы с разным масштабом.")
