# app.py — YouTube Analytics Tools (stable UI)
# Features:
# - Группы каналов -> внутри несколько CSV-отчётов (НЕ суммируем по умолчанию, показываем сегментацию)
# - Отдельная загрузка CSV с доходами (привязка по video_id или по дате)
# - Фильтр вертикал/горизонтал (Shorts/Format/<=60s)
# - Нормализатор содержимого groups[] (устраняет TypeError при pack["df"])
# - Аккуратный UI (CSS skin + карточки метрик)
# - Страницы: Dashboard / Channel Explorer / Compare Groups / Manage Groups

import io
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# -------------------------------------------------
#                BASE UI / THEME
# -------------------------------------------------
st.set_page_config(page_title="YouTube Analytics Tools", layout="wide")

# компактная «шкурка» + карточки метрик + фикс читабельности
st.markdown("""
<style>
/* общий ритм */
.block-container { padding-top: 0.8rem; padding-bottom: 2rem; }

/* фикс мелкого текста у радио/кнопок на 100% масштабе */
[data-testid="stRadio"] label, .sidebar-content label { font-size: 0.95rem !important; }

/* карточки метрик */
.metric-card {
  border: 1px solid #e7e7e9; border-radius: 12px; padding: 14px 16px;
  background: #fff; box-shadow: 0 1px 2px rgba(0,0,0,0.03);
}
.metric-title { color:#6b7280; font-size: 0.85rem; margin-bottom:6px; }
.metric-value { font-weight: 800; font-size: 1.35rem; line-height:1.2; }
.metric-sub   { color:#9ca3af; font-size: 0.8rem; }

/* секции */
.section { padding: 6px 0 10px 0; }
.section h3 { margin: 6px 0 12px 0; }

/* таблицам чуть больше воздуха */
[data-testid="stDataFrame"] { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

def render_metric_card(title: str, value: str, sub: str = ""):
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-title">{title}</div>
          <div class="metric-value">{value}</div>
          <div class="metric-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -------------------------------------------------
#                  HELPERS
# -------------------------------------------------
def _num(x):
    """Безопасное приведение к float (поддержка '1 234,5' и '%')."""
    if pd.isna(x): return np.nan
    try:
        if isinstance(x, str):
            s = x.strip().replace("\u202f","").replace("\xa0","").replace(" ","")
            if s.endswith("%"): s = s[:-1]
            if "," in s and "." not in s: s = s.replace(",", ".")
            return float(s)
        return float(x)
    except Exception:
        return np.nan

def parse_duration_to_seconds(val) -> Optional[int]:
    """Поддержка 'MM:SS', 'H:MM:SS', '12m 3s', '605' (сек)."""
    if pd.isna(val): return None
    s = str(val).strip()
    if s.isdigit(): return int(s)
    m = re.match(r'(?:(\d+)\s*h)?\s*(?:(\d+)\s*m)?\s*(?:(\d+)\s*s)?', s, re.I)
    if m and any(m.groups()):
        h = int(m.group(1)) if m.group(1) else 0
        mm = int(m.group(2)) if m.group(2) else 0
        ss = int(m.group(3)) if m.group(3) else 0
        return h*3600 + mm*60 + ss
    parts = s.split(":")
    try:
        if len(parts) == 3:
            h, mm, ss = map(int, parts); return h*3600 + mm*60 + ss
        if len(parts) == 2:
            mm, ss = map(int, parts);    return mm*60 + ss
    except Exception:
        pass
    return None

def seconds_to_hms(x: float) -> str:
    if pd.isna(x): return "—"
    x = int(round(x))
    h = x // 3600; m = (x % 3600) // 60; s = x % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"

def human_int(x: float) -> str:
    if pd.isna(x): return "—"
    x = float(x)
    for unit in ["", "K", "M", "B", "T"]:
        if abs(x) < 1000:
            return f"{x:,.0f}{unit}".replace(",", " ")
        x /= 1000.0
    return f"{x:.1f}P"

def detect_delimiter(buf: bytes) -> str:
    head = buf[:4000].decode("utf-8", errors="ignore")
    return ";" if head.count(";") > head.count(",") else ","

# стандартизация имён
COLUMN_ALIASES: Dict[str, str] = {
    # id
    "video id":"video_id","ид видео":"video_id","id видео":"video_id","content id":"video_id",
    # title
    "title":"title","video title":"title","название видео":"title","название":"title","content":"title","контент":"title",
    # publish time / daily
    "video publish time":"publish_time","publish time":"publish_time","publish date":"publish_time",
    "upload date":"publish_time","время публикации видео":"publish_time","дата публикации":"publish_time","дата":"publish_time",
    "date":"date","day":"date","report date":"date","дата отчета":"date",
    # metrics
    "views":"views","просмотры":"views",
    "impressions":"impressions","показы":"impressions","показы для значков":"impressions",
    "impressions click-through rate":"ctr","ctr":"ctr","ctr (%)":"ctr","ctr для значков":"ctr","ctr для значков видео (%)":"ctr",
    "watch time (hours)":"watch_hours","watch time hours":"watch_hours","часы просмотра":"watch_hours","время просмотра (часы)":"watch_hours",
    "watch time (minutes)":"watch_minutes","время просмотра (мин)":"watch_minutes",
    "average view duration":"avd","avg view duration":"avd","средняя продолжительность просмотра":"avd",
    "estimated revenue":"revenue","estimated partner revenue":"revenue","доход":"revenue",
    # duration / format
    "duration":"duration","длительность":"duration",
    "format":"format","тип контента":"format",
    "shorts":"shorts","is shorts":"shorts"
}

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: COLUMN_ALIASES.get(str(c).strip().lower(), c) for c in df.columns})

def read_csv_smart(file) -> pd.DataFrame:
    raw = file.read()
    delim = detect_delimiter(raw)
    df = pd.read_csv(io.BytesIO(raw), sep=delim, encoding="utf-8", engine="python")
    df = standardize_columns(df)

    # числовые
    for col in ["views","impressions","watch_hours","watch_minutes","ctr","revenue"]:
        if col in df.columns: df[col] = df[col].map(_num)

    # CTR: если все <=1, считаем что это доля -> в %
    if "ctr" in df.columns and df["ctr"].dropna().max() <= 1.0:
        df["ctr"] = df["ctr"] * 100.0

    # время
    if "publish_time" in df.columns:
        df["publish_time"] = pd.to_datetime(df["publish_time"], errors="coerce")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # AVD (сек) — если есть текстовая длительность
    if "avd" in df.columns and "avd_sec" not in df.columns:
        df["avd_sec"] = df["avd"].apply(parse_duration_to_seconds)
    # duration_sec
    if "duration" in df.columns:
        df["duration_sec"] = df["duration"].apply(parse_duration_to_seconds)
    elif "duration_sec" not in df.columns:
        df["duration_sec"] = np.nan

    # format detect
    if "format" not in df.columns:
        df["format"] = np.nan
    if "shorts" in df.columns:
        df.loc[df["shorts"].astype(str).str.lower().isin(["1","true","да","yes"]), "format"] = "vertical"
    # эвристика по длительности
    df.loc[df["format"].isna() & (df["duration_sec"] <= 60), "format"] = "vertical"
    df["format"] = df["format"].fillna("horizontal")

    # watch_minutes -> watch_hours
    if "watch_minutes" in df.columns and "watch_hours" not in df.columns:
        df["watch_hours"] = df["watch_minutes"] / 60.0

    # id
    if "video_id" in df.columns:
        df["video_id"] = df["video_id"].astype(str).str.strip()

    return df

# доходы
def attach_revenue(base_df: pd.DataFrame, revenue_packs: Optional[List[Dict]]) -> pd.DataFrame:
    """Подмешать доход из revenue CSV (по video_id или по date). Безопасно к любым структурам."""
    if not revenue_packs: return base_df
    df = base_df.copy()
    df["revenue_ext"] = np.nan

    for pack in revenue_packs:
        r = pack.get("df") if isinstance(pack, dict) else None
        if r is None or not isinstance(r, pd.DataFrame): continue

        cols = [c.lower() for c in r.columns]
        # вариант 1: video_id, revenue
        if "video_id" in cols and "revenue" in cols:
            r2 = r.rename(columns={r.columns[cols.index("video_id")]: "video_id",
                                   r.columns[cols.index("revenue")]: "revenue"}).copy()
            r2["video_id"] = r2["video_id"].astype(str).str.strip()
            r2["revenue"] = r2["revenue"].map(_num)
            df = df.merge(r2[["video_id","revenue"]], on="video_id", how="left", suffixes=("", "_ext"))
            df["revenue_ext"] = df["revenue_ext"].fillna(df["revenue_ext"])
        # вариант 2: date, revenue — сопоставим грубо по дате публикации
        elif "date" in cols and "revenue" in cols:
            r2 = r.rename(columns={r.columns[cols.index("date")]: "date",
                                   r.columns[cols.index("revenue")]: "revenue"}).copy()
            r2["date"] = pd.to_datetime(r2["date"], errors="coerce")
            daily = r2.groupby("date", as_index=False)["revenue"].sum()
            if "publish_time" in df.columns:
                df["pub_date"] = df["publish_time"].dt.floor("D")
                df = df.merge(daily, left_on="pub_date", right_on="date", how="left", suffixes=("", "_rday"))
                df["revenue_ext"] = df["revenue_ext"].fillna(df["revenue_rday"])
                df.drop(columns=["date","pub_date","revenue_rday"], inplace=True, errors="ignore")

    # финальная колонка дохода
    if "revenue" in df.columns:
        df["revenue_final"] = df["revenue"].fillna(df["revenue_ext"])
    else:
        df["revenue_final"] = df["revenue_ext"]
    return df

# сводка по одному df (с фильтром формата)
def summarize_one_file(df: pd.DataFrame, only_format: str="all") -> Dict[str, float]:
    d = df.copy()
    if only_format in ("vertical","horizontal"):
        d = d.loc[d["format"] == only_format]
    return {
        "videos": len(d),
        "views": d["views"].sum(skipna=True) if "views" in d.columns else np.nan,
        "impressions": d["impressions"].sum(skipna=True) if "impressions" in d.columns else np.nan,
        "ctr": d["ctr"].mean(skipna=True) if "ctr" in d.columns else np.nan,
        "avd_sec": d["avd_sec"].mean(skipna=True) if "avd_sec" in d.columns else np.nan,
        "watch_hours": d["watch_hours"].sum(skipna=True) if "watch_hours" in d.columns else np.nan,
        "revenue": d["revenue_final"].sum(skipna=True) if "revenue_final" in d.columns else np.nan,
    }

# объединение для общего графика (опционально)
def combine_files(files: List[Dict], only_format: str="all") -> pd.DataFrame:
    if not files: return pd.DataFrame()
    dfs = []
    for p in files:
        df = p["df"].copy()
        if only_format in ("vertical","horizontal"):
            df = df.loc[df["format"] == only_format]
        df["__file__"] = p["name"]
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

# нормализатор содержимого groups[group] -> List[{"name":..., "df":...}]
def normalize_packs(packs_raw) -> List[Dict]:
    norm = []
    if not isinstance(packs_raw, list): return norm
    for i, item in enumerate(packs_raw):
        if isinstance(item, dict) and "df" in item and "name" in item and isinstance(item["df"], pd.DataFrame):
            norm.append(item)
        elif isinstance(item, pd.DataFrame):
            norm.append({"name": f"report_{i}.csv", "df": item})
        else:
            continue
    return norm

# -------------------------------------------------
#              SESSION STORAGE
# -------------------------------------------------
if "groups" not in st.session_state or not isinstance(st.session_state.get("groups"), dict):
    st.session_state["groups"] = {}   # { group_name: [ {"name": str, "df": DataFrame}, ... ] }
if "revenues" not in st.session_state or not isinstance(st.session_state.get("revenues"), dict):
    st.session_state["revenues"] = {} # { group_name: [ {"name": str, "df": DataFrame}, ... ] }

# -------------------------------------------------
#                   SIDEBAR
# -------------------------------------------------
st.sidebar.title("📺 YouTube Analytics Tools")
page = st.sidebar.radio("Навигация", ["Dashboard","Channel Explorer","Compare Groups","Manage Groups"], index=0)

with st.sidebar.expander("➕ Добавить/обновить группу", expanded=True):
    gname = st.text_input("Название группы (канала)")
    add_files = st.file_uploader("CSV отчёты (1..N)", type=["csv"], accept_multiple_files=True)
    rev_files = st.file_uploader("CSV с доходами (опционально)", type=["csv"], accept_multiple_files=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Сохранить/обновить"):
            if not gname.strip():
                st.warning("Введите название группы.")
            elif not add_files and not rev_files:
                st.warning("Загрузите хотя бы один CSV (отчёт или доходы).")
            else:
                st.session_state["groups"].setdefault(gname, [])
                # отчёты
                for f in add_files or []:
                    df = read_csv_smart(f)
                    # подмешаем доходы, если уже есть
                    df = attach_revenue(df, st.session_state["revenues"].get(gname, []))
                    st.session_state["groups"][gname].append({"name": f.name, "df": df})
                # доходы
                if rev_files:
                    st.session_state["revenues"].setdefault(gname, [])
                    for rf in rev_files:
                        r_df = read_csv_smart(rf)
                        st.session_state["revenues"][gname].append({"name": rf.name, "df": r_df})
                st.success(f"Группа «{gname}» обновлена.")
    with c2:
        if st.button("Очистить все группы"):
            st.session_state["groups"] = {}
            st.session_state["revenues"] = {}
            st.experimental_rerun()

# список групп
st.sidebar.markdown("### Ваши группы:")
if not st.session_state["groups"]:
    st.sidebar.info("Пока нет групп.")
else:
    for name, packs in st.session_state["groups"].items():
        st.sidebar.write(f"• **{name}** ({len(packs)} отч.)")

# -------------------------------------------------
#                    PAGES
# -------------------------------------------------
if page == "Dashboard":
    st.title("📊 Dashboard")
    if not st.session_state["groups"]:
        st.info("Добавьте группу слева."); st.stop()

    g = st.selectbox("Группа", list(st.session_state["groups"].keys()))
    files_raw = st.session_state["groups"].get(g, [])
    files = normalize_packs(files_raw)
    rev_packs = st.session_state["revenues"].get(g, [])

    if not files:
        st.warning("В группе нет валидных отчётов."); st.stop()

    fmt = st.radio("Формат контента", ["all","horizontal","vertical"], horizontal=True)

    # Ещё раз аккуратно подмешаем доходы к каждому файлу (на случай новых revenue CSV)
    files = [{"name": p["name"], "df": attach_revenue(p["df"], rev_packs)} for p in files]

    # --- Сводка по каждому отчёту (СЕГМЕНТАЦИЯ, без суммирования) ---
    st.subheader("Сводка по отчётам (сегментация, без суммирования)")
    rows = []
    for p in files:
        s = summarize_one_file(p["df"], only_format=fmt)
        rows.append({
            "Отчёт": p["name"],
            "Видео": s["videos"],
            "Просмотры": s["views"],
            "Показы": s["impressions"],
            "CTR, %": s["ctr"],
            "AVD (ср.)": seconds_to_hms(s["avd_sec"]),
            "Часы просмотра": s["watch_hours"],
            "Доход": s["revenue"],
        })
    seg_df = pd.DataFrame(rows)
    # скрыть доход, если его нет ни в одном файле
    if "Доход" in seg_df and seg_df["Доход"].notna().sum() == 0:
        seg_df.drop(columns=["Доход"], inplace=True)

    st.dataframe(
        seg_df.style.format({
            "Просмотры":"{:,.0f}", "Показы":"{:,.0f}", "CTR, %":"{:.2f}", "Часы просмотра":"{:,.1f}"
        }).hide(axis="index"),
        use_container_width=True, height=320
    )

    if not seg_df.empty and "Просмотры" in seg_df.columns:
        st.plotly_chart(
            px.bar(seg_df, x="Отчёт", y="Просмотры", title="Просмотры по отчётам (с учётом фильтра формата)",
                   template="simple_white"),
            use_container_width=True
        )

    st.divider()
    # --- Необязательная общая агрегация (по желанию) ---
    combine = st.toggle("Показать общий обзор по ВСЕМ отчётам (агрегировано, только для визуализации)", value=False)
    if combine:
        comb = combine_files(files, only_format=fmt)
        if comb.empty:
            st.info("Нет данных для общего обзора.")
        else:
            # KPI-карточки по объединённым данным (только визуально)
            s = summarize_one_file(comb, only_format="all")
            c1,c2,c3,c4 = st.columns(4)
            with c1: render_metric_card("Просмотры (сумма)", human_int(s.get("views", np.nan)))
            with c2: render_metric_card("Показы (сумма)",   human_int(s.get("impressions", np.nan)))
            with c3: render_metric_card("CTR, % (ср.)",     f"{s.get('ctr',np.nan):.2f}%" if not pd.isna(s.get('ctr',np.nan)) else "—")
            with c4: render_metric_card("AVD (ср.)",        seconds_to_hms(s.get("avd_sec", np.nan)))

            # распределение просмотров
            st.plotly_chart(
                px.histogram(comb, x="views", color="__file__", nbins=30,
                             title="Распределение просмотров (все файлы)", template="simple_white"),
                use_container_width=True
            )
            # топ-10 видео
            if {"title","views"}.issubset(comb.columns):
                top10 = comb.sort_values("views", ascending=False).head(10)[["title","views","__file__"]]
                st.dataframe(top10.rename(columns={"title":"Название","views":"Просмотры","__file__":"Отчёт"}),
                             use_container_width=True)

elif page == "Channel Explorer":
    st.title("🔎 Channel Explorer")
    if not st.session_state["groups"]:
        st.info("Добавьте группу слева."); st.stop()

    g = st.selectbox("Группа", list(st.session_state["groups"].keys()), key="expl_g")
    files = normalize_packs(st.session_state["groups"].get(g, []))
    rev_packs = st.session_state["revenues"].get(g, [])

    if not files:
        st.warning("В этой группе нет валидных отчётов."); st.stop()

    file_names = [p["name"] for p in files]
    fname = st.selectbox("Отчёт", file_names)
    pack = files[file_names.index(fname)] if file_names else None
    if not (isinstance(pack, dict) and "df" in pack and isinstance(pack["df"], pd.DataFrame)):
        st.error("Структура отчёта повреждена. Удалите и добавьте его заново."); st.stop()

    df = attach_revenue(pack["df"], rev_packs)

    fmt = st.radio("Формат", ["all","horizontal","vertical"], horizontal=True, key="expl_fmt")
    if fmt in ("horizontal","vertical"):
        df = df.loc[df["format"] == fmt]
    st.caption(f"Строк в отчёте: {len(df)}")

    # доступные метрики
    metrics = [m for m in ["views","impressions","ctr","watch_hours","revenue_final"] if m in df.columns]
    if not metrics:
        st.warning("Не нашёл метрик для визуализации."); st.stop()

    m = st.selectbox("Метрика", metrics, index=0)
    chart_type = st.selectbox("Тип графика", ["Bar","Scatter","Histogram"], index=0)

    xcol = "title" if "title" in df.columns else df.columns[0]
    if chart_type == "Bar":
        fig = px.bar(df.nlargest(30, m), x=xcol, y=m, title=f"Top-30 по {m}", template="simple_white")
        fig.update_layout(xaxis_title="", yaxis_title=m)
        st.plotly_chart(fig, use_container_width=True)
    elif chart_type == "Scatter":
        ycand = [i for i in ["impressions","ctr","watch_hours","revenue_final"] if i in df.columns and i != m]
        yaxis = st.selectbox("Ось Y", ycand) if ycand else m
        st.plotly_chart(px.scatter(df, x=m, y=yaxis, hover_data=[xcol], title=f"{m} vs {yaxis}", template="simple_white"),
                        use_container_width=True)
    else:
        st.plotly_chart(px.histogram(df, x=m, nbins=40, title=f"Распределение {m}", template="simple_white"),
                        use_container_width=True)

    st.divider()
    # таблица (с кликабельной ссылкой, если есть video_id/link)
    def yt_link(row):
        link = row.get("video_link") if "video_link" in row else None
        if isinstance(link, str) and link.strip(): return link.strip()
        vid = row.get("video_id") if "video_id" in row else None
        if isinstance(vid, str) and vid.strip():  return f"https://www.youtube.com/watch?v={vid.strip()}"
        return None

    view = df.copy()
    if "ctr" in view: view["CTR, %"] = view["ctr"].round(2)
    if "revenue_final" in view: view["Доход"] = view["revenue_final"]
    if {"watch_hours","views"}.issubset(view.columns):
        safe_v = view["views"].replace(0,np.nan)
        view["AVD"] = ((view["watch_hours"]*3600)/safe_v).apply(lambda s: seconds_to_hms(s) if pd.notna(s) else "—")
    view["YouTube"] = view.apply(yt_link, axis=1)
    show_cols = [c for c in ["title","views","impressions","CTR, %","watch_hours","Доход","format","YouTube","publish_time"] if c in view.columns]
    st.dataframe(view[show_cols].rename(columns={
        "title":"Название","views":"Просмотры","impressions":"Показы","watch_hours":"Часы просмотра",
        "format":"Формат","publish_time":"Публикация"
    }), use_container_width=True)

elif page == "Compare Groups":
    st.title("🆚 Compare Groups")
    if len(st.session_state["groups"]) < 2:
        st.info("Нужно минимум две группы."); st.stop()

    glist = list(st.session_state["groups"].keys())
    a = st.selectbox("Группа A", glist, key="cmp_a")
    b = st.selectbox("Группа B", [x for x in glist if x != a], key="cmp_b")
    fmt = st.radio("Формат", ["all","horizontal","vertical"], horizontal=True, key="cmp_fmt")

    def group_summary(gname: str) -> Dict[str, float]:
        files = normalize_packs(st.session_state["groups"].get(gname, []))
        rev = st.session_state["revenues"].get(gname, [])
        # по умолчанию СУММИРУЕМ ВНУТРИ группы для сравнения «канал vs канал»
        comb = combine_files(files, only_format=fmt)
        comb = attach_revenue(comb, rev) if not comb.empty else comb
        return summarize_one_file(comb, only_format="all")

    sA, sB = group_summary(a), group_summary(b)
    c1,c2,c3,c4 = st.columns(4)
    with c1: render_metric_card(f"{a}: Просмотры", human_int(sA.get("views", np.nan)))
    with c2: render_metric_card(f"{a}: CTR, %",    f"{sA.get('ctr',np.nan):.2f}%" if not pd.isna(sA.get('ctr',np.nan)) else "—")
    with c3: render_metric_card(f"{b}: Просмотры", human_int(sB.get("views", np.nan)))
    with c4: render_metric_card(f"{b}: CTR, %",    f"{sB.get('ctr',np.nan):.2f}%" if not pd.isna(sB.get('ctr',np.nan)) else "—")

    # простая сравнительная таблица
    table = pd.DataFrame([
        {"Группа": a, "Просмотры": sA.get("views", np.nan),
         "Показы": sA.get("impressions", np.nan), "CTR, %": sA.get("ctr", np.nan),
         "AVD": seconds_to_hms(sA.get("avd_sec", np.nan)),
         "Часы просмотра": sA.get("watch_hours", np.nan),
         "Доход": sA.get("revenue", np.nan)},
        {"Группа": b, "Просмотры": sB.get("views", np.nan),
         "Показы": sB.get("impressions", np.nan), "CTR, %": sB.get("ctr", np.nan),
         "AVD": seconds_to_hms(sB.get("avd_sec", np.nan)),
         "Часы просмотра": sB.get("watch_hours", np.nan),
         "Доход": sB.get("revenue", np.nan)},
    ])
    # убрать «Доход», если пусто
    if table["Доход"].notna().sum() == 0:
        table.drop(columns=["Доход"], inplace=True)

    st.dataframe(table.style.format({"Просмотры":"{:,.0f}","Показы":"{:,.0f}","CTR, %":"{:.2f}","Часы просмотра":"{:,.1f}"}).hide(axis="index"),
                 use_container_width=True)

elif page == "Manage Groups":
    st.title("🧰 Manage Groups")
    if not st.session_state["groups"]:
        st.info("Нет групп."); st.stop()

    g = st.selectbox("Группа", list(st.session_state["groups"].keys()), key="mgmt_g")
    packs = normalize_packs(st.session_state["groups"].get(g, []))
    st.write(f"Валидных файлов: **{len(packs)}**")
    if not packs:
        st.info("Добавьте отчёты в сайдбаре выше."); st.stop()

    for i, pack in enumerate(list(packs)):
        with st.expander(f"Отчёт: {pack['name']}", expanded=False):
            st.write(f"Строк: {len(pack['df'])}")
            c1, c2 = st.columns([1,1])
            with c1:
                if st.button("Удалить этот отчёт", key=f"del_{g}_{i}"):
                    raw = st.session_state["groups"][g]
                    # удалить по совпадению имени (или позиции)
                    idx_to_del = None
                    for j, item in enumerate(raw):
                        if isinstance(item, dict) and item.get("name") == pack["name"]:
                            idx_to_del = j; break
                    if idx_to_del is None and i < len(raw):
                        idx_to_del = i
                    if idx_to_del is not None:
                        st.session_state["groups"][g].pop(idx_to_del)
                    st.experimental_rerun()
            with c2:
                st.download_button("Скачать CSV (нормализ.)",
                                   data=pack["df"].to_csv(index=False).encode("utf-8"),
                                   file_name=f"{pack['name']}_normalized.csv",
                                   mime="text/csv")

    if st.button("Удалить всю группу"):
        st.session_state["groups"].pop(g, None)
        st.session_state["revenues"].pop(g, None)
        st.experimental_rerun()
