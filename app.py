import io
import re
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from datetime import datetime

st.set_page_config(page_title="YouTube Analytics Tools", layout="wide")

# ----------------- helpers -----------------
def _num(x):
    if pd.isna(x):
        return np.nan
    try:
        if isinstance(x, str):
            x = x.replace(" ", "").replace(",", ".")
        return float(x)
    except Exception:
        return np.nan

def parse_duration_to_seconds(val) -> Optional[int]:
    if pd.isna(val):
        return None
    s = str(val).strip()
    if s.isdigit():
        return int(s)
    m = re.match(r'(?:(\d+)\s*h)?\s*(?:(\d+)\s*m)?\s*(?:(\d+)\s*s)?', s, re.I)
    if m and any(m.groups()):
        h = int(m.group(1)) if m.group(1) else 0
        mm = int(m.group(2)) if m.group(2) else 0
        ss = int(m.group(3)) if m.group(3) else 0
        return h*3600 + mm*60 + ss
    parts = s.split(":")
    try:
        if len(parts) == 3:
            h, mm, ss = map(int, parts)
            return h*3600 + mm*60 + ss
        elif len(parts) == 2:
            mm, ss = map(int, parts)
            return mm*60 + ss
    except Exception:
        pass
    return None

def seconds_to_hms(x: float) -> str:
    if pd.isna(x):
        return "—"
    x = int(round(x))
    h = x // 3600
    m = (x % 3600) // 60
    s = x % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"

def detect_delimiter(buf: bytes) -> str:
    head = buf[:4000].decode("utf-8", errors="ignore")
    return ";" if head.count(";") > head.count(",") else ","

COLUMN_ALIASES: Dict[str, str] = {
    "video id": "video_id", "ид видео": "video_id", "id видео": "video_id", "content id": "video_id",
    "title": "title", "название видео": "title", "name": "title",
    "publish time": "publish_time", "дата публикации видео": "publish_time",
    "время публикации видео": "publish_time", "publish date": "publish_time", "date": "date",
    "views": "views", "просмотры": "views",
    "impressions": "impressions", "показы": "impressions",
    "ctr": "ctr", "ctr для значков видео (%)": "ctr", "impressions click-through rate": "ctr",
    "avg view duration": "avd", "average view duration": "avd",
    "средняя продолжительность просмотра": "avd",
    "watch time (hours)": "watch_hours", "watch time hours": "watch_hours", "часы просмотра": "watch_hours",
    "duration": "duration", "длительность": "duration",
    "format": "format", "тип контента": "format",
    "shorts": "shorts", "is shorts": "shorts",
    "estimated revenue": "revenue", "estimated partner revenue": "revenue", "доход": "revenue",
}

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: COLUMN_ALIASES.get(str(c).strip().lower(), c) for c in df.columns})

def read_csv_smart(file) -> pd.DataFrame:
    data = file.read()
    delim = detect_delimiter(data)
    df = pd.read_csv(io.BytesIO(data), sep=delim, encoding="utf-8", engine="python")
    df = standardize_columns(df)
    if "views" in df.columns: df["views"] = df["views"].map(_num)
    if "impressions" in df.columns: df["impressions"] = df["impressions"].map(_num)
    if "ctr" in df.columns:
        df["ctr"] = df["ctr"].map(_num)
        if df["ctr"].dropna().max() <= 1.0: df["ctr"] = df["ctr"] * 100
    if "watch_hours" in df.columns: df["watch_hours"] = df["watch_hours"].map(_num)
    if "publish_time" in df.columns: df["publish_time"] = pd.to_datetime(df["publish_time"], errors="coerce")
    if "date" in df.columns: df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "duration" in df.columns: df["duration_sec"] = df["duration"].apply(parse_duration_to_seconds)
    else: df["duration_sec"] = np.nan

    df["format"] = df.get("format")
    if "shorts" in df.columns:
        df.loc[df["shorts"].astype(str).str.lower().isin(["1","true","да","yes"]), "format"] = "vertical"
    df.loc[df["format"].isna() & (df["duration_sec"] <= 60), "format"] = "vertical"
    df["format"] = df["format"].fillna("horizontal")

    if "revenue" in df.columns: df["revenue"] = df["revenue"].map(_num)
    if "video_id" in df.columns: df["video_id"] = df["video_id"].astype(str).str.strip()
    return df

def human_int(x: float) -> str:
    if pd.isna(x): return "—"
    x = float(x)
    for unit in ["", "K", "M", "B"]:
        if abs(x) < 1000: return f"{x:,.0f}{unit}".replace(",", " ")
        x /= 1000.0
    return f"{x:.1f}T"

def attach_revenue(base_df: pd.DataFrame, revenue_packs: Optional[List[Dict]]) -> pd.DataFrame:
    if not revenue_packs:
        return base_df
    df = base_df.copy()
    df["revenue_ext"] = np.nan
    for pack in revenue_packs:
        r = pack.get("df") if isinstance(pack, dict) else None
        if r is None or not isinstance(r, pd.DataFrame):
            continue
        cols = [c.lower() for c in r.columns]
        if "video_id" in cols and "revenue" in cols:
            r2 = r.rename(columns={r.columns[cols.index("video_id")]: "video_id",
                                   r.columns[cols.index("revenue")]: "revenue"}).copy()
            r2["video_id"] = r2["video_id"].astype(str).str.strip()
            r2["revenue"] = r2["revenue"].map(_num)
            df = df.merge(r2[["video_id","revenue"]], on="video_id", how="left", suffixes=("", "_extjoin"))
            df["revenue_ext"] = df["revenue_ext"].fillna(df["revenue_extjoin"])
            df.drop(columns=[c for c in df.columns if c.endswith("_extjoin")], inplace=True)
        elif "date" in cols and "revenue" in cols:
            r2 = r.rename(columns={r.columns[cols.index("date")]: "date",
                                   r.columns[cols.index("revenue")]: "revenue"}).copy()
            r2["date"] = pd.to_datetime(r2["date"], errors="coerce")
            daily = r2.groupby("date", as_index=False)["revenue"].sum()
            if "publish_time" in df.columns:
                df["pub_date"] = df["publish_time"].dt.floor("d")
                df = df.merge(daily, left_on="pub_date", right_on="date", how="left", suffixes=("", "_rday"))
                df["revenue_ext"] = df["revenue_ext"].fillna(df["revenue_rday"])
                df.drop(columns=["date","pub_date","revenue_rday"], inplace=True, errors="ignore")
    if "revenue" in df.columns:
        df["revenue_final"] = df["revenue"].fillna(df["revenue_ext"])
    else:
        df["revenue_final"] = df["revenue_ext"]
    return df

def summarize_one_file(df: pd.DataFrame, only_format: str="all") -> Dict[str, float]:
    d = df.copy()
    if only_format in ("vertical","horizontal"):
        d = d.loc[d["format"] == only_format]
    return {
        "videos": len(d),
        "views": d["views"].sum(skipna=True) if "views" in d.columns else np.nan,
        "impressions": d["impressions"].sum(skipna=True) if "impressions" in d.columns else np.nan,
        "ctr": d["ctr"].mean(skipna=True) if "ctr" in d.columns else np.nan,
        "avd_sec": d["duration_sec"].mean(skipna=True) if "duration_sec" in d.columns else np.nan,
        "watch_hours": d["watch_hours"].sum(skipna=True) if "watch_hours" in d.columns else np.nan,
        "revenue": d["revenue_final"].sum(skipna=True) if "revenue_final" in d.columns else np.nan,
    }

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

# ---- NEW: normalizer of group content ----
def normalize_packs(packs_raw) -> List[Dict]:
    """
    Привести содержимое группы к списку словарей {"name": str, "df": DataFrame}.
    Пропускаем элементы, которые невозможно восстановить.
    """
    norm = []
    if not isinstance(packs_raw, list):
        return norm
    for i, item in enumerate(packs_raw):
        if isinstance(item, dict) and "df" in item and "name" in item and isinstance(item["df"], pd.DataFrame):
            norm.append(item)
        elif isinstance(item, pd.DataFrame):
            norm.append({"name": f"report_{i}.csv", "df": item})
        else:
            # строка/None/неподдерживаемый — пропускаем
            continue
    return norm

# ----------------- session -----------------
if "groups" not in st.session_state:
    st.session_state.groups: Dict[str, List[Dict]] = {}
if "revenues" not in st.session_state:
    st.session_state.revenues: Dict[str, List[Dict]] = {}

# ----------------- sidebar add group -----------------
st.sidebar.title("🖥️ YouTube Analytics Tools")
page = st.sidebar.radio("Навигация",
                        ["Dashboard", "Channel Explorer", "Compare Groups", "Manage Groups"], index=0)

with st.sidebar.expander("➕ Добавить/обновить группу", expanded=True):
    new_group_name = st.text_input("Название группы (канала)")
    add_files = st.file_uploader("CSV отчёты", type=["csv"], accept_multiple_files=True)
    rev_files = st.file_uploader("CSV с доходами (опц.)", type=["csv"], accept_multiple_files=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Добавить группу"):
            if new_group_name:
                st.session_state.groups.setdefault(new_group_name, [])
                for f in add_files or []:
                    df = read_csv_smart(f)
                    df = attach_revenue(df, st.session_state.revenues.get(new_group_name, []))
                    st.session_state.groups[new_group_name].append({"name": f.name, "df": df})
                if rev_files:
                    st.session_state.revenues.setdefault(new_group_name, [])
                    for rf in rev_files:
                        r_df = read_csv_smart(rf)
                        st.session_state.revenues[new_group_name].append({"name": rf.name, "df": r_df})
                st.success("Группа добавлена/обновлена")
            else:
                st.warning("Введите название группы")
    with c2:
        if st.button("Очистить все группы"):
            st.session_state.groups = {}
            st.session_state.revenues = {}
            st.experimental_rerun()

st.sidebar.markdown("### Ваши группы:")
if not st.session_state.groups:
    st.sidebar.info("Пока нет групп")
else:
    for g in st.session_state.groups.keys():
        st.sidebar.write(f"• **{g}** ({len(st.session_state.groups[g])} отч.)")

# ----------------- pages -----------------
if page == "Dashboard":
    st.title("Dashboard")
    if not st.session_state.groups:
        st.info("Добавь группу в левой панели.")
        st.stop()
    gname = st.selectbox("Группа", list(st.session_state.groups.keys()))
    files_raw = st.session_state.groups.get(gname, [])
    files = normalize_packs(files_raw)
    rev_packs = st.session_state.revenues.get(gname, [])

    if not files:
        st.warning("В группе нет валидных отчётов.")
        st.stop()

    fmt = st.radio("Формат", ["all","horizontal","vertical"], horizontal=True)
    # ре-attach revenue для каждого файла
    files = [{"name": p["name"], "df": attach_revenue(p["df"], rev_packs)} for p in files]

    combine_toggle = st.toggle("Объединить все отчёты для общей диаграммы", value=False)

    st.subheader("Сводка по каждому отчёту")
    rows = []
    for p in files:
        s = summarize_one_file(p["df"], only_format=fmt)
        rows.append({"Отчёт": p["name"], "Видео": s["videos"], "Просмотры": s["views"],
                     "Показы": s["impressions"], "CTR, %": s["ctr"],
                     "AVD": seconds_to_hms(s["avd_sec"]), "Часы просмотра": s["watch_hours"],
                     "Доход": s["revenue"]})
    seg_df = pd.DataFrame(rows)
    if "Доход" in seg_df and seg_df["Доход"].notna().sum() == 0:
        seg_df.drop(columns=["Доход"], inplace=True)
    st.dataframe(seg_df.style.format({"Просмотры":"{:,.0f}","Показы":"{:,.0f}",
                                      "CTR, %":"{:.2f}","Часы просмотра":"{:,.1f}"}).hide(axis="index"),
                 use_container_width=True, height=300)

    if not seg_df.empty and "Просмотры" in seg_df:
        st.plotly_chart(px.bar(seg_df, x="Отчёт", y="Просмотры",
                               title="Просмотры по отчётам (с учётом фильтра)"),
                        use_container_width=True)

    st.divider()
    if combine_toggle:
        comb = combine_files(files, only_format=fmt)
        if comb.empty:
            st.info("Нет данных для общей диаграммы.")
        else:
            st.plotly_chart(px.histogram(comb, x="views", color="__file__", nbins=30,
                                         title="Распределение просмотров (все файлы)"),
                            use_container_width=True)
            if {"title","views"}.issubset(comb.columns):
                top10 = comb.sort_values("views", ascending=False).head(10)[["title","views","__file__"]]
                st.dataframe(top10.rename(columns={"title":"Название","views":"Просмотры","__file__":"Отчёт"}),
                             use_container_width=True)

elif page == "Channel Explorer":
    st.title("Channel Explorer")
    if not st.session_state.groups:
        st.info("Добавь группу.")
        st.stop()
    gname = st.selectbox("Группа", list(st.session_state.groups.keys()), key="expl_g")
    files = normalize_packs(st.session_state.groups.get(gname, []))
    rev_packs = st.session_state.revenues.get(gname, [])

    if not files:
        st.warning("В этой группе нет валидных отчётов.")
        st.stop()

    file_names = [p["name"] for p in files]
    fname = st.selectbox("Отчёт", file_names)
    pack = files[file_names.index(fname)] if file_names else None
    if not (isinstance(pack, dict) and "df" in pack):
        st.error("Неверная структура отчёта в группе (ожидался словарь с 'df'). Удалите и добавьте отчёт заново.")
        st.stop()

    df = attach_revenue(pack["df"], rev_packs)
    fmt = st.radio("Формат", ["all","horizontal","vertical"], horizontal=True, key="expl_fmt")
    if fmt in ("horizontal","vertical"):
        df = df.loc[df["format"] == fmt]

    st.caption(f"Строк: {len(df)}")

    metrics = [m for m in ["views","impressions","ctr","watch_hours","revenue_final"] if m in df.columns]
    if not metrics:
        st.warning("Метрик не найдено.")
        st.stop()

    m = st.selectbox("Метрика", metrics, index=0)
    chart_type = st.selectbox("Тип графика", ["Bar","Scatter","Histogram"], index=0)

    xcol = "title" if "title" in df.columns else df.columns[0]
    if chart_type == "Bar":
        fig = px.bar(df.nlargest(30, m), x=xcol, y=m, title=f"Top-30 по {m}")
        fig.update_layout(xaxis_title="", yaxis_title=m)
        st.plotly_chart(fig, use_container_width=True)
    elif chart_type == "Scatter":
        ycand = [i for i in ["impressions","ctr","watch_hours","revenue_final"] if i in df.columns and i != m]
        yaxis = st.selectbox("Ось Y", ycand) if ycand else m
        st.plotly_chart(px.scatter(df, x=m, y=yaxis, hover_data=[xcol], title=f"{m} vs {yaxis}"),
                        use_container_width=True)
    else:
        st.plotly_chart(px.histogram(df, x=m, nbins=40, title=f"Распределение {m}"),
                        use_container_width=True)

    st.divider()
    show_cols = [c for c in ["title","views","impressions","ctr","watch_hours","revenue_final","format"] if c in df.columns]
    st.dataframe(df[show_cols].rename(columns={
        "title":"Название","views":"Просмотры","impressions":"Показы","ctr":"CTR, %","watch_hours":"Часы просмотра",
        "revenue_final":"Доход","format":"Формат"
    }), use_container_width=True)

elif page == "Compare Groups":
    st.title("Compare Groups")
    if len(st.session_state.groups) < 2:
        st.info("Нужны минимум две группы.")
        st.stop()

    g_list = list(st.session_state.groups.keys())
    a = st.selectbox("Группа A", g_list, key="cmp_a")
    b = st.selectbox("Группа B", [x for x in g_list if x != a], key="cmp_b")
    fmt = st.radio("Формат", ["all","horizontal","vertical"], horizontal=True, key="cmp_fmt")

    def sum_group(gname: str) -> Dict[str, float]:
        files = normalize_packs(st.session_state.groups.get(gname, []))
        rev = st.session_state.revenues.get(gname, [])
        if st.toggle(f"Объединить файлы для {gname}", key=f"merge_{gname}", value=True):
            comb = combine_files(files, only_format=fmt)
            comb = attach_revenue(comb, rev) if not comb.empty else comb
            return summarize_one_file(comb, only_format="all")
        else:
            acc = []
            for p in files:
                d = attach_revenue(p["df"], rev)
                acc.append(summarize_one_file(d, only_format=fmt))
            if not acc: return {}
            dfm = pd.DataFrame(acc).mean(numeric_only=True).to_dict()
            dfm["videos"] = np.mean([x["videos"] for x in acc]) if acc else np.nan
            return dfm

    sA, sB = sum_group(a), sum_group(b)
    colA, colB = st.columns(2)
    with colA:
        st.markdown(f"### {a}")
        st.metric("Просмотры", human_int(sA.get("views", np.nan)))
        st.metric("Показы", human_int(sA.get("impressions", np.nan)))
        st.metric("CTR, %", f"{sA.get('ctr', np.nan):.2f}" if not pd.isna(sA.get("ctr", np.nan)) else "—")
        st.metric("AVD", seconds_to_hms(sA.get("avd_sec", np.nan)))
        if not pd.isna(sA.get("revenue", np.nan)): st.metric("Доход", human_int(sA.get("revenue", np.nan)))
    with colB:
        st.markdown(f"### {b}")
        st.metric("Просмотры", human_int(sB.get("views", np.nan)))
        st.metric("Показы", human_int(sB.get("impressions", np.nan)))
        st.metric("CTR, %", f"{sB.get('ctr', np.nan):.2f}" if not pd.isna(sB.get("ctr", np.nan)) else "—")
        st.metric("AVD", seconds_to_hms(sB.get("avd_sec", np.nan)))
        if not pd.isna(sB.get("revenue", np.nan)): st.metric("Доход", human_int(sB.get("revenue", np.nan)))

elif page == "Manage Groups":
    st.title("Manage Groups")
    if not st.session_state.groups:
        st.info("Нет групп")
        st.stop()
    gname = st.selectbox("Группа", list(st.session_state.groups.keys()), key="mgmt_g")
    packs = normalize_packs(st.session_state.groups.get(gname, []))
    st.write(f"Валидных файлов: {len(packs)}")

    for i, pack in enumerate(list(packs)):
        with st.expander(f"Отчёт: {pack['name']}", expanded=False):
            st.write(f"Строк: {len(pack['df'])}")
            c1, c2 = st.columns([1,1])
            with c1:
                if st.button("Удалить этот отчёт", key=f"del_{gname}_{i}"):
                    # удалить исходно из raw списка (чтобы не рассинхрониться)
                    raw = st.session_state.groups[gname]
                    # ищем по имени и размеру
                    for j, item in enumerate(raw):
                        if isinstance(item, dict) and item.get("name")==pack["name"]:
                            st.session_state.groups[gname].pop(j); break
                    st.experimental_rerun()
            with c2:
                st.download_button("Скачать CSV", data=pack["df"].to_csv(index=False).encode("utf-8"),
                                   file_name=f"{pack['name']}_normalized.csv", mime="text/csv")

    if st.button("Удалить всю группу"):
        st.session_state.groups.pop(gname, None)
        st.session_state.revenues.pop(gname, None)
        st.experimental_rerun()
