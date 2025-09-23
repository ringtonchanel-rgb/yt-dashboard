import io
import re
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ------------------------- #
# ------- Utilities -------- #
# ------------------------- #

st.set_page_config(page_title="YouTube Analytics Tools", layout="wide")

def _num(x):
    """Безопасно привести к числу (float)."""
    if pd.isna(x):
        return np.nan
    try:
        if isinstance(x, str):
            x = x.replace(" ", "").replace(",", ".")
        return float(x)
    except Exception:
        return np.nan

def parse_duration_to_seconds(val) -> Optional[int]:
    """
    Поддержка форматов:
      - 'MM:SS'
      - 'H:MM:SS'
      - '00:01:23'
      - '123' (секунды)
      - '12m 3s' (случайные текстовые варианты - по возможности)
    """
    if pd.isna(val):
        return None
    s = str(val).strip()

    # чистый int?
    if s.isdigit():
        return int(s)

    # 12m 3s / 2m / 45s
    m = re.match(r'(?:(\d+)\s*h)?\s*(?:(\d+)\s*m)?\s*(?:(\d+)\s*s)?', s, re.I)
    if m and any(m.groups()):
        h = int(m.group(1)) if m.group(1) else 0
        m_ = int(m.group(2)) if m.group(2) else 0
        sec = int(m.group(3)) if m.group(3) else 0
        if h or m_ or sec:
            return h * 3600 + m_ * 60 + sec

    # H:MM:SS или MM:SS
    parts = s.split(":")
    try:
        if len(parts) == 3:
            h, m_, sec = map(int, parts)
            return h * 3600 + m_ * 60 + sec
        elif len(parts) == 2:
            m_, sec = map(int, parts)
            return m_ * 60 + sec
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
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"

def detect_delimiter(buffer: bytes) -> str:
    """Грубый детект разделителя."""
    head = buffer[:4000].decode("utf-8", errors="ignore")
    if head.count(";") > head.count(","):
        return ";"
    return ","


# Стандартизация имен столбцов: возможные варианты -> единому имени
COLUMN_ALIASES: Dict[str, str] = {
    # id
    "video id": "video_id",
    "ид видео": "video_id",
    "id видео": "video_id",
    "content id": "video_id",

    # title
    "title": "title",
    "название видео": "title",
    "name": "title",

    # publish time
    "publish time": "publish_time",
    "дата публикации видео": "publish_time",
    "время публикации видео": "publish_time",
    "publish date": "publish_time",
    "date": "date",  # если дневной отчёт

    # metrics
    "views": "views",
    "просмотры": "views",

    "impressions": "impressions",
    "показы": "impressions",

    "ctr": "ctr",
    "ctr для значков видео (%)": "ctr",
    "impressions click-through rate": "ctr",

    "avg view duration": "avd",
    "average view duration": "avd",
    "средняя продолжительность просмотра": "avd",

    "watch time (hours)": "watch_hours",
    "watch time hours": "watch_hours",
    "часы просмотра": "watch_hours",

    "duration": "duration",
    "длительность": "duration",

    "format": "format",
    "тип контента": "format",
    "shorts": "shorts",
    "is shorts": "shorts",

    "estimated revenue": "revenue",
    "estimated partner revenue": "revenue",
    "доход": "revenue",
}

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = {}
    for c in df.columns:
        key = str(c).strip().lower()
        new_cols[c] = COLUMN_ALIASES.get(key, c)
    return df.rename(columns=new_cols)

def read_csv_smart(file) -> pd.DataFrame:
    data = file.read()
    delim = detect_delimiter(data)
    df = pd.read_csv(io.BytesIO(data), sep=delim, encoding="utf-8", engine="python")
    df = standardize_columns(df)

    # приведение типов
    if "views" in df.columns:
        df["views"] = df["views"].map(_num)
    if "impressions" in df.columns:
        df["impressions"] = df["impressions"].map(_num)
    if "ctr" in df.columns:
        # CTR может быть проценты (5.3) или 0.053 — оставляем как %
        df["ctr"] = df["ctr"].map(_num)
        # если есть значения <=1 и есть >= 1, не трогаем; если всё <=1 — умножим на 100
        if df["ctr"].dropna().max() <= 1.0:
            df["ctr"] = df["ctr"] * 100.0

    if "watch_hours" in df.columns:
        df["watch_hours"] = df["watch_hours"].map(_num)

    # publish_time/date как datetime, если есть
    if "publish_time" in df.columns:
        df["publish_time"] = pd.to_datetime(df["publish_time"], errors="coerce")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # длительность → секунды
    if "duration" in df.columns:
        df["duration_sec"] = df["duration"].apply(parse_duration_to_seconds)
    else:
        df["duration_sec"] = np.nan

    # формат: vertical/horizontal
    df["format"] = df.get("format")  # возможно уже есть
    # Если есть явный флаг шортов
    if "shorts" in df.columns:
        df.loc[df["shorts"].astype(str).str.lower().isin(["1", "true", "да", "yes"]), "format"] = "vertical"
    # Если нет, то эвристика по длительности
    df.loc[df["format"].isna() & (df["duration_sec"] <= 60), "format"] = "vertical"
    df["format"] = df["format"].fillna("horizontal")

    # revenue если есть
    if "revenue" in df.columns:
        df["revenue"] = df["revenue"].map(_num)

    # нормализуем id
    if "video_id" in df.columns:
        df["video_id"] = df["video_id"].astype(str).str.strip()

    return df


# ------------------------- #
# ---- Session storage ----- #
# ------------------------- #

if "groups" not in st.session_state:
    # groups: { group_name: [ {"name": filename, "df": DataFrame}, ... ] }
    st.session_state.groups: Dict[str, List[Dict]] = {}

if "revenues" not in st.session_state:
    # отдельные файлы доходов (по video_id или по дате), приаттачиваются к последней выбранной группе
    # revenues[group_name] = [ {"name": filename, "df": df}, ... ]
    st.session_state.revenues: Dict[str, List[Dict]] = {}


# ------------------------- #
# ---- Helper metrics ------ #
# ------------------------- #

def human_int(x: float) -> str:
    if pd.isna(x):
        return "—"
    x = float(x)
    for unit in ["", "K", "M", "B"]:
        if abs(x) < 1000:
            return f"{x:,.0f}{unit}".replace(",", " ")
        x /= 1000.0
    return f"{x:.1f}T"

def attach_revenue(base_df: pd.DataFrame, revenue_packs: List[Dict]) -> pd.DataFrame:
    """
    Присоединить доходы, если их загрузили отдельно.
    Поддерживаем 2 типа:
      - по video_id: [video_id, revenue]
      - по date: [date, revenue] (тогда агрегируем доход по дате публикации)
    """
    if not revenue_packs:
        return base_df

    df = base_df.copy()
    df["revenue_ext"] = np.nan

    # попробуем по video_id
    for pack in revenue_packs:
        r = pack["df"]
        cols = [c.lower() for c in r.columns]
        if "video_id" in cols and "revenue" in cols:
            r2 = r.rename(columns={r.columns[cols.index("video_id")]: "video_id",
                                   r.columns[cols.index("revenue")]: "revenue"}).copy()
            r2["video_id"] = r2["video_id"].astype(str).str.strip()
            r2["revenue"] = r2["revenue"].map(_num)
            df = df.merge(r2[["video_id", "revenue"]], on="video_id", how="left", suffixes=("", "_extjoin"))
            df["revenue_ext"] = df["revenue_ext"].fillna(df["revenue_extjoin"])
            df.drop(columns=[c for c in df.columns if c.endswith("_extjoin")], inplace=True)
        elif "date" in cols and "revenue" in cols:
            r2 = r.rename(columns={r.columns[cols.index("date")]: "date",
                                   r.columns[cols.index("revenue")]: "revenue"}).copy()
            r2["date"] = pd.to_datetime(r2["date"], errors="coerce")
            daily = r2.groupby("date", as_index=False)["revenue"].sum()
            # сопоставим доход по дате публикации (грубо)
            if "publish_time" in df.columns:
                df["pub_date"] = df["publish_time"].dt.floor("d")
                df = df.merge(daily, left_on="pub_date", right_on="date", how="left", suffixes=("", "_rday"))
                df["revenue_ext"] = df["revenue_ext"].fillna(df["revenue_rday"])
                df.drop(columns=["date", "pub_date", "revenue_rday"], inplace=True, errors="ignore")

    # финальная колонка revenue_final
    if "revenue" in df.columns:
        df["revenue_final"] = df["revenue"].fillna(df["revenue_ext"])
    else:
        df["revenue_final"] = df["revenue_ext"]

    return df


def summarize_one_file(df: pd.DataFrame, only_format: str = "all") -> Dict[str, float]:
    """
    Сводка по одному загруженному отчету.
    По умолчанию НЕ суммируем с чем-то ещё — вызывается для каждого файла.
    """
    d = df.copy()

    # фильтр вертикал/горизонтал/все
    if only_format in ("vertical", "horizontal"):
        d = d.loc[d["format"] == only_format]

    out = {
        "videos": len(d),
        "views": d["views"].sum(skipna=True) if "views" in d.columns else np.nan,
        "impressions": d["impressions"].sum(skipna=True) if "impressions" in d.columns else np.nan,
        "ctr": d["ctr"].mean(skipna=True) if "ctr" in d.columns else np.nan,
        "avd_sec": d["duration_sec"].mean(skipna=True) if "duration_sec" in d.columns else np.nan,
        "watch_hours": d["watch_hours"].sum(skipna=True) if "watch_hours" in d.columns else np.nan,
        "revenue": d["revenue_final"].sum(skipna=True) if "revenue_final" in d.columns else np.nan,
    }
    return out


def combine_files(files: List[Dict], only_format: str = "all") -> pd.DataFrame:
    """
    Если всё-таки понадобится объединить файлы — аккуратно соединяем (без дедупа по умолчанию),
    только для визуализации.
    """
    if not files:
        return pd.DataFrame()
    dfs = []
    for pack in files:
        df = pack["df"].copy()
        if only_format in ("vertical", "horizontal"):
            df = df.loc[df["format"] == only_format]
        df["__file__"] = pack["name"]
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


# ------------------------- #
# --------- UI ------------- #
# ------------------------- #

st.sidebar.title("🖥️ YouTube Analytics Tools")

page = st.sidebar.radio(
    "Навигация",
    ["Dashboard", "Channel Explorer", "Compare Groups", "Manage Groups"],
    index=0
)

# --- блок добавления группы/отчётов
with st.sidebar.expander("➕ Добавить/обновить группу", expanded=True):
    new_group_name = st.text_input("Название группы (канала)")
    add_files = st.file_uploader("Загрузите один или несколько CSV", type=["csv"], accept_multiple_files=True)
    rev_files = st.file_uploader("Отдельные CSV с доходом (опционально)", type=["csv"], accept_multiple_files=True)

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("Добавить группу"):
            if new_group_name:
                if new_group_name not in st.session_state.groups:
                    st.session_state.groups[new_group_name] = []
                # добавим отчеты
                for f in add_files or []:
                    df = read_csv_smart(f)
                    # прикрутим внешний доход, если уже загружен ранее
                    df = attach_revenue(df, st.session_state.revenues.get(new_group_name, []))
                    st.session_state.groups[new_group_name].append({"name": f.name, "df": df})
                # доходы отдельно сохраняем
                if rev_files:
                    st.session_state.revenues.setdefault(new_group_name, [])
                    for rf in rev_files:
                        r_df = read_csv_smart(rf)
                        st.session_state.revenues[new_group_name].append({"name": rf.name, "df": r_df})
                st.success("Группа добавлена/обновлена")
            else:
                st.warning("Введите название группы")

    with col_btn2:
        if st.button("Очистить все группы"):
            st.session_state.groups = {}
            st.session_state.revenues = {}
            st.experimental_rerun()

# список групп
st.sidebar.markdown("### Ваши группы:")
if not st.session_state.groups:
    st.sidebar.info("Пока нет групп")
else:
    for gname in st.session_state.groups.keys():
        st.sidebar.write(f"• **{gname}** ({len(st.session_state.groups[gname])} отч.)")


# ------------------------- #
# ------- DASHBOARD -------- #
# ------------------------- #

if page == "Dashboard":
    st.title("Dashboard")

    if not st.session_state.groups:
        st.info("Добавь хотя бы одну группу и отчёты в неё — в левой панели.")
        st.stop()

    gname = st.selectbox("Выбери группу", list(st.session_state.groups.keys()))
    files = st.session_state.groups.get(gname, [])
    rev_packs = st.session_state.revenues.get(gname, [])

    st.caption(f"Файлов в группе **{gname}**: {len(files)}")

    # формат фильтр
    fmt = st.radio("Фильтр формата", ["all", "horizontal", "vertical"], horizontal=True)

    # прикрутить внешний доход к каждому файлу (свежая синхронизация)
    updated_files = []
    for pack in files:
        df = attach_revenue(pack["df"], rev_packs)
        updated_files.append({"name": pack["name"], "df": df})
    files = updated_files

    # режим агрегации
    combine_toggle = st.toggle("Объединить все отчёты для общей диаграммы (в дополнение к сегментации)", value=False)

    st.subheader("Сводка по каждому отчёту (сегментация)")

    rows = []
    for pack in files:
        s = summarize_one_file(pack["df"], only_format=fmt)
        rows.append({
            "Отчёт": pack["name"],
            "Видео": s["videos"],
            "Просмотры": s["views"],
            "Показы": s["impressions"],
            "CTR, %": s["ctr"],
            "AVD": seconds_to_hms(s["avd_sec"]),
            "Часы просмотра": s["watch_hours"],
            "Доход": s["revenue"],
        })

    seg_df = pd.DataFrame(rows)
    # показываем доход только если есть хоть одно ненулевое значение
    if "Доход" in seg_df.columns and seg_df["Доход"].notna().sum() == 0:
        seg_df = seg_df.drop(columns=["Доход"])

    st.dataframe(
        seg_df.style.format({
            "Просмотры": "{:,.0f}",
            "Показы": "{:,.0f}",
            "CTR, %": "{:.2f}",
            "Часы просмотра": "{:,.1f}",
            "Доход": "{:,.2f}",
        }).hide(axis="index"),
        use_container_width=True,
        height=300
    )

    # мини-график по отчётам: кто дал больше просмотров
    if not seg_df.empty and "Просмотры" in seg_df.columns:
        st.plotly_chart(
            px.bar(seg_df, x="Отчёт", y="Просмотры", title="Просмотры по отчётам (с учётом фильтра формата)"),
            use_container_width=True
        )

    st.divider()

    if combine_toggle:
        st.subheader("Общий график по всем загруженным отчётам (не вместо сегментации, а дополнение)")
        comb = combine_files(files, only_format=fmt)
        if comb.empty:
            st.info("Нет данных.")
        else:
            # простая распределённая диаграмма просмотров по файлам
            fig = px.histogram(comb, x="views", color="__file__", nbins=30, title="Распределение просмотров (все файлы)")
            st.plotly_chart(fig, use_container_width=True)

            # Топ-10 видео по просмотрам
            if "title" in comb.columns and "views" in comb.columns:
                top10 = comb.sort_values("views", ascending=False).head(10)[["title", "views", "__file__"]]
                st.dataframe(top10.rename(columns={"title": "Название", "views": "Просмотры", "__file__": "Отчёт"}),
                             use_container_width=True)


# ------------------------- #
# ---- Channel Explorer ---- #
# ------------------------- #

elif page == "Channel Explorer":
    st.title("Channel Explorer")
    if not st.session_state.groups:
        st.info("Добавь группу слева.")
        st.stop()

    gname = st.selectbox("Группа", list(st.session_state.groups.keys()), key="expl_g")
    files = st.session_state.groups.get(gname, [])
    rev_packs = st.session_state.revenues.get(gname, [])

    if not files:
        st.info("В этой группе нет файлов.")
        st.stop()

    file_names = [f["name"] for f in files]
    fname = st.selectbox("Выбери отчёт", file_names)
    pack = files[file_names.index(fname)]
    df = attach_revenue(pack["df"], rev_packs)

    fmt = st.radio("Формат", ["all", "horizontal", "vertical"], horizontal=True, key="expl_fmt")
    if fmt in ("horizontal", "vertical"):
        df = df.loc[df["format"] == fmt]

    st.caption(f"Строк: {len(df)}")

    # Конструктор простых срезов: метрика и вид графика
    metrics = []
    if "views" in df.columns:
        metrics.append("views")
    if "impressions" in df.columns:
        metrics.append("impressions")
    if "ctr" in df.columns:
        metrics.append("ctr")
    if "watch_hours" in df.columns:
        metrics.append("watch_hours")
    if "revenue_final" in df.columns:
        metrics.append("revenue_final")

    if not metrics:
        st.warning("Не нашёл ни одной метрики для графиков (views/impressions/ctr/watch_hours/revenue).")
        st.stop()

    m = st.selectbox("Метрика", metrics, index=0)
    chart_type = st.selectbox("Тип графика", ["Bar", "Scatter", "Histogram"], index=0)

    # по умолчанию по названию
    if "title" in df.columns:
        xcol = "title"
    else:
        # fallback
        xcol = df.columns[0]

    if chart_type == "Bar":
        fig = px.bar(df.nlargest(30, m), x=xcol, y=m, title=f"Top-30 по {m}")
        fig.update_layout(xaxis_title="", yaxis_title=m)
        st.plotly_chart(fig, use_container_width=True)
    elif chart_type == "Scatter":
        # Scatter views vs impressions (если есть)
        ycand = [i for i in ["impressions", "ctr", "watch_hours", "revenue_final"] if i in df.columns and i != m]
        yaxis = st.selectbox("Вторая ось (Y)", ycand) if ycand else m
        fig = px.scatter(df, x=m, y=yaxis, hover_data=[xcol], title=f"{m} vs {yaxis}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = px.histogram(df, x=m, nbins=40, title=f"Распределение {m}")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Таблица")
    show_cols = [c for c in ["title", "views", "impressions", "ctr", "watch_hours", "revenue_final", "format"] if c in df.columns]
    st.dataframe(df[show_cols].rename(columns={
        "title": "Название",
        "views": "Просмотры",
        "impressions": "Показы",
        "ctr": "CTR, %",
        "watch_hours": "Часы просмотра",
        "revenue_final": "Доход",
        "format": "Формат",
    }), use_container_width=True)


# ------------------------- #
# ---- Compare Groups -------#
# ------------------------- #

elif page == "Compare Groups":
    st.title("Compare Groups")
    if len(st.session_state.groups) < 2:
        st.info("Нужно минимум две группы.")
        st.stop()

    g1, g2 = st.columns(2)
    with g1:
        a = st.selectbox("Группа A", list(st.session_state.groups.keys()), key="cmp_a")
    with g2:
        b = st.selectbox("Группа B", [x for x in st.session_state.groups.keys() if x != a], key="cmp_b")

    fmt = st.radio("Формат", ["all", "horizontal", "vertical"], horizontal=True, key="cmp_fmt")

    def sum_group(group_name: str) -> Dict[str, float]:
        files = st.session_state.groups[group_name]
        rev = st.session_state.revenues.get(group_name, [])
        # НЕ суммируем по файлам между собой? – в сравнении групп чаще нужен именно общий итог.
        # Сделаем свитч:
        if st.toggle(f"Объединить файлы для {group_name}", key=f"merge_{group_name}", value=True):
            comb = combine_files(files, only_format=fmt)
            comb = attach_revenue(comb, rev) if not comb.empty else comb
            return summarize_one_file(comb, only_format="all")
        else:
            # если отключили — считаем среднее по файлам (как "средняя группа")
            acc = []
            for pack in files:
                df = attach_revenue(pack["df"], rev)
                acc.append(summarize_one_file(df, only_format=fmt))
            if not acc:
                return {}
            tmp = pd.DataFrame(acc).mean(numeric_only=True).to_dict()
            tmp["videos"] = np.mean([x["videos"] for x in acc]) if acc else np.nan
            return tmp

    sA = sum_group(a)
    sB = sum_group(b)

    colA, colB = st.columns(2)
    with colA:
        st.markdown(f"### {a}")
        st.metric("Просмотры", human_int(sA.get("views", np.nan)))
        st.metric("Показы", human_int(sA.get("impressions", np.nan)))
        st.metric("CTR, %", f"{sA.get('ctr', np.nan):.2f}" if not pd.isna(sA.get("ctr", np.nan)) else "—")
        st.metric("AVD", seconds_to_hms(sA.get("avd_sec", np.nan)))
        if not pd.isna(sA.get("revenue", np.nan)):
            st.metric("Доход", human_int(sA.get("revenue", np.nan)))

    with colB:
        st.markdown(f"### {b}")
        st.metric("Просмотры", human_int(sB.get("views", np.nan)))
        st.metric("Показы", human_int(sB.get("impressions", np.nan)))
        st.metric("CTR, %", f"{sB.get('ctr', np.nan):.2f}" if not pd.isna(sB.get("ctr", np.nan)) else "—")
        st.metric("AVD", seconds_to_hms(sB.get("avd_sec", np.nan)))
        if not pd.isna(sB.get("revenue", np.nan)):
            st.metric("Доход", human_int(sB.get("revenue", np.nan)))


# ------------------------- #
# ----- Manage Groups ------ #
# ------------------------- #

elif page == "Manage Groups":
    st.title("Manage Groups")
    if not st.session_state.groups:
        st.info("Нет групп")
        st.stop()

    gname = st.selectbox("Выбери группу", list(st.session_state.groups.keys()), key="mgmt_g")
    packs = st.session_state.groups[gname]

    st.write(f"Всего файлов: {len(packs)}")
    for i, pack in enumerate(list(packs)):
        with st.expander(f"Отчёт: {pack['name']}", expanded=False):
            st.write(f"Строк: {len(pack['df'])}")
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("Удалить этот отчёт", key=f"del_{gname}_{i}"):
                    st.session_state.groups[gname].pop(i)
                    st.experimental_rerun()
            with col2:
                st.download_button("Скачать как CSV", data=pack["df"].to_csv(index=False).encode("utf-8"),
                                   file_name=f"{pack['name']}_normalized.csv", mime="text/csv")

    if st.button("Удалить всю группу"):
        st.session_state.groups.pop(gname, None)
        st.session_state.revenues.pop(gname, None)
        st.experimental_rerun()
