import streamlit as st
import pandas as pd
import numpy as np
import io, re
import plotly.express as px

st.set_page_config(page_title="YouTube Channelytics", layout="wide")

# ============================
# ГАРАНТИРОВАННАЯ ИНИЦИАЛИЗАЦИЯ STATE
# ============================
if "groups" not in st.session_state or not isinstance(st.session_state.get("groups"), dict):
    st.session_state["groups"] = {}          # имя_группы -> {"df": DataFrame, "allow_dups": bool}

def reset_state():
    st.session_state["groups"] = {}
    st.success("Состояние сброшено.")

# ============================
# Утилиты нормализации колонок
# ============================
def _norm(s: str) -> str:
    return str(s).strip().lower()

COLMAP = {
    "publish_time": [
        "video publish time","publish time","publish date","upload date",
        "время публикации видео","дата публикации","дата"
    ],
    "title": ["title","video title","название видео","название","content","контент"],
    "video_id": ["video id","id","контент","ид видео","ид"],
    "video_link": ["youtube link","link","ссылка","url"],
    "views": ["views","просмотры","просмторы"],
    "impressions": ["impressions","показы","показы для значков"],
    "ctr": ["impressions click-through rate","ctr","ctr (%)","ctr для значков","ctr видео"],
    "watch_hours": ["watch time (hours)","watch time hours","время просмотра (часы)","время просмотра (часов)"],
    "watch_minutes":["watch time (minutes)","watch time (mins)","время просмотра (мин)","время просмотра (минуты)"],
}

def find_col(df, names):
    if isinstance(names, str): names=[names]
    pool = {_norm(c): c for c in df.columns}
    # точное
    for n in names:
        nn=_norm(n)
        if nn in pool: return pool[nn]
    # частичное
    for n in names:
        nn=_norm(n)
        for c in df.columns:
            if nn in _norm(c): return c
    return None

def detect_columns(df):
    return {k: find_col(df, v) for k,v in COLMAP.items()}

def to_number(x):
    if x is None: return np.nan
    if isinstance(x,(int,float,np.number)): return float(x)
    s = str(x).strip()
    if s=="" or s.lower() in {"nan","none"}: return np.nan
    s = s.replace(" ", "").replace("\u202f","").replace("\xa0","")
    if s.endswith("%"): s = s[:-1]
    if "," in s and "." not in s:
        s = s.replace(",", ".")
    try: return float(s)
    except: return np.nan

def fmt_int(n):
    try:
        n = int(round(float(n)))
        return f"{n:,}".replace(",", " ")
    except:
        return "—"

def fmt_time_from_hours(hours):
    if pd.isna(hours) or hours<=0: return "—"
    sec = int(hours*3600)
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    if h>0: return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"

def yt_link(row):
    link = row.get("video_link")
    if isinstance(link,str) and link.strip():
        return link.strip()
    vid  = row.get("video_id")
    if isinstance(vid,str) and vid.strip():
        return f"https://www.youtube.com/watch?v={vid.strip()}"
    return None

# ============================
# Парсер CSV (мультифайлы)
# ============================
def parse_many(files, allow_dups=True):
    dfs=[]
    meta=[]
    for uf in files:
        raw = uf.getvalue()
        df=None
        for enc in (None,"utf-8-sig","cp1251"):
            try:
                df = pd.read_csv(io.BytesIO(raw), encoding=enc) if enc else pd.read_csv(io.BytesIO(raw))
                break
            except Exception:
                pass
        if df is None or df.empty:
            meta.append(f"❌ {uf.name}: не удалось прочитать CSV")
            continue

        df.columns=[c.strip() for c in df.columns]
        cols = detect_columns(df)

        if not cols["publish_time"]:
            meta.append(f"⚠️ {uf.name}: нет даты публикации — пропускаю.")
            continue

        out = pd.DataFrame()
        out["publish_time"] = pd.to_datetime(df[cols["publish_time"]], errors="coerce")
        out = out.dropna(subset=["publish_time"])

        if cols["title"]: out["title"] = df[cols["title"]].astype(str)
        if cols["video_id"]: out["video_id"] = df[cols["video_id"]].astype(str)
        if cols["video_link"]: out["video_link"] = df[cols["video_link"]].astype(str)

        if cols["views"]: out["views"] = pd.to_numeric(df[cols["views"]].apply(to_number), errors="coerce")
        if cols["impressions"]: out["impressions"] = pd.to_numeric(df[cols["impressions"]].apply(to_number), errors="coerce")
        if cols["ctr"]: out["ctr"] = pd.to_numeric(df[cols["ctr"]].apply(to_number), errors="coerce")
        if cols["watch_hours"]:
            out["watch_hours"] = pd.to_numeric(df[cols["watch_hours"]].apply(to_number), errors="coerce")
        elif cols["watch_minutes"]:
            out["watch_hours"] = pd.to_numeric(df[cols["watch_minutes"]].apply(to_number), errors="coerce")/60.0

        out["pub_year"] = out["publish_time"].dt.year
        out["pub_month"] = out["publish_time"].dt.month
        dfs.append(out)
        meta.append(f"✅ {uf.name}: {out.shape[0]} строк")

    if not dfs:
        return None, meta

    big = pd.concat(dfs, ignore_index=True)
    if not allow_dups and "title" in big:
        before = len(big)
        big = big.drop_duplicates(subset=["title","publish_time"])
        meta.append(f"↪️ удалены дубликаты: {before-len(big)}")
    return big, meta

# ============================
# Sidebar: навигация + группы
# ============================
st.sidebar.markdown("### 📺 YouTube Analytics Tools")
page = st.sidebar.radio("Навигация", ["Dashboard","Channel Explorer","Compare Groups","Manage Groups"], index=0)

st.sidebar.markdown("---")
with st.sidebar.expander("➕ Добавить/обновить группу", expanded=(page=="Manage Groups")):
    with st.form("add_group_form", clear_on_submit=False):
        gname = st.text_input("Название группы (канала)", value="")
        uploaded = st.file_uploader("Загрузите один или несколько CSV", type=["csv"], accept_multiple_files=True)
        allow_dups = st.checkbox("Разрешать дубли строк", value=False)
        submitted = st.form_submit_button("Сохранить/обновить группу")
    if submitted:
        if not gname.strip():
            st.warning("Дайте имя группе.")
        elif not uploaded:
            st.warning("Прикрепите хотя бы один CSV.")
        else:
            df_parsed, notes = parse_many(uploaded, allow_dups=allow_dups)
            for n in notes: st.write(n)
            if df_parsed is not None and not df_parsed.empty:
                st.session_state["groups"][gname] = {"df": df_parsed, "allow_dups": allow_dups}
                st.success(f"Группа «{gname}» сохранена: {df_parsed.shape[0]} строк.")

# безопасный список групп
groups = st.session_state.get("groups", {})
group_names = sorted(list(groups.keys()))

if groups:
    st.sidebar.markdown("#### Ваши группы:")
    # БЕЗ прямого обращения атрибутом (только словарь)
    for k in list(groups.keys()):
        colA, colB = st.sidebar.columns([3,1])
        colA.write(k)
        if colB.button("✖", key=f"del_{k}"):
            groups.pop(k, None)
            st.session_state["groups"] = groups
            st.experimental_rerun()

    if st.sidebar.button("Очистить все группы"):
        reset_state()
        st.experimental_rerun()

# ======================
# KPI helpers
# ======================
def kpi_for_df(dff):
    views = dff["views"].sum() if "views" in dff else np.nan
    impr  = dff["impressions"].sum() if "impressions" in dff else np.nan
    ctr   = dff["ctr"].dropna().mean() if "ctr" in dff else np.nan
    wh    = dff["watch_hours"].sum() if "watch_hours" in dff else np.nan
    avd   = np.nan
    if "views" in dff and "watch_hours" in dff:
        safe_views = dff["views"].replace(0,np.nan)
        avd = (dff["watch_hours"]*3600).sum() / safe_views.sum()
    return views, impr, ctr, avd

def monthly_agg(dff, metric):
    if metric not in dff.columns: return pd.DataFrame(columns=["ym","value"])
    agg = (dff.groupby([dff["publish_time"].dt.to_period("M")])[metric]
           .sum()
           .reset_index(name="value"))
    agg["ym"] = agg["publish_time"].astype(str)
    return agg[["ym","value"]]

def by_year_agg(dff, metric):
    if metric not in dff.columns: return pd.DataFrame(columns=["Год","value"])
    a = dff.groupby("pub_year")[metric].sum().reset_index().rename(columns={"pub_year":"Год","value":metric})
    return a

# ======================
# DASHBOARD
# ======================
if page=="Dashboard":
    st.title("📊 Dashboard")
    if not group_names:
        st.info("Сначала добавьте группу во вкладке **Manage Groups**.")
        st.stop()

    g = st.selectbox("Группа", group_names, index=0)
    df_g = groups[g]["df"].copy()

    years_sorted = sorted(df_g["pub_year"].dropna().unique())
    y_from, y_to = st.select_slider("Диапазон лет публикации", options=years_sorted, value=(years_sorted[0], years_sorted[-1]))
    mask = (df_g["pub_year"]>=y_from) & (df_g["pub_year"]<=y_to)
    df_g = df_g.loc[mask].copy()

    v, imp, ctr, avd = kpi_for_df(df_g)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Показы (сумма)", fmt_int(imp))
    c2.metric("Просмотры (сумма)", fmt_int(v))
    c3.metric("Средний CTR", f"{round(ctr,2)}%" if pd.notna(ctr) else "—")
    c4.metric("Средний AVD", fmt_time_from_hours(avd/3600) if pd.notna(avd) else "—")

    st.markdown("### Тренд по месяцам")
    metric = st.selectbox("Метрика для тренда", [m for m in ["views","impressions","watch_hours","ctr"] if m in df_g.columns],
                          format_func=lambda x: {"views":"Просмотры","impressions":"Показы","watch_hours":"Часы просмотра","ctr":"CTR"}[x])
    chart_type = st.radio("Тип графика", ["line","bar","area"], horizontal=True)
    smooth = st.slider("Сглаживание (rolling, месяцев)", 1, 6, 1)
    logy = st.checkbox("Логарифмическая шкала", value=False)

    ma = monthly_agg(df_g, metric)
    if ma.empty:
        st.warning("Эта метрика отсутствует в данных.")
    else:
        ma["value_smooth"] = ma["value"].rolling(smooth, min_periods=1).mean()
        y_col = "value_smooth" if smooth>1 else "value"
        if chart_type=="line":
            fig = px.line(ma, x="ym", y=y_col, markers=True, template="simple_white")
        elif chart_type=="bar":
            fig = px.bar(ma, x="ym", y=y_col, template="simple_white")
        else:
            fig = px.area(ma, x="ym", y=y_col, template="simple_white")
        fig.update_layout(height=420, xaxis_title="Месяц публикации", yaxis_title=metric)
        if logy: fig.update_yaxes(type="log")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Топ-видео")
    kw = st.text_input("Поиск по названию", value="")
    sort_by = st.selectbox("Сортировка", [c for c in ["views","impressions","ctr","watch_hours"] if c in df_g.columns],
                           format_func=lambda x: {"views":"Просмотры","impressions":"Показы","ctr":"CTR","watch_hours":"Часы просмотра"}[x])
    topn = st.slider("Сколько видео показать", 5, 50, 15)
    df_top = df_g.copy()
    if kw.strip():
        df_top = df_top[df_top["title"].str.contains(kw, case=False, na=False)]
    df_top = df_top.sort_values(sort_by, ascending=False).head(topn)

    df_view = df_top.copy()
    df_view["YouTube"] = df_view.apply(yt_link, axis=1)
    if "ctr" in df_view: df_view["CTR"] = df_view["ctr"].round(2).astype(str)+"%"
    if "watch_hours" in df_view and "views" in df_view:
        safe_v = df_view["views"].replace(0,np.nan)
        df_view["AVD"] = ((df_view["watch_hours"]*3600)/safe_v).apply(lambda s: fmt_time_from_hours(s/3600))
    cols_show = [c for c in ["title","views","impressions","CTR","AVD","YouTube","publish_time"] if c in df_view.columns]
    st.dataframe(df_view[cols_show].rename(columns={
        "title":"Название","views":"Просмотры","impressions":"Показы","publish_time":"Публикация"
    }), use_container_width=True)

# ======================
# CHANNEL EXPLORER
# ======================
elif page=="Channel Explorer":
    st.title("🔎 Channel Explorer")
    if not group_names:
        st.info("Добавьте группу во вкладке **Manage Groups**.")
        st.stop()
    g = st.selectbox("Группа", group_names, index=0)
    df_g = groups[g]["df"].copy()

    metric = st.selectbox("Метрика", [m for m in ["views","impressions","watch_hours","ctr"] if m in df_g.columns],
                          format_func=lambda x: {"views":"Просмотры","impressions":"Показы","watch_hours":"Часы просмотра","ctr":"CTR"}[x])
    years_sorted = sorted(df_g["pub_year"].dropna().unique())
    y_from, y_to = st.select_slider("Года публикации", options=years_sorted, value=(years_sorted[0], years_sorted[-1]))
    mask = (df_g["pub_year"]>=y_from) & (df_g["pub_year"]<=y_to)
    df_g = df_g.loc[mask].copy()

    st.subheader("По годам выпуска")
    byyear = by_year_agg(df_g, metric)
    if byyear.empty:
        st.warning("Нет данных этой метрики.")
    else:
        fig = px.bar(byyear.rename(columns={metric:"value"}), x="Год", y="value", template="simple_white",
                     color_discrete_sequence=["#4e79a7"])
        fig.update_layout(height=420, yaxis_title=metric)
        st.plotly_chart(fig, use_container_width=True)

# ======================
# COMPARE GROUPS
# ======================
elif page=="Compare Groups":
    st.title("🆚 Compare Groups")
    if len(group_names)<2:
        st.info("Нужно минимум две группы.")
        st.stop()
    selected = st.multiselect("Выберите группы", group_names, default=group_names[:2])
    if not selected: st.stop()

    records=[]
    for g in selected:
        d = groups[g]["df"]
        v, imp, ctr, avd = kpi_for_df(d)
        records.append({
            "Группа": g,
            "Показы": imp,
            "Просмотры": v,
            "CTR (ср.)": ctr,
            "AVD (ср.)": avd
        })
    table = pd.DataFrame(records)
    if "CTR (ср.)" in table: table["CTR (ср.)"] = table["CTR (ср.)"].apply(lambda x: f"{round(x,2)}%" if pd.notna(x) else "—")
    if "AVD (ср.)" in table: table["AVD (ср.)"] = table["AVD (ср.)"].apply(lambda s: fmt_time_from_hours(s/3600) if pd.notna(s) else "—")
    if "Показы" in table: table["Показы"] = table["Показы"].apply(fmt_int)
    if "Просмотры" in table: table["Просмотры"] = table["Просмотры"].apply(fmt_int)
    st.dataframe(table, use_container_width=True)

# ======================
# MANAGE GROUPS
# ======================
elif page=="Manage Groups":
    st.title("🧰 Manage Groups")
    st.button("Сбросить состояние (очистить всё)", on_click=reset_state, type="secondary")
    if not group_names:
        st.info("Пока нет групп. Добавьте их в сайдбаре (вверху).")
    else:
        for g in group_names:
            with st.expander(f"Группа: {g}", expanded=False):
                df_g = groups[g]["df"]
                allow_dups = groups[g]["allow_dups"]
                st.write(f"Строк: **{len(df_g)}**, колонок: **{df_g.shape[1]}**, дубли разрешены: **{allow_dups}**")
                st.dataframe(df_g.head(20), use_container_width=True)
