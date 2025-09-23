import streamlit as st
import pandas as pd
import numpy as np
import io
import re
import os
from datetime import timedelta
import plotly.express as px

# ================== APP CONFIG ==================
st.set_page_config(page_title="Channelytics", layout="wide")

# ================== CSS (твоя тема) ==================
CUSTOM_CSS = """
<style>
section.main > div { padding-top: 0.5rem !important; }

.header-wrap{ display:flex; align-items:center; gap:14px; margin:8px 0 4px 0; }
.avatar{ width:64px;height:64px; border-radius:14px;
  background:linear-gradient(135deg,#49c6ff,#2f79ff);
  display:flex;align-items:center;justify-content:center;
  color:#fff;font-weight:800;font-size:28px; }
.channel-info h1{margin:0;font-size:22px;line-height:1.1;}
.channel-info .handle{opacity:.7; font-size:14px;}
.badge{background:#f2f4f7;border-radius:999px;padding:4px 10px;font-size:12px;margin-left:6px;}
.sub-badges{display:flex;gap:6px;align-items:center;}

.kpi-row{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:10px 0 2px 0;}
.kpi-card{ background:#fff;border:1px solid #f0f0f0;border-radius:12px;padding:14px 16px;
  box-shadow:0 1px 3px rgba(16,24,40,.06); }
.kpi-card h3{margin:0;font-size:12px;opacity:.7;font-weight:600;}
.kpi-value{font-size:26px;font-weight:800;margin-top:6px;}
.kpi-delta{font-size:12px;margin-top:4px;}
.delta-up{color:#12b76a;font-weight:700;}
.delta-down{color:#f04438;font-weight:700;}
.delta-zero{opacity:.6}

.segment{ background:#fff;border:1px solid #e6e8ec;border-radius:10px;display:inline-flex;gap:0;overflow:hidden; }
.segment button{ border:none;padding:8px 12px;font-size:13px;background:transparent;cursor:pointer; }
.segment button.active{background:#111827;color:#fff;}
.segment button:hover{background:#f5f5f6}

.card{background:#fff;border:1px solid #f0f0f0;border-radius:12px;padding:14px 16px;
      box-shadow:0 1px 3px rgba(16,24,40,.06);}
.card h3{margin:0 0 10px 0;font-size:14px;opacity:.7}
.muted{opacity:.7;font-size:12px}
.two-cols{display:grid;grid-template-columns:2fr 1fr;gap:14px;}
</style>
"""
st.write(CUSTOM_CSS, unsafe_allow_html=True)

# ================== СХЕМА STATE (мульти-отчёты + доходы) ==================
# st.session_state["groups"] = {
#   group_name: {
#       "reports": [ {"name": str, "df": DataFrame}, ... ],
#       "revenues": [ {"name": str, "df": DataFrame}, ... ],
#       "allow_dups": bool
#   }, ...
# }

if "groups" not in st.session_state or not isinstance(st.session_state["groups"], dict):
    st.session_state["groups"] = {}

def ensure_group_shape():
    """Совместимость со старым форматом {'df': ...}."""
    for g, val in list(st.session_state["groups"].items()):
        if isinstance(val, dict) and "reports" in val:
            continue
        if isinstance(val, dict) and "df" in val:
            st.session_state["groups"][g] = {
                "reports": [{"name": f"{g}_legacy.csv", "df": val["df"]}],
                "revenues": [],
                "allow_dups": bool(val.get("allow_dups", False)),
            }
        else:
            st.session_state["groups"][g] = {"reports": [], "revenues": [], "allow_dups": False}
ensure_group_shape()

def reset_state():
    st.session_state["groups"] = {}
    st.success("State cleared.")

# ================== ПАРСИНГ / НОРМАЛИЗАЦИЯ ==================
def _norm(s: str) -> str:
    return str(s).strip().lower()

COLMAP = {
    "publish_time": ["video publish time","publish time","publish date","upload date","время публикации","дата"],
    "title": ["title","video title","название","content","контент"],
    "video_id": ["video id","id","ид"],
    "video_link": ["youtube link","link","ссылка","url"],
    "views": ["views","просмотры"],
    "impressions": ["impressions","показы"],
    "ctr": ["ctr","impressions click-through rate","ctr для значков"],
    "watch_hours": ["watch time (hours)","время просмотра (часы)"],
    "watch_minutes": ["watch time (minutes)","время просмотра (мин)"],
    "duration": ["duration","длительность"],
    "shorts": ["shorts","is shorts","шортс","короткое видео"],
    "format": ["format","тип контента"],
    "revenue": ["estimated revenue","estimated partner revenue","доход"]
}

def find_col(df, names):
    pool = {_norm(c): c for c in df.columns}
    for n in names:
        n = _norm(n)
        if n in pool: return pool[n]
    for n in names:
        n = _norm(n)
        for c in df.columns:
            if n in _norm(c): return c
    return None

def detect_columns(df): return {k: find_col(df, v) for k, v in COLMAP.items()}

def to_num(x):
    if x is None: return np.nan
    if isinstance(x,(int,float,np.number)): return float(x)
    s = str(x).strip().replace("\u202f","").replace("\xa0","").replace(" ", "")
    if s.endswith("%"): s = s[:-1]
    if "," in s and "." not in s: s = s.replace(",", ".")
    try: return float(s)
    except: return np.nan

def parse_duration_to_seconds(val):
    if pd.isna(val): return np.nan
    s = str(val).strip()
    if s.isdigit(): return float(s)
    m = re.match(r'(?:(\d+)\s*h)?\s*(?:(\d+)\s*m)?\s*(?:(\d+)\s*s)?', s, re.I)
    if m and any(m.groups()):
        h = int(m.group(1)) if m.group(1) else 0
        mm = int(m.group(2)) if m.group(2) else 0
        ss = int(m.group(3)) if m.group(3) else 0
        return float(h*3600 + mm*60 + ss)
    parts = s.split(":")
    try:
        if len(parts) == 3:
            h, mm, ss = map(int, parts); return float(h*3600 + mm*60 + ss)
        if len(parts) == 2:
            mm, ss = map(int, parts);    return float(mm*60 + ss)
    except: pass
    return np.nan

def parse_one_file(uploaded_file, allow_dups=True):
    raw = uploaded_file.getvalue()
    df = None
    for enc in (None, "utf-8-sig", "cp1251"):
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=enc) if enc else pd.read_csv(io.BytesIO(raw))
            break
        except: pass
    if df is None or df.empty:
        return None, f"❌ {uploaded_file.name}: не удалось прочитать CSV"

    cols = detect_columns(df)
    if not cols["publish_time"]:
        return None, f"⚠️ {uploaded_file.name}: нет даты публикации — пропускаю"

    out = pd.DataFrame()
    out["publish_time"] = pd.to_datetime(df[cols["publish_time"]], errors="coerce")
    out = out.dropna(subset=["publish_time"])
    if cols["title"]: out["title"] = df[cols["title"]].astype(str)
    if cols["video_id"]: out["video_id"] = df[cols["video_id"]].astype(str).str.strip()
    if cols["video_link"]: out["video_link"] = df[cols["video_link"]].astype(str)

    if cols["views"]: out["views"] = pd.to_numeric(df[cols["views"]].apply(to_num), errors="coerce")
    if cols["impressions"]: out["impressions"] = pd.to_numeric(df[cols["impressions"]].apply(to_num), errors="coerce")
    if cols["ctr"]:
        out["ctr"] = pd.to_numeric(df[cols["ctr"]].apply(to_num), errors="coerce")
        if out["ctr"].dropna().max() <= 1.0: out["ctr"] = out["ctr"] * 100.0
    if cols["watch_hours"]:
        out["watch_hours"] = pd.to_numeric(df[cols["watch_hours"]].apply(to_num), errors="coerce")
    elif cols["watch_minutes"]:
        out["watch_hours"] = pd.to_numeric(df[cols["watch_minutes"]].apply(to_num), errors="coerce")/60.0

    if cols["duration"]:
        dur_raw = df[cols["duration"]].astype(str).str.strip()
        out["duration_sec"] = dur_raw.apply(parse_duration_to_seconds)
    else:
        out["duration_sec"] = np.nan

    out["format"] = np.nan
    if cols["shorts"]:
        short_col = df[cols["shorts"]].astype(str).str.lower()
        out.loc[short_col.isin(["1","true","да","yes","y","short","shorts"]), "format"] = "vertical"
    if cols["format"]:
        fmt_col = df[cols["format"]].astype(str).str.lower()
        out.loc[fmt_col.str.contains("short"), "format"] = "vertical"
    out.loc[out["format"].isna() & (out["duration_sec"] <= 60), "format"] = "vertical"
    out["format"] = out["format"].fillna("horizontal")

    if cols["revenue"]:
        out["revenue"] = pd.to_numeric(df[cols["revenue"]].apply(to_num), errors="coerce")

    if not allow_dups and "title" in out:
        out = out.drop_duplicates(subset=["title","publish_time"])

    out["pub_date"] = out["publish_time"].dt.date
    return out, f"✅ {uploaded_file.name}: {out.shape[0]} строк"

def attach_revenue(base_df: pd.DataFrame, revenue_packs):
    if not revenue_packs: return base_df
    df = base_df.copy()
    df["revenue_ext"] = np.nan
    for pack in revenue_packs:
        r = pack.get("df")
        if not isinstance(r, pd.DataFrame): continue
        cols = [c.lower() for c in r.columns]
        if "video_id" in cols and "revenue" in cols:
            r2 = r.rename(columns={r.columns[cols.index("video_id")]: "video_id",
                                   r.columns[cols.index("revenue")]: "revenue"}).copy()
            r2["video_id"] = r2["video_id"].astype(str).str.strip()
            r2["revenue"] = r2["revenue"].apply(to_num)
            df = df.merge(r2[["video_id","revenue"]], on="video_id", how="left", suffixes=("", "_ext"))
            df["revenue_ext"] = df["revenue_ext"].fillna(df["revenue_ext"])
        elif "date" in cols and "revenue" in cols:
            r2 = r.rename(columns={r.columns[cols.index("date")]: "date",
                                   r.columns[cols.index("revenue")]: "revenue"}).copy()
            r2["date"] = pd.to_datetime(r2["date"], errors="coerce")
            daily = r2.groupby("date", as_index=False)["revenue"].sum()
            if "publish_time" in df.columns:
                df["pub_date_dt"] = pd.to_datetime(df["pub_date"])
                df = df.merge(daily, left_on="pub_date_dt", right_on="date", how="left", suffixes=("", "_rday"))
                df["revenue_ext"] = df["revenue_ext"].fillna(df["revenue_rday"])
                df.drop(columns=["date","pub_date_dt","revenue_rday"], inplace=True, errors="ignore")

    if "revenue" in df.columns:
        df["revenue_final"] = df["revenue"].fillna(df["revenue_ext"])
    else:
        df["revenue_final"] = df["revenue_ext"]
    return df

# ================== Sidebar ==================
st.sidebar.markdown("### 📊 YouTube Analytics Tools")

page = st.sidebar.radio("Навигация", ["Channelytics", "Manage Groups", "AI Assistant"], index=0)

with st.sidebar.expander("➕ Добавить/обновить группу", expanded=(page!="Channelytics")):
    with st.form("add_group_form", clear_on_submit=False):
        gname = st.text_input("Название группы (канала)", value="")
        uploaded = st.file_uploader("Загрузите CSV отчёты (1..N)", type=["csv"], accept_multiple_files=True)
        uploaded_rev = st.file_uploader("CSV с доходом (опционально, можно несколько)", type=["csv"], accept_multiple_files=True)
        allow_dups = st.checkbox("Разрешать дубли в отчётах", value=False)
        ok = st.form_submit_button("Сохранить")
    if ok:
        if not gname.strip():
            st.warning("Дайте имя группе.")
        elif not uploaded and not uploaded_rev:
            st.warning("Прикрепите хотя бы один файл (отчёт или доход).")
        else:
            st.session_state["groups"].setdefault(gname, {"reports": [], "revenues": [], "allow_dups": allow_dups})
            st.session_state["groups"][gname]["allow_dups"] = allow_dups
            for uf in uploaded or []:
                df_parsed, note = parse_one_file(uf, allow_dups=allow_dups)
                st.write(note)
                if df_parsed is not None and not df_parsed.empty:
                    st.session_state["groups"][gname]["reports"].append({"name": uf.name, "df": df_parsed})
            for rf in uploaded_rev or []:
                raw = rf.getvalue()
                try:
                    rdf = pd.read_csv(io.BytesIO(raw))
                except Exception:
                    rdf = None
                if isinstance(rdf, pd.DataFrame) and not rdf.empty:
                    st.session_state["groups"][gname]["revenues"].append({"name": rf.name, "df": rdf})
                    st.write(f"💰 Доход: {rf.name} загружен ({len(rdf)} строк).")
                else:
                    st.write(f"❌ Не удалось прочитать доход: {rf.name}")
            st.success(f"Группа «{gname}» сохранена/обновлена.")

groups = st.session_state["groups"]
group_names = sorted(groups.keys())

# ================== KPI / утилиты ==================
def kpi_for_df(dff):
    v = dff["views"].sum() if "views" in dff else np.nan
    imp = dff["impressions"].sum() if "impressions" in dff else np.nan
    ctr = dff["ctr"].dropna().mean() if "ctr" in dff else np.nan
    return v, imp, ctr

def period_slice(df, end_date, days):
    if days == 0:  # Max
        return df, None
    start = end_date - timedelta(days=days)
    return df[df["publish_time"].between(start, end_date)], (start, end_date)

def previous_slice(df, end_date, days):
    if days == 0: return None, None
    start_prev = end_date - timedelta(days=days*2)
    end_prev = end_date - timedelta(days=days)
    return df[df["publish_time"].between(start_prev, end_prev)], (start_prev, end_prev)

def fmt_int(n):
    try: return f"{int(round(float(n))):,}".replace(",", " ")
    except: return "—"

def fmt_delta(cur, prev):
    if pd.isna(cur) or pd.isna(prev): return "—", "delta-zero"
    diff = cur - prev
    if abs(prev) < 1e-9:
        return f"+{fmt_int(diff)}", "delta-up" if diff>0 else "delta-down"
    pct = diff/prev*100
    if diff>0: return f"+{fmt_int(diff)} (+{pct:.1f}%)", "delta-up"
    if diff<0: return f"{fmt_int(diff)} ({pct:.1f}%)", "delta-down"
    return "0 (0%)", "delta-zero"

def apply_format_filter(df, fmt_value):
    if fmt_value == "vertical":
        return df[df["format"] == "vertical"] if "format" in df else df
    if fmt_value == "horizontal":
        return df[df["format"] == "horizontal"] if "format" in df else df
    return df

# ================== CHANNELYTICS (дашборд) ==================
if page == "Channelytics":
    st.markdown("⚠️ _Важно: 7D/28D/… здесь — **по датам публикации роликов**, а не по датам просмотров (как в нативной YouTube Analytics)._")

    if not group_names:
        st.info("Добавьте хотя бы одну группу во вкладке **Manage Groups**.")
        st.stop()

    colA, colB = st.columns([3,1])
    with colA:
        g = st.selectbox("Выберите канал/группу", group_names, index=0)
    with colB:
        rpm = st.number_input("RPM ($ на 1000 просмотров)", min_value=0.0, max_value=200.0, value=2.0, step=0.5)

    group = groups[g]
    reports = group["reports"]
    revpacks = group.get("revenues", [])
    if not reports:
        st.warning("В этой группе нет отчётов.")
        st.stop()

    # Шапка
    initials = "".join([w[0] for w in re.sub(r"[^A-Za-zА-Яа-я0-9 ]","", g).split()[:2]]).upper() or "YT"
    st.markdown(
        f"""
        <div class="header-wrap">
          <div class="avatar">{initials}</div>
          <div class="channel-info">
            <h1>{g}</h1>
            <div class="handle">@{re.sub(r'\\W','', g.lower())}</div>
          </div>
          <div class="sub-badges"><span class="badge">Channelytics</span></div>
        </div>
        """, unsafe_allow_html=True
    )

    fmt_filter = st.radio("Формат контента", ["all","horizontal","vertical"], horizontal=True, index=0)

    all_pub = pd.concat([p["df"][["publish_time"]] for p in reports if "publish_time" in p["df"]], ignore_index=True)
    today = all_pub["publish_time"].max() if not all_pub.empty else pd.Timestamp.today()

    seg = st.session_state.get("seg", "28D")
    seg_cols = st.columns([1,1,1,1,1,6])
    with seg_cols[0]:
        if st.button("7D", key="seg7", use_container_width=True): seg="7D"
    with seg_cols[1]:
        if st.button("28D", key="seg28", use_container_width=True): seg="28D"
    with seg_cols[2]:
        if st.button("3M", key="seg3m", use_container_width=True): seg="3M"
    with seg_cols[3]:
        if st.button("1Y", key="seg1y", use_container_width=True): seg="1Y"
    with seg_cols[4]:
        if st.button("Max", key="segmax", use_container_width=True): seg="Max"
    st.session_state["seg"] = seg
    days_map = {"7D":7, "28D":28, "3M":90, "1Y":365, "Max":0}
    days = days_map[seg]

    st.subheader("Сводка по каждому отчёту (без суммирования)")
    rows, combined_cur, combined_prev = [], [], []
    for pack in reports:
        df0 = attach_revenue(pack["df"], revpacks)
        df0 = apply_format_filter(df0, fmt_filter)
        cur, _ = period_slice(df0, today, days)
        prev, _ = previous_slice(df0, today, days)
        v, i, c = kpi_for_df(cur)
        rows.append({
            "Отчёт": pack["name"],
            "Видео (шт.)": len(cur),
            "Просмотры": v,
            "Показы": i,
            "CTR, % (ср.)": c,
            "Доход": cur["revenue_final"].sum() if "revenue_final" in cur else np.nan
        })
        combined_cur.append(cur)
        if prev is not None: combined_prev.append(prev)

    seg_df = pd.DataFrame(rows)
    if "Доход" in seg_df and seg_df["Доход"].notna().sum() == 0:
        seg_df.drop(columns=["Доход"], inplace=True)

    st.dataframe(
        seg_df.style.format({"Просмотры":"{:,.0f}","Показы":"{:,.0f}","CTR, % (ср.)":"{:.2f}"}).hide(axis="index"),
        use_container_width=True, height=280
    )

    show_combined = st.toggle("Показать общий обзор (агрегировано для KPI/графиков)", value=True)
    if show_combined:
        cur_all = pd.concat(combined_cur, ignore_index=True) if combined_cur else pd.DataFrame()
        prev_all = pd.concat(combined_prev, ignore_index=True) if combined_prev else None
        if not cur_all.empty:
            cur_all = apply_format_filter(cur_all, fmt_filter)
            cur_views, cur_impr, cur_ctr = kpi_for_df(cur_all)
            if prev_all is not None and not prev_all.empty:
                prev_views, prev_impr, prev_ctr = kpi_for_df(prev_all)
            else:
                prev_views = prev_impr = prev_ctr = np.nan
            rpm = st.session_state.get("rpm_override", 2.0) or 2.0
            rev_cur = (cur_views/1000.0)*rpm if pd.notna(cur_views) else np.nan
            rev_prev = (prev_views/1000.0)*rpm if pd.notna(prev_views) else np.nan
            dv, cls_v = fmt_delta(cur_views, prev_views)
            di, cls_i = fmt_delta(cur_impr, prev_impr)
            dr, cls_r = fmt_delta(rev_cur, rev_prev)
            dc, cls_c = fmt_delta(cur_ctr, prev_ctr)
            st.markdown('<div class="kpi-row">', unsafe_allow_html=True)
            st.markdown(f"""<div class="kpi-card"><h3>VIEWS ({seg})</h3><div class="kpi-value">{fmt_int(cur_views)}</div><div class="kpi-delta {cls_v}">{dv}</div></div>""", unsafe_allow_html=True)
            st.markdown(f"""<div class="kpi-card"><h3>IMPRESSIONS ({seg})</h3><div class="kpi-value">{fmt_int(cur_impr)}</div><div class="kpi-delta {cls_i}">{di}</div></div>""", unsafe_allow_html=True)
            st.markdown(f"""<div class="kpi-card"><h3>EST REV ({seg})</h3><div class="kpi-value">${fmt_int(rev_cur)}</div><div class="kpi-delta {cls_r}">{dr}</div></div>""", unsafe_allow_html=True)
            if not pd.isna(cur_ctr):
                st.markdown(f"""<div class="kpi-card"><h3>CTR AVG ({seg})</h3><div class="kpi-value">{round(cur_ctr,2)}%</div><div class="kpi-delta {cls_c}">{dc}</div></div>""", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            df_trend = cur_all.copy()
            if "publish_time" in df_trend:
                if df_trend["publish_time"].dt.normalize().nunique() > 1:
                    freq = "D"; df_trend["bucket"] = df_trend["publish_time"].dt.date
                else:
                    freq = "M"; df_trend["bucket"] = df_trend["publish_time"].dt.to_period("M").astype(str)
            else:
                freq = None

            st.markdown('<div class="two-cols">', unsafe_allow_html=True)
            st.markdown('<div class="card"><h3>Views trend</h3>', unsafe_allow_html=True)
            if freq:
                trend = df_trend.groupby("bucket")["views"].sum().reset_index()
                fig = px.area(trend, x="bucket", y="views", template="simple_white")
                fig.update_layout(height=360, xaxis_title="", yaxis_title="Views")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Нет данных для тренда.")
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div>', unsafe_allow_html=True)
            st.markdown('<div class="card"><h3>Most recent video</h3>', unsafe_allow_html=True)
            if not cur_all.empty:
                last = cur_all.sort_values("publish_time", ascending=False).iloc[0]
                title = last.get("title", "—")
                link = last.get("video_link") or (f"https://www.youtube.com/watch?v={last.get('video_id')}" if pd.notna(last.get("video_id")) else None)
                st.write(f"**{title}**")
                st.write(f"Published: {pd.to_datetime(last['publish_time']).date()}")
                st.write(f"Views: {fmt_int(last.get('views'))}")
                if link: st.markdown(f"[Open on YouTube]({link})")
            else:
                st.write("—")
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="card"><h3>Long vs Shorts</h3>', unsafe_allow_html=True)
            if "duration_sec" in cur_all:
                short = (cur_all["duration_sec"]<=60).sum()
                lng = (cur_all["duration_sec"]>60).sum()
                pie = pd.DataFrame({"type":["Shorts","Longs"], "count":[short,lng]})
                pfig = px.pie(pie, names="type", values="count", color="type",
                              color_discrete_map={"Shorts":"#ef4444","Longs":"#4f46e5"})
                pfig.update_layout(height=260, legend_title=None)
                st.plotly_chart(pfig, use_container_width=True)
            else:
                st.write("Нет длительности — не могу разделить на Longs/Shorts.")
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

# ================== MANAGE GROUPS ==================
elif page == "Manage Groups":
    st.title("🧰 Manage Groups")
    if st.button("Сбросить состояние"):
        reset_state(); st.experimental_rerun()

    if not group_names:
        st.info("Пока нет групп. Добавьте их слева в «Добавить/обновить группу».")
    else:
        for g in group_names:
            grp = groups[g]
            with st.expander(f"Группа: {g}", expanded=False):
                st.write(f"Отчётов: **{len(grp['reports'])}**, доходов: **{len(grp.get('revenues', []) )}**")
                for i, pack in enumerate(list(grp["reports"])):
                    st.markdown(f"**Отчёт:** {pack['name']}  ·  строк: {len(pack['df'])}")
                    st.dataframe(pack["df"].head(30), use_container_width=True)
                    cols = st.columns(3)
                    with cols[0]:
                        if st.button("Удалить отчёт", key=f"del_rep_{g}_{i}"):
                            groups[g]["reports"].pop(i); st.experimental_rerun()
                    with cols[1]:
                        st.download_button("Скачать CSV", data=pack["df"].to_csv(index=False).encode("utf-8"),
                                           file_name=f"{pack['name']}_normalized.csv", mime="text/csv")
                    with cols[2]:
                        st.caption("")
                st.markdown("---")
                st.write("**Файлы доходов:**")
                for j, rpack in enumerate(list(grp.get("revenues", []))):
                    st.markdown(f"• {rpack['name']}  ·  строк: {len(rpack['df'])}")
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("Удалить доход", key=f"del_rev_{g}_{j}"):
                            groups[g]["revenues"].pop(j); st.experimental_rerun()
                    with c2:
                        st.download_button("Скачать доход CSV",
                                           data=rpack["df"].to_csv(index=False).encode("utf-8"),
                                           file_name=f"{rpack['name']}",
                                           mime="text/csv")
                st.markdown("---")
                if st.button(f"Удалить всю группу «{g}»", key=f"del_group_{g}"):
                    groups.pop(g, None); st.experimental_rerun()

# ================== AI ASSISTANT (чаты с ИИ) ==================
elif page == "AI Assistant":
    st.title("🤖 AI Assistant")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []  # list[{"role":"user/assistant/system","content":str}]

    if not group_names:
        st.info("Добавьте хотя бы одну группу во вкладке **Manage Groups** для контекстного анализа.")

    # ---- Выбор контекста данных ----
    left, right = st.columns([3,1])
    with left:
        g = st.selectbox("Группа для контекста (опционально)", ["(нет)"] + group_names, index=0)
    with right:
        seg = st.selectbox("Период", ["7D","28D","3M","1Y","Max"], index=1)
    fmt_filter = st.radio("Формат", ["all","horizontal","vertical"], horizontal=True, index=0)
    attach_ctx = st.checkbox("Прикреплять текущий срез данных к каждому запросу", value=True)

    # ---- Настройки LLM (через Secrets/ENV) ----
    st.markdown("**Модель ИИ:** использует OpenAI-совместимый API.")
    api_key = st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = st.secrets.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    model = st.secrets.get("OPENAI_MODEL") or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini"
    temperature = st.slider("Temperature", 0.0, 1.2, 0.3, 0.1)

    if not api_key:
        st.warning("Добавь `OPENAI_API_KEY` (и при желании `OPENAI_MODEL`, `OPENAI_BASE_URL`) в **Streamlit Secrets**.")

    # ---- Сбор контекста ----
    def build_context_text():
        if g == "(нет)" or g not in groups or not groups[g]["reports"]:
            return "no_context"
        group = groups[g]
        reports = group["reports"]
        revpacks = group.get("revenues", [])
        # крайняя дата
        all_pub = pd.concat([p["df"][["publish_time"]] for p in reports if "publish_time" in p["df"]], ignore_index=True)
        end_date = all_pub["publish_time"].max() if not all_pub.empty else pd.Timestamp.today()
        days_map = {"7D":7,"28D":28,"3M":90,"1Y":365,"Max":0}
        days = days_map[seg]
        # текущий агрегированный срез
        parts = []
        combined = []
        for pack in reports:
            df0 = attach_revenue(pack["df"], revpacks)
            df0 = apply_format_filter(df0, fmt_filter)
            cur, _ = period_slice(df0, end_date, days)
            if not cur.empty:
                cur["__report__"] = pack["name"]
                combined.append(cur)
        if not combined:
            return f"context(group={g}, seg={seg}, fmt={fmt_filter}): empty"
        cur_all = pd.concat(combined, ignore_index=True)
        v, i, c = kpi_for_df(cur_all)
        top = cur_all.sort_values("views", ascending=False).head(15)
        # компактный JSON-лайт (строки усечём)
        def short(s, n=80):
            s = str(s) if pd.notna(s) else ""
            return s if len(s)<=n else s[:n-1]+"…"
        top_rows = [
            {
                "publish": str(pd.to_datetime(r["publish_time"]).date()) if "publish_time" in r else "",
                "title": short(r.get("title","")),
                "views": int(r.get("views") or 0),
                "impr": int(r.get("impressions") or 0) if "impressions" in cur_all else None,
                "ctr": round(float(r.get("ctr")),2) if pd.notna(r.get("ctr", np.nan)) else None,
                "fmt": r.get("format",""),
                "id": r.get("video_id","")
            }
            for _, r in top.iterrows()
        ]
        ctx = {
            "group": g,
            "period": seg,
            "format": fmt_filter,
            "kpi": {"views_sum": int(v or 0), "impr_sum": int(i or 0) if pd.notna(i) else None,
                    "ctr_avg": round(float(c),2) if pd.notna(c) else None,
                    "videos": int(len(cur_all))},
            "top_videos": top_rows
        }
        return ctx

    # ---- Вызов LLM (OpenAI-совместимый) ----
    def call_llm(messages, stream=True):
        """
        messages = [{"role":"system/user/assistant","content":"..."}]
        """
        if not api_key:
            return "Нет API-ключа. Добавь OPENAI_API_KEY в Secrets."
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url=base_url)
            # Используем chat.completions (широко совместимы)
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                stream=stream,
            )
            if stream:
                full = ""
                with st.chat_message("assistant"):
                    placeholder = st.empty()
                    for chunk in resp:
                        delta = chunk.choices[0].delta.content or ""
                        if delta:
                            full += delta
                            placeholder.markdown(full)
                return full
            else:
                text = resp.choices[0].message.content
                return text
        except Exception as e:
            return f"Ошибка вызова LLM: {e}"

    # ---- Рендер истории ----
    for m in st.session_state["chat_history"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # ---- Обработчик ввода ----
    prompt = st.chat_input("Спроси про канал, тренды, идеи контента, гипотезы…")
    if prompt:
        # USER
        st.session_state["chat_history"].append({"role":"user","content":prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # SYSTEM + CONTEXT
        sys = {
            "role":"system",
            "content":(
                "Ты — аналитик YouTube. Говори кратко, по делу, с числами. "
                "Если дан контекст, используй его для выводов и идей тестов. "
                "Если данные неполные (например, только дата публикации), обязательно поясняй ограничения."
            )
        }
        msgs = [sys]
        # прикрепим контекст, если включено
        if attach_ctx and group_names:
            ctx = build_context_text()
            msgs.append({"role":"system","content":f"DATA_CONTEXT:\n{ctx}"})

        msgs += st.session_state["chat_history"][-6:]  # последние сообщения (не раздуваем токены)

        # ASSISTANT
        answer = call_llm(msgs, stream=True)
        st.session_state["chat_history"].append({"role":"assistant","content":answer})
