import streamlit as st
import pandas as pd
import numpy as np
import io
import re
from datetime import timedelta
import plotly.express as px

st.set_page_config(page_title="Channelytics", layout="wide")

# ------------------ CSS: карточки/сегменты/шапка ------------------
CUSTOM_CSS = """
<style>
/* Общий фон чуть светлее */
section.main > div { padding-top: 0.5rem !important; }

.header-wrap{
  display:flex; align-items:center; gap:14px; margin:8px 0 4px 0;
}
.avatar{
  width:64px;height:64px; border-radius:14px;
  background:linear-gradient(135deg,#49c6ff,#2f79ff);
  display:flex;align-items:center;justify-content:center;
  color:#fff;font-weight:800;font-size:28px;
}
.channel-info h1{margin:0;font-size:22px;line-height:1.1;}
.channel-info .handle{opacity:.7; font-size:14px;}
.badge{background:#f2f4f7;border-radius:999px;padding:4px 10px;font-size:12px;margin-left:6px;}
.sub-badges{display:flex;gap:6px;align-items:center;}

.kpi-row{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:10px 0 2px 0;}
.kpi-card{
  background:#fff;border:1px solid #f0f0f0;border-radius:12px;padding:14px 16px;
  box-shadow:0 1px 3px rgba(16,24,40,.06);
}
.kpi-card h3{margin:0;font-size:12px;opacity:.7;font-weight:600;}
.kpi-value{font-size:26px;font-weight:800;margin-top:6px;}
.kpi-delta{font-size:12px;margin-top:4px;}
.delta-up{color:#12b76a;font-weight:700;}
.delta-down{color:#f04438;font-weight:700;}
.delta-zero{opacity:.6}

.segment{
  background:#fff;border:1px solid #e6e8ec;border-radius:10px;display:inline-flex;gap:0;overflow:hidden;
}
.segment button{
  border:none;padding:8px 12px;font-size:13px;background:transparent;cursor:pointer;
}
.segment button.active{background:#111827;color:#fff;}
.segment button:hover{background:#f5f5f6}

.card{background:#fff;border:1px solid #f0f0f0;border-radius:12px;padding:14px 16px;
      box-shadow:0 1px 3px rgba(16,24,40,.06);}
.card h3{margin:0 0 10px 0;font-size:14px;opacity:.7}
.muted{opacity:.7;font-size:12px}

/* донат-пирог справа */
.two-cols{display:grid;grid-template-columns:2fr 1fr;gap:14px;}
</style>
"""
st.write(CUSTOM_CSS, unsafe_allow_html=True)

# ------------------ Безопасная инициализация state ------------------
if "groups" not in st.session_state or not isinstance(st.session_state["groups"], dict):
    st.session_state["groups"] = {}   # {name: {"df": DataFrame, "allow_dups": bool}}

def reset_state():
    st.session_state["groups"] = {}
    st.success("State cleared.")

# ------------------ Нормализация и парсинг ------------------
def _norm(s: str) -> str:
    return str(s).strip().lower()

COLMAP = {
    "publish_time": ["video publish time", "publish time", "publish date", "upload date", "время публикации", "дата"],
    "title": ["title", "video title", "название", "content", "контент"],
    "video_id": ["video id", "id", "ид"],
    "video_link": ["youtube link", "link", "ссылка", "url"],
    "views": ["views", "просмотры"],
    "impressions": ["impressions", "показы"],
    "ctr": ["ctr", "impressions click-through rate", "ctr для значков"],
    "watch_hours": ["watch time (hours)", "время просмотра (часы)"],
    "watch_minutes": ["watch time (minutes)", "время просмотра (мин)"],
    "duration": ["duration", "длительность"]
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

def parse_many(files, allow_dups=True):
    dfs, notes = [], []
    for uf in files:
        raw = uf.getvalue()
        df = None
        for enc in (None, "utf-8-sig", "cp1251"):
            try:
                df = pd.read_csv(io.BytesIO(raw), encoding=enc) if enc else pd.read_csv(io.BytesIO(raw))
                break
            except: pass
        if df is None or df.empty:
            notes.append(f"❌ {uf.name}: не удалось прочитать CSV")
            continue

        cols = detect_columns(df)
        if not cols["publish_time"]:
            notes.append(f"⚠️ {uf.name}: нет даты публикации — пропускаю")
            continue

        out = pd.DataFrame()
        out["publish_time"] = pd.to_datetime(df[cols["publish_time"]], errors="coerce")
        out = out.dropna(subset=["publish_time"])
        if cols["title"]: out["title"] = df[cols["title"]].astype(str)
        if cols["video_id"]: out["video_id"] = df[cols["video_id"]].astype(str)
        if cols["video_link"]: out["video_link"] = df[cols["video_link"]].astype(str)

        if cols["views"]: out["views"] = pd.to_numeric(df[cols["views"]].apply(to_num), errors="coerce")
        if cols["impressions"]: out["impressions"] = pd.to_numeric(df[cols["impressions"]].apply(to_num), errors="coerce")
        if cols["ctr"]: out["ctr"] = pd.to_numeric(df[cols["ctr"]].apply(to_num), errors="coerce")
        if cols["watch_hours"]:
            out["watch_hours"] = pd.to_numeric(df[cols["watch_hours"]].apply(to_num), errors="coerce")
        elif cols["watch_minutes"]:
            out["watch_hours"] = pd.to_numeric(df[cols["watch_minutes"]].apply(to_num), errors="coerce")/60.0
        if cols["duration"]:
            # поддержка «10:05» / «605» сек / минуты
            dur_raw = df[cols["duration"]].astype(str).str.strip()
            def parse_dur(s):
                if ":" in s:
                    parts = [int(p or 0) for p in s.split(":")[-2:]]
                    m, s2 = (parts[0], parts[1]) if len(parts)==2 else (0, parts[0])
                    return m*60 + s2
                return to_num(s)
            out["duration_sec"] = dur_raw.apply(parse_dur)

        out["pub_date"] = out["publish_time"].dt.date
        dfs.append(out)
        notes.append(f"✅ {uf.name}: {out.shape[0]} строк")

    if not dfs: return None, notes
    big = pd.concat(dfs, ignore_index=True)
    if not allow_dups and "title" in big:
        before = len(big)
        big = big.drop_duplicates(subset=["title","publish_time"])
        notes.append(f"↪️ удалены дубли: {before-len(big)}")
    return big, notes

# ------------------ Sidebar: Навигация + группы ------------------
st.sidebar.markdown("### 📊 YouTube Analytics Tools")
page = st.sidebar.radio("Навигация", ["Channelytics", "Manage Groups"], index=0)

with st.sidebar.expander("➕ Добавить/обновить группу", expanded=(page=="Manage Groups")):
    with st.form("add_group_form", clear_on_submit=False):
        gname = st.text_input("Название группы (канала)", value="")
        uploaded = st.file_uploader("Загрузите CSV (1..N)", type=["csv"], accept_multiple_files=True)
        allow_dups = st.checkbox("Разрешать дубли", value=False)
        ok = st.form_submit_button("Сохранить")
    if ok:
        if not gname.strip(): st.warning("Дайте имя группе.")
        elif not uploaded: st.warning("Прикрепите CSV.")
        else:
            df_parsed, notes = parse_many(uploaded, allow_dups=allow_dups)
            for n in notes: st.write(n)
            if df_parsed is not None and not df_parsed.empty:
                st.session_state["groups"][gname] = {"df": df_parsed, "allow_dups": allow_dups}
                st.success(f"Группа «{gname}» сохранена: {df_parsed.shape[0]} строк.")

groups = st.session_state["groups"]
group_names = sorted(groups.keys())

# ------------------ KPI utils ------------------
def kpi_for_df(dff):
    v = dff["views"].sum() if "views" in dff else np.nan
    imp = dff["impressions"].sum() if "impressions" in dff else np.nan
    ctr = dff["ctr"].dropna().mean() if "ctr" in dff else np.nan
    subs = dff["subs"].dropna().sum() if "subs" in dff else np.nan  # на случай отчётов с подписчиками
    return v, imp, ctr, subs

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

# ------------------ CHANNELYTICS ------------------
if page == "Channelytics":
    st.markdown("⚠️ _Внимание: в CSV обычно есть дата **публикации**. KPI за 7D/28D/… здесь — **по опубликованным роликам** в периоде, а не фактические просмотры по дням из YouTube API._")

    if not group_names:
        st.info("Добавьте хотя бы одну группу во вкладке **Manage Groups**.")
        st.stop()

    colA, colB = st.columns([3,1])
    with colA:
        g = st.selectbox("Выберите канал/группу", group_names, index=0)
    with colB:
        rpm = st.number_input("RPM ($ на 1000 просмотров)", min_value=0.0, max_value=200.0, value=2.0, step=0.5)

    df = groups[g]["df"].copy()
    if df.empty:
        st.warning("В этой группе нет данных.")
        st.stop()

    # ---------- «Шапка» канала ----------
    # берём первые буквы названия для аватара
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

    # ---------- сегмент времени ----------
    today = df["publish_time"].max() if "publish_time" in df else pd.Timestamp.today()
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

    cur, cur_range = period_slice(df, today, days)
    prev, prev_range = previous_slice(df, today, days)

    # ---------- KPI карточки ----------
    cur_views, cur_impr, cur_ctr, cur_subs = kpi_for_df(cur)
    prev_views, prev_impr, prev_ctr, prev_subs = (kpi_for_df(prev) if prev is not None else (np.nan,np.nan,np.nan,np.nan))

    rev_cur = (cur_views/1000.0)*rpm if pd.notna(cur_views) else np.nan
    rev_prev = (prev_views/1000.0)*rpm if pd.notna(prev_views) else np.nan

    dv, cls_v = fmt_delta(cur_views, prev_views)
    ds, cls_s = fmt_delta(cur_subs, prev_subs)
    dr, cls_r = fmt_delta(rev_cur, rev_prev)
    dc, cls_c = fmt_delta(cur_ctr, prev_ctr)

    st.markdown('<div class="kpi-row">', unsafe_allow_html=True)
    st.markdown(f"""
      <div class="kpi-card">
        <h3>VIEWS ({seg})</h3>
        <div class="kpi-value">{fmt_int(cur_views)}</div>
        <div class="kpi-delta {cls_v}">{dv}</div>
      </div>
    """, unsafe_allow_html=True)
    if not pd.isna(cur_subs):
        st.markdown(f"""
          <div class="kpi-card">
            <h3>SUBS ({seg})</h3>
            <div class="kpi-value">{fmt_int(cur_subs)}</div>
            <div class="kpi-delta {cls_s}">{ds}</div>
          </div>
        """, unsafe_allow_html=True)
    st.markdown(f"""
      <div class="kpi-card">
        <h3>EST REV ({seg})</h3>
        <div class="kpi-value">${fmt_int(rev_cur)}</div>
        <div class="kpi-delta {cls_r}">{dr}</div>
      </div>
    """, unsafe_allow_html=True)
    if "ctr" in df and not pd.isna(cur_ctr):
        st.markdown(f"""
          <div class="kpi-card">
            <h3>CTR AVG ({seg})</h3>
            <div class="kpi-value">{round(cur_ctr,2)}%</div>
            <div class="kpi-delta {cls_c}">{dc}</div>
          </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------- Тренд + боковые карточки ----------
    # по возможности по дням, иначе по месяцам
    df_trend = cur.copy()
    if "publish_time" in df_trend:
        if df_trend["publish_time"].dt.normalize().nunique() > 1:
            freq = "D"
            df_trend["bucket"] = df_trend["publish_time"].dt.date
        else:
            freq = "M"
            df_trend["bucket"] = df_trend["publish_time"].dt.to_period("M").astype(str)
    else:
        st.warning("Нет даты публикации — тренд недоступен.")
        freq = None

    st.markdown('<div class="two-cols">', unsafe_allow_html=True)

    # Левая — график
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<h3>Views trend</h3>", unsafe_allow_html=True)
    if freq:
        trend = df_trend.groupby("bucket")["views"].sum().reset_index()
        xcol = "bucket"
        fig = px.area(trend, x=xcol, y="views", template="simple_white")
        fig.update_layout(height=360, xaxis_title="", yaxis_title="Views")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Нет данных для тренда.")
    st.markdown('</div>', unsafe_allow_html=True)

    # Правая — most recent + long/shorts
    st.markdown('<div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<h3>Most recent video</h3>", unsafe_allow_html=True)
    if not cur.empty:
        last = cur.sort_values("publish_time", ascending=False).iloc[0]
        title = last.get("title", "—")
        link = last.get("video_link") or (f"https://www.youtube.com/watch?v={last.get('video_id')}" if pd.notna(last.get("video_id")) else None)
        st.write(f"**{title}**")
        st.write(f"Published: {pd.to_datetime(last['publish_time']).date()}")
        st.write(f"Views: {fmt_int(last.get('views'))}")
        if link:
            st.markdown(f"[Open on YouTube]({link})")
    else:
        st.write("—")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<h3>Long vs Shorts</h3>", unsafe_allow_html=True)
    if "duration_sec" in cur:
        short = (cur["duration_sec"]<=60).sum()
        lng = (cur["duration_sec"]>60).sum()
        pie = pd.DataFrame({"type":["Shorts","Longs"], "count":[short,lng]})
        pfig = px.pie(pie, names="type", values="count", color="type",
                      color_discrete_map={"Shorts":"#ef4444","Longs":"#4f46e5"})
        pfig.update_layout(height=260, legend_title=None)
        st.plotly_chart(pfig, use_container_width=True)
    else:
        st.write("Нет длительности — не могу разделить на Longs/Shorts.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # two-cols

# ------------------ MANAGE GROUPS ------------------
elif page == "Manage Groups":
    st.title("🧰 Manage Groups")
    if st.button("Сбросить состояние"):
        reset_state()
        st.experimental_rerun()

    if not group_names:
        st.info("Пока нет групп. Добавьте их слева в «Добавить/обновить группу».")
    else:
        for g in group_names:
            with st.expander(f"Группа: {g}", expanded=False):
                df = groups[g]["df"]
                st.write(f"Строк: **{len(df)}**, колонок: **{df.shape[1]}**")
                st.dataframe(df.head(50), use_container_width=True)
                if st.button(f"Удалить группу «{g}»", key=f"del_{g}"):
                    groups.pop(g, None)
                    st.session_state["groups"] = groups
                    st.experimental_rerun()
