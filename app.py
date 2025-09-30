import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io, re, hashlib

# =========================
# Общие настройки
# =========================
USE_EMOJI = True
ICON_DASH  = "📊 " if USE_EMOJI else ""
ICON_GROUP = "🧩 " if USE_EMOJI else ""
ICON_BRAND = "📺 " if USE_EMOJI else ""

st.set_page_config(page_title="YouTube Analytics Tools", layout="wide")

st.sidebar.markdown(
    f"<div style='font-weight:700;font-size:1.05rem;letter-spacing:.1px;'>{ICON_BRAND}YouTube Analytics Tools</div>",
    unsafe_allow_html=True,
)
st.sidebar.divider()

nav = st.sidebar.radio(
    "Навигация",
    options=[f"{ICON_DASH}Dashboard", f"{ICON_GROUP}Group Analytics"],
    label_visibility="visible",
)

st.sidebar.divider()

# =========================
# Утилиты
# =========================
def _norm(s: str) -> str:
    return str(s).strip().lower()

MAP = {
    "publish_time": [
        "video publish time","publish time","время публикации видео","дата публикации","publish date"
    ],
    "views": ["views","просмотры","просмторы","просмотры (views)"],
    "impressions": ["impressions","показы","показы (impressions)","показы значков","показы для значков"],
    "ctr": [
        "impressions click-through rate","ctr","ctr (%)",
        "ctr for thumbnails (%)","ctr для значков","ctr для значков видео (%)",
        "ctr для значков (%)","ctr для значков видео","ctr видео"
    ],
    "avd": [
        "average view duration","avg view duration",
        "средняя продолжительность просмотра",
        "средняя продолжительность просмотра видео",
        "average view duration (hh:mm:ss)"
    ],
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
    if s == "" or s.lower() in {"none","nan"}:
        return np.nan
    s = s.replace(" ", "").replace("\u202f","").replace("\xa0","")
    is_percent = s.endswith("%")
    if is_percent:
        s = s[:-1]
    if "," in s and "." not in s:
        s = s.replace(",", ".")
    try:
        val = float(s)
        return val if not is_percent else val
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

def read_csv_safely(uploaded_file) -> pd.DataFrame | None:
    try:
        return pd.read_csv(uploaded_file)
    except Exception:
        try:
            if hasattr(uploaded_file, "getvalue"):
                data = uploaded_file.getvalue()
            else:
                data = uploaded_file.read()
            return pd.read_csv(io.BytesIO(data), encoding="utf-8-sig")
        except Exception:
            try:
                return pd.read_csv(io.BytesIO(data), encoding="cp1251")
            except Exception:
                return None

def file_hash(uploaded_file) -> str:
    """Стабильный хэш содержимого файла для де-дупликации."""
    if hasattr(uploaded_file, "getvalue"):
        raw = uploaded_file.getvalue()
    else:
        raw = uploaded_file.read()
    return hashlib.md5(raw).hexdigest()

# структура state: groups = [ {"name": str, "files": [ {"name": str, "hash": str, "df": DataFrame, "meta": str} ] } ]
if "groups" not in st.session_state:
    st.session_state["groups"] = []

def all_hashes() -> set[str]:
    hs = set()
    for g in st.session_state["groups"]:
        for f in g["files"]:
            hs.add(f["hash"])
    return hs

def kpis_for_group(group) -> dict:
    total_impr = 0.0
    total_views = 0.0
    ctr_values = []
    avd_vals_sec = []
    for f in group["files"]:
        df = f["df"]
        C = detect_columns(df)
        if C["impressions"] and C["impressions"] in df.columns:
            impr = pd.to_numeric(df[C["impressions"]].apply(to_number), errors="coerce").fillna(0)
            total_impr += float(impr.sum())
        if C["views"] and C["views"] in df.columns:
            views = pd.to_numeric(df[C["views"]].apply(to_number), errors="coerce").fillna(0)
            total_views += float(views.sum())
        if C["ctr"] and C["ctr"] in df.columns:
            ctr_col = df[C["ctr"]].apply(to_number)
            ctr_values.extend(list(ctr_col.dropna().values))
        if C["avd"] and C["avd"] in df.columns:
            avd_sec = df[C["avd"]].apply(parse_duration_to_seconds)
            avd_vals_sec.extend(list(avd_sec.dropna().values))
    avg_ctr = float(np.nanmean(ctr_values)) if ctr_values else np.nan
    avg_avd_sec = float(np.nanmean(avd_vals_sec)) if avd_vals_sec else np.nan
    return dict(
        impressions=int(total_impr),
        views=int(total_views),
        ctr=avg_ctr,
        avd_sec=avg_avd_sec
    )

def concat_groups(groups_idx: list[int]) -> pd.DataFrame:
    """Сшиваем все данные выбранных групп в один df."""
    frames = []
    for idx in groups_idx:
        if idx < 0 or idx >= len(st.session_state["groups"]):
            continue
        for f in st.session_state["groups"][idx]["files"]:
            frames.append(f["df"])
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

# =========================
# DASHBOARD
# =========================
if nav.endswith("Dashboard"):
    st.header("Dashboard")

    # ----------- Добавление новой группы -----------
    with st.sidebar.expander("➕ Добавить группу данных", expanded=True):
        group_name = st.text_input("Название группы (канала)", value=f"Group {len(st.session_state['groups'])+1}")
        files = st.file_uploader(
            "Загрузите один или несколько CSV (все типы отчётов YouTube Studio)",
            type=["csv"],
            accept_multiple_files=True,
            key="add_group_files",
        )
        add_btn = st.button("Добавить группу")

        if add_btn:
            if not group_name.strip():
                st.warning("Введите название группы.")
            elif not files:
                st.warning("Загрузите хотя бы один CSV.")
            else:
                known = all_hashes()
                new_files = []
                skipped = 0
                for f in files:
                    h = file_hash(f)
                    if h in known:
                        skipped += 1
                        continue
                    df = read_csv_safely(f)
                    if df is None or df.empty:
                        continue
                    df.columns = [c.strip() for c in df.columns]
                    new_files.append({"name": f.name, "hash": h, "df": df, "meta": f"✅ {f.name}: {df.shape[0]} строк, {df.shape[1]} колонок."})
                    known.add(h)
                if new_files:
                    st.session_state["groups"].append({"name": group_name.strip(), "files": new_files})
                    if skipped:
                        st.info(f"Группа добавлена. Пропущено дублей: {skipped}.")
                    else:
                        st.success("Группа добавлена.")
                    st.experimental_rerun()
                else:
                    st.error("Ни одного нового файла не добавлено (возможно, все дубли).")

    # ----------- Управление существующими группами -----------
    if not st.session_state["groups"]:
        st.info("Добавьте хотя бы одну группу в сайдбаре.")
    else:
        st.markdown("### Управление группами")
        for gi, g in enumerate(st.session_state["groups"]):
            with st.expander(f"Группа: {g['name']}", expanded=False):
                # Переименование
                new_name = st.text_input("Название", value=g["name"], key=f"rename_{gi}")
                if st.button("Сохранить название", key=f"save_name_{gi}"):
                    g["name"] = new_name.strip() if new_name.strip() else g["name"]
                    st.success("Название сохранено.")
                    st.experimental_rerun()

                # Добавление файлов в группу
                add_more = st.file_uploader(
                    "Добавить отчёты в эту группу",
                    type=["csv"],
                    accept_multiple_files=True,
                    key=f"append_files_{gi}"
                )
                if st.button("Добавить отчёты", key=f"append_btn_{gi}"):
                    if not add_more:
                        st.warning("Выберите файлы.")
                    else:
                        known = all_hashes()
                        added, skipped = 0, 0
                        for f in add_more:
                            h = file_hash(f)
                            if h in known:
                                skipped += 1
                                continue
                            df = read_csv_safely(f)
                            if df is None or df.empty():
                                continue
                            df.columns = [c.strip() for c in df.columns]
                            g["files"].append({"name": f.name, "hash": h, "df": df, "meta": f"✅ {f.name}: {df.shape[0]} строк, {df.shape[1]} колонок."})
                            known.add(h)
                            added += 1
                        if added:
                            st.success(f"Добавлено файлов: {added}. Пропущено дублей: {skipped}.")
                            st.experimental_rerun()
                        else:
                            st.info("Новые файлы не добавлены (возможно, все дубликаты).")

                # Список файлов с возможностью удаления
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
                                st.experimental_rerun()

                st.divider()
                # Удаление группы
                col_del1, col_del2 = st.columns([1,5])
                with col_del1:
                    if st.button("Удалить группу", key=f"del_group_{gi}"):
                        st.session_state["groups"].pop(gi)
                        st.experimental_rerun()

        st.divider()

        # ----------- KPI карточки и сводная таблица -----------
        st.markdown("### Сводка по группам")
        kpi_rows = []
        for gi, g in enumerate(st.session_state["groups"]):
            kp = kpis_for_group(g)
            st.subheader(f"Группа: {g['name']}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Показы (сумма)", f"{kp['impressions']:,}".replace(",", " "))
            c2.metric("Просмотры (сумма)", f"{kp['views']:,}".replace(",", " "))
            ctr_txt = "—" if np.isnan(kp["ctr"]) else f"{kp['ctr']:.2f}%"
            avd_txt = seconds_to_hhmmss(kp["avd_sec"])
            c3.metric("Средний CTR по видео", ctr_txt)
            c4.metric("Средний AVD", avd_txt)

            kpi_rows.append({
                "Группа": g["name"],
                "Показы": kp["impressions"],
                "Просмотры": kp["views"],
                "CTR, % (среднее)": None if np.isnan(kp["ctr"]) else round(kp["ctr"], 2),
                "AVD (ср.)": avd_txt
            })
            st.divider()

        if kpi_rows:
            st.markdown("### Сравнение групп")
            comp_df = pd.DataFrame(kpi_rows)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

# =========================
# GROUP ANALYTICS
# =========================
else:
    st.header("Group Analytics")
    tool = st.sidebar.selectbox("Выберите инструмент анализа", ["Сравнение по годам (Year Mix)"])

    if tool.startswith("Сравнение по годам"):
        st.subheader("Сравнение по годам (Year Mix)")

        source_mode = st.sidebar.radio("Источник данных", ["Группы из Dashboard", "Загрузить файлы"])

        # ---- Источник: группы
        if source_mode == "Группы из Dashboard":
            if not st.session_state["groups"]:
                st.info("Нет групп данных. Сначала добавьте их в Dashboard.")
                st.stop()
            # мультивыбор групп
            group_names = [g["name"] for g in st.session_state["groups"]]
            selected = st.sidebar.multiselect("Выберите одну или несколько групп", group_names, default=group_names[:1])
            if not selected:
                st.info("Выберите хотя бы одну группу.")
                st.stop()
            idxs = [group_names.index(n) for n in selected]
            df = concat_groups(idxs)

        # ---- Источник: новые файлы + возможность сохранить в группу
        else:
            up_files = st.sidebar.file_uploader("Загрузите CSV (можно несколько)", type=["csv"], accept_multiple_files=True, key="ga_upload")
            df_list = []
            if up_files:
                for f in up_files:
                    d = read_csv_safely(f)
                    if d is not None and not d.empty:
                        d.columns = [c.strip() for c in d.columns]
                        df_list.append(d)
            if not df_list:
                st.info("Загрузите хотя бы один CSV.")
                st.stop()
            df = pd.concat(df_list, ignore_index=True)

            # Сохранить их в группу?
            save_opt = st.sidebar.checkbox("Сохранить эти файлы в группу")
            if save_opt:
                choice = st.sidebar.radio("Куда сохранить", ["В существующую группу", "Создать новую"])
                if choice == "В существующую группу":
                    if not st.session_state["groups"]:
                        st.warning("Нет групп. Выберите «Создать новую».")
                    else:
                        names = [g["name"] for g in st.session_state["groups"]]
                        gi = st.sidebar.selectbox("Группа", list(range(len(names))), format_func=lambda i: names[i])
                        if st.sidebar.button("Сохранить"):
                            known = all_hashes()
                            added, skipped = 0, 0
                            for f in up_files:
                                h = file_hash(f)
                                if h in known:
                                    skipped += 1
                                    continue
                                d = read_csv_safely(f)
                                if d is None or d.empty:
                                    continue
                                d.columns = [c.strip() for c in d.columns]
                                st.session_state["groups"][gi]["files"].append({"name": f.name, "hash": h, "df": d, "meta": f"✅ {f.name}: {d.shape[0]} строк, {d.shape[1]} колонок."})
                                known.add(h)
                                added += 1
                            if added:
                                st.success(f"Добавлено файлов: {added}. Пропущено дублей: {skipped}.")
                                st.experimental_rerun()
                else:
                    new_name = st.sidebar.text_input("Название новой группы", value=f"GA Group {len(st.session_state['groups'])+1}")
                    if st.sidebar.button("Создать и сохранить"):
                        known = all_hashes()
                        new_files = []
                        for f in up_files:
                            h = file_hash(f)
                            if h in known:
                                continue
                            d = read_csv_safely(f)
                            if d is None or d.empty:
                                continue
                            d.columns = [c.strip() for c in d.columns]
                            new_files.append({"name": f.name, "hash": h, "df": d, "meta": f"✅ {f.name}: {d.shape[0]} строк, {d.shape[1]} колонок."})
                            known.add(h)
                        if new_files:
                            st.session_state["groups"].append({"name": new_name.strip() or "New Group", "files": new_files})
                            st.success("Группа создана.")
                            st.experimental_rerun()

        # --- дальше построение Year Mix как прежде ---
        # удалим явные "Итоги" если были слиты
        try:
            df = df[~df.apply(lambda r: r.astype(str).str.contains("итог", case=False).any(), axis=1)]
        except Exception:
            pass

        C = detect_columns(df)
        pub_col  = C["publish_time"]
        views_col = C["views"]

        missing = []
        if not (pub_col and pub_col in df.columns): missing.append("Дата публикации")
        if not (views_col and views_col in df.columns): missing.append("Просмотры")
        if missing:
            st.error("Не хватает колонок в данных: " + ", ".join(missing))
            st.stop()

        df[pub_col] = pd.to_datetime(df[pub_col], errors="coerce")
        df = df[df[pub_col].notna()].copy()
        df["_views_num"] = pd.to_numeric(df[views_col].apply(to_number), errors="coerce")
        df["_year"] = df[pub_col].dt.year

        views_year = (df.groupby("_year", as_index=False)["_views_num"]
                        .sum()
                        .rename(columns={"_year":"Год","_views_num":"Суммарное количество просмотров"})
                        .sort_values("Год"))
        count_year = (df.groupby("_year", as_index=False)
                        .size()
                        .rename(columns={"_year":"Год","size":"Количество видео"})
                        .sort_values("Год"))

        if views_year.empty or count_year.empty:
            st.info("Недостаточно данных для построения графиков по годам.")
            st.stop()

        years_list = sorted(views_year["Год"].dropna().astype(int).unique())
        default_ref = 2024 if 2024 in years_list else int(max(years_list))
        ref_year = st.selectbox("Опорный год для текста-аналитики", years_list,
                                index=years_list.index(default_ref))

        c1, c2 = st.columns(2)
        fig1 = px.bar(views_year, x="Год", y="Суммарное количество просмотров",
                      text="Суммарное количество просмотров", template="simple_white")
        fig1.update_traces(marker_color="#4e79a7", texttemplate="%{text:,}", textposition="outside")
        fig1.update_layout(title="Суммарное количество просмотров по годам",
                           xaxis_title="Год публикации",
                           yaxis_title="Суммарное количество просмотров",
                           showlegend=False, margin=dict(l=10,r=10,t=50,b=10), height=430)
        fig1.update_xaxes(type="category", categoryorder="category ascending")
        c1.plotly_chart(fig1, use_container_width=True)

        fig2 = px.bar(count_year, x="Год", y="Количество видео", text="Количество видео", template="simple_white")
        fig2.update_traces(marker_color="#4e79a7", texttemplate="%{text}", textposition="outside")
        fig2.update_layout(title="Количество видео по годам",
                           xaxis_title="Год публикации",
                           yaxis_title="Количество видео",
                           showlegend=False, margin=dict(l=10,r=10,t=50,b=10), height=430)
        fig2.update_xaxes(type="category", categoryorder="category ascending")
        c2.plotly_chart(fig2, use_container_width=True)

        st.markdown("### 🧠 Автокомментарий")
        vy = dict(zip(views_year["Год"], views_year["Суммарное количество просмотров"]))
        cy = dict(zip(count_year["Год"], count_year["Количество видео"]))

        ranking = sorted(vy.items(), key=lambda x: x[1], reverse=True)
        ranking_years = [str(int(y)) for y,_ in ranking[:5]]

        older_sum = sum(v for y,v in vy.items() if y < ref_year)
        ref_sum   = vy.get(ref_year, np.nan)
        prev_year = ref_year - 1 if (ref_year - 1) in vy else None
        views_ref = vy.get(ref_year, np.nan)
        views_prev = vy.get(prev_year, np.nan) if prev_year else np.nan
        cnt_ref = cy.get(ref_year, np.nan)
        cnt_prev = cy.get(prev_year, np.nan) if prev_year else np.nan

        def close_enough(a, b, tol=0.12):
            if pd.isna(a) or pd.isna(b): return False
            base = max(abs(b), 1e-9)
            return abs(a - b)/base <= tol

        parts = []
        parts.append(f"Опорная точка — **{ref_year}**.")
        if ranking_years:
            parts.append("Лидируют по просмотрам: **" + " → ".join(ranking_years) + "**.")
        if not pd.isna(ref_sum) and older_sum > ref_sum:
            total_pair = older_sum + ref_sum
            share_old = f" (≈{older_sum/total_pair*100:.0f}% от «старый+{ref_year}»)" if total_pair>0 else ""
            parts.append(f"**Старый контент** (до {ref_year}) собрал больше просмотров, чем {ref_year}-й год{share_old}.")
        frame = [y for y in [2022, 2023, 2024] if y in vy]
        if len(frame) >= 2:
            vals = [vy[y] for y in frame]
            mx, mn = max(vals), min(vals)
            if mx>0 and (mx-mn)/mx <= 0.15:
                parts.append("В **2022–2024** суммарные просмотры держались примерно на одном уровне (±15%).")
        if prev_year and not any(pd.isna(x) for x in [views_ref,views_prev,cnt_ref,cnt_prev]):
            if close_enough(views_ref, views_prev, 0.12) and cnt_ref>cnt_prev:
                times = cnt_ref/max(cnt_prev,1)
                parts.append(f"При близких просмотрах у {prev_year} и {ref_year} в {ref_year}-м понадобилось больше видео (≈×{times:.1f}).")

        if parts:
            for s in parts: st.markdown("• " + s)
        else:
            st.write("Недостаточно данных для содержательного вывода.")
