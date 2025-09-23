# app.py — Sidebar Navigation
# DASHBOARD: несколько групп (каналов), много отчётов, автомаппинг колонок и KPI
# GROUP ANALYTICS: Сравнение по годам (как раньше)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io
import re

# ===== Настройки =====
USE_EMOJI = True
ICON_DASH  = "📊 " if USE_EMOJI else ""
ICON_GROUP = "🧩 " if USE_EMOJI else ""
ICON_BRAND = "📺 " if USE_EMOJI else ""

st.set_page_config(page_title="YouTube Analytics Tools", layout="wide")

# ---------------- Sidebar (НАВИГАЦИЯ) ----------------
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

# ======================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ: Автомаппинг колонок/парсинг значений
# ======================================================================

def _norm(s: str) -> str:
    return str(s).strip().lower()

# Возможные названия колонок в разных отчётах (ru/en)
MAP = {
    "publish_time": [
        "video publish time","publish time","время публикации видео","дата публикации","publish date"
    ],
    "views": ["views","просмотры","просмторы","просмотры (views)"],
    "impressions": [
        "impressions","показы","показы (impressions)","показы значков","показы для значков"
    ],
    "ctr": [
        "impressions click-through rate","ctr","ctr (%)",
        "ctr for thumbnails (%)","ctr для значков","ctr для значков видео (%)",
        "ctr для значков (%)","ctr для значков видео","ctr видео"
    ],
    "avd": [
        "average view duration",
        "avg view duration",
        "средняя продолжительность просмотра",
        "средняя продолжительность просмотра видео",
        "average view duration (hh:mm:ss)"
    ],
    "title": ["title","название видео","video title","видео","название"],
}

def find_col(df: pd.DataFrame, names) -> str | None:
    """Ищем колонку: точное совпадение по нормализованному имени, затем подстрока."""
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
    """Парсим '12 345', '5,6%', '5.6%', '1 234' -> float. Возвращаем NaN если не удалось."""
    if x is None:
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip()
    if s == "" or s.lower() in {"none","nan"}:
        return np.nan
    # убираем пробелы, узкие пробелы, нецифровые (кроме знаков . , % :)
    s = s.replace(" ", "").replace("\u202f", "").replace("\xa0", "")
    # процент
    is_percent = s.endswith("%")
    if is_percent:
        s = s[:-1]
    # заменить запятую на точку, если нет точки
    if "," in s and "." not in s:
        s = s.replace(",", ".")
    try:
        val = float(s)
        return val if not is_percent else val  # CTR далее сам приведём к %
    except Exception:
        return np.nan

def parse_duration_to_seconds(x):
    """Парсим AVD: '0:01:47'/'2:45'/'1:12:05' -> сек. Если число — считаем, что уже секунды."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        # иногда в отчётах AVD могут быть в секундах
        return float(x)
    s = str(x).strip()
    if s == "":
        return np.nan
    # Форматы: hh:mm:ss или mm:ss
    m = re.match(r"^(\d+):(\d{2}):(\d{2})$", s)
    if m:
        h, m_, s_ = map(int, m.groups())
        return h*3600 + m_*60 + s_
    m = re.match(r"^(\d+):(\d{2})$", s)
    if m:
        m_, s_ = map(int, m.groups())
        return m_*60 + s_
    # Если прилетело странное — попробуем числом
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
    """Пытаемся читать CSV с BOM/без него, fallback на cp1251/utf-8."""
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

# ======================================================================
# DASHBOARD
# ======================================================================

if nav.endswith("Dashboard"):
    st.header("Dashboard")

    # Хранение наборов (групп) в сессии
    if "groups" not in st.session_state:
        st.session_state["groups"] = []  # [{name:str, dfs:[pd.DataFrame,..], meta:[]}, ...]

    with st.sidebar.expander("➕ Добавить группу данных", expanded=True):
        group_name = st.text_input("Название группы (канала)", value=f"Group {len(st.session_state['groups'])+1}")
        files = st.file_uploader(
            "Загрузите один или несколько отчетов CSV",
            type=["csv"],
            accept_multiple_files=True,
            key="dashboard_files",
            help="Можно загрузить отчёты разных типов из YouTube Studio.",
        )
        add_btn = st.button("Добавить группу")

        if add_btn:
            if not group_name.strip():
                st.warning("Введите название группы.")
            elif not files:
                st.warning("Загрузите хотя бы один CSV.")
            else:
                dfs = []
                metas = []
                for f in files:
                    df = read_csv_safely(f)
                    if df is None or df.empty:
                        metas.append(f"❌ {f.name}: не удалось прочитать CSV или он пуст.")
                        continue
                    df.columns = [c.strip() for c in df.columns]
                    dfs.append(df)
                    metas.append(f"✅ {f.name}: {df.shape[0]} строк, {df.shape[1]} колонок.")
                if dfs:
                    st.session_state["groups"].append({"name": group_name.strip(), "dfs": dfs, "meta": metas})
                    st.success(f"Группа «{group_name}» добавлена ({len(dfs)} файл(а)).")
                else:
                    st.error("Не удалось добавить группу — нет валидных файлов.")

    if st.session_state["groups"]:
        col_clear, col_cnt = st.columns([1,3])
        with col_clear:
            if st.button("🧹 Очистить все группы"):
                st.session_state["groups"].clear()
                st.experimental_rerun()
        with col_cnt:
            st.write(f"Загружено групп: **{len(st.session_state['groups'])}**")

        st.divider()

        # --- Подсчёт KPI по каждой группе ---
        kpi_rows = []   # для общей таблицы сравнения
        for g in st.session_state["groups"]:
            name = g["name"]
            dfs  = g["dfs"]

            total_impr = 0.0
            total_views = 0.0
            ctr_values = []  # средняя по видео (простая)
            avd_vals_sec = []

            # пробежимся по всем загруженным отчётам группы
            for df in dfs:
                C = detect_columns(df)

                # Импрессии и просмотры — суммой
                if C["impressions"] and C["impressions"] in df.columns:
                    impr = pd.to_numeric(df[C["impressions"]].apply(to_number), errors="coerce").fillna(0)
                    total_impr += float(impr.sum())

                if C["views"] and C["views"] in df.columns:
                    views = pd.to_numeric(df[C["views"]].apply(to_number), errors="coerce").fillna(0)
                    total_views += float(views.sum())

                # CTR — среднее по видеозаписям
                if C["ctr"] and C["ctr"] in df.columns:
                    ctr_col = df[C["ctr"]].apply(to_number)  # 5.6 -> 5.6 (%)
                    # иногда в отчётах CTR в долях (0.056); попробуем поправить если < 1 и не все нули
                    # но аккуратно — не будем менять, просто возьмём как есть в процентах
                    ctr_values.extend(list(ctr_col.dropna().values))

                # AVD — среднее по видео (перевод в секунды)
                if C["avd"] and C["avd"] in df.columns:
                    avd_sec = df[C["avd"]].apply(parse_duration_to_seconds)
                    avd_vals_sec.extend(list(avd_sec.dropna().values))

            # агрегаты
            avg_ctr = float(np.nanmean(ctr_values)) if ctr_values else np.nan
            avg_avd_sec = float(np.nanmean(avd_vals_sec)) if avd_vals_sec else np.nan

            # --- Визуализация карточек KPI ---
            st.subheader(f"Группа: {name}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Показы (сумма)", f"{int(total_impr):,}".replace(",", " "))
            c2.metric("Просмотры (сумма)", f"{int(total_views):,}".replace(",", " "))

            ctr_txt = "—" if np.isnan(avg_ctr) else f"{avg_ctr:.2f}%"
            avd_txt = seconds_to_hhmmss(avg_avd_sec)
            c3.metric("Средний CTR по видео", ctr_txt)
            c4.metric("Средний AVD", avd_txt)

            # сохранить для сравнительной таблицы
            kpi_rows.append({
                "Группа": name,
                "Показы": int(total_impr),
                "Просмотры": int(total_views),
                "CTR, % (среднее)": None if np.isnan(avg_ctr) else round(avg_ctr, 2),
                "AVD (ср.)": avd_txt
            })

            # показать служебные сообщения по файлам
            with st.expander(f"Файлы набора «{name}»"):
                for m in g["meta"]:
                    st.write(m)

            st.divider()

        # --- Сводная таблица по всем группам ---
        if kpi_rows:
            st.markdown("### Сравнение групп")
            comp_df = pd.DataFrame(kpi_rows)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

    else:
        st.info("Добавьте хотя бы одну группу данных в сайдбаре, чтобы увидеть KPI.")

# ======================================================================
# GROUP ANALYTICS — Сравнение по годам (Year Mix)
# ======================================================================

else:
    st.header("Group Analytics")
    tool = st.sidebar.selectbox("Выберите инструмент анализа", ["Сравнение по годам (Year Mix)"])

    # ---------------------- YEAR MIX ----------------------
    if tool.startswith("Сравнение по годам"):
        st.subheader("Сравнение по годам (Year Mix)")

        # Данные для инструмента
        st.sidebar.markdown("### Данные")
        file = st.sidebar.file_uploader(
            "Загрузите CSV из YouTube Studio", type=["csv"], key="upload_yearmix"
        )
        show_table = st.sidebar.checkbox("Показать таблицу с цифрами", value=False)

        if not file:
            st.info("👆 Загрузите CSV — построю два графика и автокомментарий по годам.")
            st.stop()

        df = read_csv_safely(file)
        if df is None or df.empty:
            st.error("Не удалось прочитать CSV.")
            st.stop()

        df.columns = [c.strip() for c in df.columns]
        # убрать «ИТОГО»
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
            st.error("Не хватает колонок в файле: " + ", ".join(missing))
            st.stop()

        # подготовка
        df[pub_col] = pd.to_datetime(df[pub_col], errors="coerce")
        df = df[df[pub_col].notna()].copy()
        df["_views_num"] = pd.to_numeric(df[views_col].apply(to_number), errors="coerce")
        df["_year"] = df[pub_col].dt.year

        # агрегации
        views_year = (
            df.groupby("_year", as_index=False)["_views_num"]
              .sum()
              .rename(columns={"_year": "Год", "_views_num": "Суммарное количество просмотров"})
              .sort_values("Год")
        )
        count_year = (
            df.groupby("_year", as_index=False)
              .size()
              .rename(columns={"_year": "Год", "size": "Количество видео"})
              .sort_values("Год")
        )

        if views_year.empty or count_year.empty:
            st.info("Недостаточно данных для построения графиков по годам.")
            st.stop()

        # Опорный год
        years_list = sorted(views_year["Год"].dropna().astype(int).unique())
        default_ref = 2024 if 2024 in years_list else int(max(years_list))
        ref_year = st.selectbox("Опорный год для текста-аналитики", years_list,
                                index=years_list.index(default_ref))

        # --- ГРАФИКИ ---
        c1, c2 = st.columns(2)

        fig1 = px.bar(
            views_year, x="Год", y="Суммарное количество просмотров",
            text="Суммарное количество просмотров", template="simple_white"
        )
        fig1.update_traces(marker_color="#4e79a7", texttemplate="%{text:,}", textposition="outside")
        fig1.update_layout(
            title="Суммарное количество просмотров по годам",
            xaxis_title="Год публикации",
            yaxis_title="Суммарное количество просмотров",
            showlegend=False, margin=dict(l=10, r=10, t=50, b=10), height=430
        )
        fig1.update_xaxes(type="category", categoryorder="category ascending")
        c1.plotly_chart(fig1, use_container_width=True)

        fig2 = px.bar(
            count_year, x="Год", y="Количество видео", text="Количество видео", template="simple_white"
        )
        fig2.update_traces(marker_color="#4e79a7", texttemplate="%{text}", textposition="outside")
        fig2.update_layout(
            title="Количество видео по годам",
            xaxis_title="Год публикации",
            yaxis_title="Количество видео",
            showlegend=False, margin=dict(l=10, r=10, t=50, b=10), height=430
        )
        fig2.update_xaxes(type="category", categoryorder="category ascending")
        c2.plotly_chart(fig2, use_container_width=True)

        # Таблица по желанию
        if show_table:
            st.markdown("### Таблица")
            tbl = pd.merge(views_year, count_year, on="Год", how="outer").sort_values("Год")
            st.dataframe(tbl, use_container_width=True)

        # --- Автотекст ---
        st.markdown("### 🧠 Автокомментарий по данным")
        vy = dict(zip(views_year["Год"], views_year["Суммарное количество просмотров"]))
        cy = dict(zip(count_year["Год"], count_year["Количество видео"]))
        ranking = sorted(vy.items(), key=lambda x: x[1], reverse=True)
        ranking_years = [str(int(y)) for y,_ in ranking[:5]]

        older_sum = sum(v for y, v in vy.items() if y < ref_year)
        ref_sum   = vy.get(ref_year, np.nan)
        prev_year = ref_year - 1 if (ref_year - 1) in vy else None
        views_ref = vy.get(ref_year, np.nan)
        views_prev = vy.get(prev_year, np.nan) if prev_year else np.nan
        cnt_ref = cy.get(ref_year, np.nan)
        cnt_prev = cy.get(prev_year, np.nan) if prev_year else np.nan

        def close_enough(a, b, tol=0.12):
            if pd.isna(a) or pd.isna(b): return False
            base = max(abs(b), 1e-9)
            return abs(a - b) / base <= tol

        parts = []
        parts.append(f"Опорная точка — **{ref_year}**. Ниже — расклад по годам: где больше всего просмотров и сколько видео вышло.")
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
            if mx > 0 and (mx - mn) / mx <= 0.15:
                parts.append("В **2022–2024** суммарные просмотры держались примерно на одном уровне (±15%).")

        if prev_year and not any(pd.isna(x) for x in [views_ref, views_prev, cnt_ref, cnt_prev]):
            if close_enough(views_ref, views_prev, tol=0.12) and cnt_ref > cnt_prev:
                times = cnt_ref / max(cnt_prev, 1)
                parts.append(
                    f"При похожем уровне просмотров у {prev_year} и {ref_year} "
                    f"в {ref_year}-м понадобилось больше видео (≈×{times:.1f}), чтобы удержать результат."
                )

        if parts:
            for s in parts:
                st.markdown("• " + s)
        else:
            st.write("Недостаточно данных для содержательного вывода — загрузите отчёт с несколькими годами.")
