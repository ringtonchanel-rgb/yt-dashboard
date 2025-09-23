# app.py — Responsive navbar + Group Analytics → Year Mix
# Кнопки навигации переносятся на новую строку при узком экране.

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ===== Настройки =====
USE_EMOJI = True  # если эмодзи не показываются в системе — поставьте False
ICON_DASH  = "📊 " if USE_EMOJI else ""
ICON_GROUP = "🧩 " if USE_EMOJI else ""
ICON_BRAND = "📺 " if USE_EMOJI else ""

st.set_page_config(page_title="YouTube Analytics Tools", layout="wide")

# ---------- Адаптивные стили ----------
st.markdown("""
<style>
/* уменьшили общий верхний отступ для плотности */
.block-container { padding-top: 0.9rem; }

/* бренд слева */
.yt-brand { font-weight: 700; font-size: 1.02rem; letter-spacing: .1px; }

/* радиогруппа навигации: делаем переносы и компактные кнопки */
[data-baseweb="radio"] > div {
  display: flex !important;
  flex-wrap: wrap;               /* ВАЖНО: перенос на новую строку при нехватке места */
  gap: .45rem .55rem;            /* компактные промежутки */
}

[data-baseweb="radio"] label {
  padding: .28rem .55rem;        /* компактные кнопки */
  border: 1px solid rgba(0,0,0,.08);
  border-radius: 8px;
  font-size: .92rem;             /* чуть мельче шрифт */
  white-space: nowrap;           /* чтобы подписи не «ломались» на 2 строки */
}

/* немного сжали заголовки, чтобы помещались при масштабе 100% */
h3, .stMarkdown h3 { margin-top: .6rem; }
</style>
""", unsafe_allow_html=True)

# ---------- Top "Navbar" ----------
left, right = st.columns([1.0, 3.0])  # больше места под кнопки справа
with left:
    st.markdown(f"<div class='yt-brand'>{ICON_BRAND}YouTube Analytics Tools</div>", unsafe_allow_html=True)

with right:
    nav = st.radio(
        label="Навигация",
        options=[f"{ICON_DASH}Dashboard", f"{ICON_GROUP}Group Analytics"],
        horizontal=True,
        label_visibility="collapsed",
        key="nav_choice",
    )

st.divider()

# ======================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (для «Сравнение по годам»)
# ======================================================================
def _norm(s: str) -> str:
    return s.strip().lower()

MAP = {
    "publish_time": [
        "video publish time","publish time","время публикации видео","дата публикации","publish date"
    ],
    "views": ["views","просмотры"],
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
    return {"publish_time": find_col(df, MAP["publish_time"]),
            "views": find_col(df, MAP["views"])}

def close_enough(a, b, tol=0.12):
    if pd.isna(a) or pd.isna(b):
        return False
    base = max(abs(b), 1e-9)
    return abs(a - b) / base <= tol

# ======================================================================
# ROUTES
# ======================================================================

if nav.endswith("Dashboard"):
    st.sidebar.header("Параметры")
    st.sidebar.info("Раздел **Dashboard** будет наполнен позже. Пока настроек нет.")
    st.subheader("Dashboard")
    st.info("Здесь будут общие метрики канала, KPI, тренды и быстрые инсайты. "
            "Страница создана и готова к наполнению.")

else:  # Group Analytics
    st.sidebar.header("Групповой анализ")
    tool = st.selectbox("Выберите инструмент анализа", ["Сравнение по годам (Year Mix)"])

    if tool.startswith("Сравнение по годам"):
        st.subheader("Сравнение по годам (Year Mix)")

        # Данные
        st.sidebar.markdown("### Данные")
        file = st.sidebar.file_uploader(
            "Загрузите CSV из YouTube Studio", type=["csv"], key="upload_yearmix"
        )
        show_table = st.sidebar.checkbox("Показать таблицу с цифрами", value=False)

        if not file:
            st.info("👆 Загрузите CSV — построю два графика и автокомментарий по годам.")
            st.stop()

        df = pd.read_csv(file)
        df.columns = [c.strip() for c in df.columns]

        # убрать возможные «ИТОГО»
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
        df["_views_num"] = pd.to_numeric(df[views_col], errors="coerce")
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
