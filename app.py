# app.py — Year Mix + Auto Narrative
# Один экран: 2 графика по годам + умный текст-вывод, который подстраивается под данные.

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ---------------- Page ----------------
st.set_page_config(page_title="YouTube Dashboard — Year Mix", layout="wide")
st.markdown("<h1 style='text-align:center'>📊 Годовой микс: просмотры и количество видео</h1>", unsafe_allow_html=True)
st.caption("Загрузи CSV из YouTube Studio. Ниже — два графика по годам и автокомментарий, который подстраивается под данные.")

# ---------------- Sidebar ----------------
st.sidebar.header("⚙️ Данные")
file = st.sidebar.file_uploader(
    "Загрузите CSV (подходит выгрузка из Advanced mode YouTube Studio)", type=["csv"]
)
show_table = st.sidebar.checkbox("Показать таблицу с цифрами", value=False)

# ---------------- Helpers ----------------
def _norm(s: str) -> str: return s.strip().lower()

MAP = {
    "publish_time": [
        "video publish time","publish time","время публикации видео","дата публикации","publish date"
    ],
    "views": ["views","просмотры"],
}

def find_col(df: pd.DataFrame, names) -> str | None:
    if isinstance(names, str): names = [names]
    by_norm = {_norm(c): c for c in df.columns}
    for n in names:
        nn = _norm(n)
        if nn in by_norm: return by_norm[nn]
    for n in names:
        nn = _norm(n)
        for c in df.columns:
            if nn in _norm(c): return c
    return None

def detect_columns(df: pd.DataFrame):
    return {"publish_time": find_col(df, MAP["publish_time"]),
            "views": find_col(df, MAP["views"])}

def pct(a, b):
    if b == 0 or pd.isna(b): return np.nan
    return 100 * (a - b) / b

def close_enough(a, b, tol=0.15):
    """Почти одинаково (в пределах tol=15%)."""
    if pd.isna(a) or pd.isna(b) or b == 0: return False
    return abs(a - b) / max(b, 1e-9) <= tol

def year_or_none(x):
    try: return int(x)
    except: return None

# ---------------- Main ----------------
if not file:
    st.info("👆 Загрузите CSV-файл — построю графики и напишу вывод автоматически.")
    st.stop()

df = pd.read_csv(file)
df.columns = [c.strip() for c in df.columns]

# убрать возможные строки 'ИТОГО'
try:
    df = df[~df.apply(lambda r: r.astype(str).str.contains("итог", case=False).any(), axis=1)]
except Exception:
    pass

C = detect_columns(df)
pub_col = C["publish_time"]
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

# опорный год: 2024 если есть, иначе максимальный
years_list = sorted(views_year["Год"].dropna().astype(int).unique())
default_ref = 2024 if 2024 in years_list else int(max(years_list))
ref_year = st.sidebar.selectbox("Опорный год для текста-аналитики", years_list, index=years_list.index(default_ref))

# ---------------- Charts ----------------
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

if show_table:
    st.markdown("### Таблица")
    tbl = pd.merge(views_year, count_year, on="Год", how="outer").sort_values("Год")
    st.dataframe(tbl, use_container_width=True)

# ---------------- Auto-Narrative ----------------
st.markdown("### 🧠 Автокомментарий по данным")

# словари по годам
vy = dict(zip(views_year["Год"], views_year["Суммарное количество просмотров"]))
cy = dict(zip(count_year["Год"], count_year["Количество видео"]))

# рейтинг по просмотрам
ranking = sorted(vy.items(), key=lambda x: x[1], reverse=True)
ranking_years = [str(int(y)) for y,_ in ranking[:5]]

# «старый» vs «свежий» контент
older_sum = sum(v for y, v in vy.items() if y < ref_year)
ref_sum   = vy.get(ref_year, np.nan)

# сравнение соседних лет ref vs ref-1
prev_year = ref_year - 1 if (ref_year - 1) in vy else None
views_ref = vy.get(ref_year, np.nan)
views_prev = vy.get(prev_year, np.nan) if prev_year else np.nan
cnt_ref = cy.get(ref_year, np.nan)
cnt_prev = cy.get(prev_year, np.nan) if prev_year else np.nan

# текст
p = []

# 1) вводная
p.append(
    f"В качестве опорной точки возьмём **{ref_year}**-й год"
    + (" — в наборе есть полный календарный год." if ref_year == 2024 else ".")
)

# 2) рейтинг
if ranking_years:
    p.append(
        "По суммарным просмотрам лидируют годы (в порядке убывания): **" +
        " → ".join(ranking_years) + "**."
    )

# 3) старый контент перформит?
if not pd.isna(ref_sum) and older_sum > ref_sum:
    share_old = older_sum / (older_sum + ref_sum) * 100 if (older_sum + ref_sum) > 0 else np.nan
    p.append(
        f"Суммарно **старый контент** (видео до {ref_year} года) собрал больше просмотров, чем контент {ref_year}-го года "
        + (f"(≈{share_old:.0f}% от пары «старый+{ref_year}»)." if not pd.isna(share_old) else ".")
    )

# 4) «2022–2024 держались на одном уровне» — обобщённая логика
frame = [y for y in [2022, 2023, 2024] if y in vy]
if len(frame) >= 2:
    vals = [vy[y] for y in frame]
    mx, mn = max(vals), min(vals)
    if mx > 0 and (mx - mn) / mx <= 0.15:  # в пределах 15%
        p.append("В разрезе **2022–2024** суммарные просмотры держались примерно на одном уровне (±15%).")

# 5) «чтобы достичь уровня предыдущего года, понадобилось больше видео»
if prev_year and not any(pd.isna(x) for x in [views_ref, views_prev, cnt_ref, cnt_prev]):
    if close_enough(views_ref, views_prev, tol=0.12) and cnt_ref > cnt_prev:
        times = cnt_ref / max(cnt_prev, 1)
        p.append(
            f"При **схожем уровне просмотров** у {prev_year} и {ref_year} "
            f"в {ref_year}-м понадобилось **больше видео** (≈×{times:.1f}), чтобы удержать результат."
        )

# вывод
if p:
    for s in p:
        st.markdown("• " + s)
else:
    st.write("Данных недостаточно для содержательного вывода — загрузите отчёт с несколькими годами.")

st.caption("Текст генерируется автоматически и подстраивается под загруженный файл. Порог «≈ одинаково» — 12–15%.")
