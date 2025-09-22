import streamlit as st
import pandas as pd
import plotly.express as px

# -------------------- Страница --------------------
st.set_page_config(page_title="YouTube Dashboard 🚀", layout="wide")
st.markdown("<h1 style='text-align:center'>📊 YouTube Dashboard 🚀</h1>", unsafe_allow_html=True)
st.write("Аналитика YouTube-канала: просмотры, CTR, удержание, доход и другие ключевые метрики.")

# -------------------- Сайдбар ---------------------
st.sidebar.header("⚙️ Настройки")
uploaded_file = st.sidebar.file_uploader("Загрузите CSV из YouTube Studio", type=["csv"])
n_videos = st.sidebar.slider("Сколько последних видео показывать:", 3, 50, 10)

# Метрики (возможные варианты имен колонок, RU/EN)
METRICS_MAP = {
    "Просмотры": ["views", "просмотры"],
    "CTR": ["impressions click-through rate", "ctr", "impressions click-through rate (%)", "ctr для значков видео"],
    "AVD (средняя продолжительность просмотра)": ["average view duration", "средняя продолжительность просмотра"],
    "Длительность видео": ["duration", "продолжительность"],
    "Доход": ["estimated partner revenue", "расчетный доход", "расчётный доход"],
    "Подписчики": ["subscribers", "подписчики"],
    "Показы": ["impressions", "показы"],
    "RPM (доход за 1000 просмотров)": ["rpm", "доход за 1000 показов", "доход на тысячу просмотров"]
}
# по умолчанию включим «Просмотры», чтобы не было пустого выбора
selected_metrics = st.sidebar.multiselect(
    "Выберите метрики:",
    list(METRICS_MAP.keys()),
    default=["Просмотры"]
)

show_top = st.sidebar.checkbox("Показать ТОП-5 видео", value=True)
show_scatter = st.sidebar.checkbox("Scatter-графики (сравнение метрик)", value=True)

# -------------------- Утилиты ---------------------
def normalize(s: str) -> str:
    return s.strip().lower()

def find_col(df: pd.DataFrame, possible_names: list[str]) -> str | None:
    """
    Возвращает имя колонки из df, если она «похожа» на один из вариантов в possible_names.
    Сравнение регистронезависимое и по «вхождению» (contains).
    """
    if df is None or df.empty:
        return None
    cols_norm = {normalize(c): c for c in df.columns}
    # прямое совпадение
    for pname in possible_names:
        pn = normalize(pname)
        if pn in cols_norm:
            return cols_norm[pn]
    # contains
    for pname in possible_names:
        pn = normalize(pname)
        for c in df.columns:
            if pn in normalize(c):
                return c
    return None

def detect_title_and_id(df: pd.DataFrame) -> tuple[str|None, str|None]:
    title_col = find_col(df, ["название видео", "title", "video title", "название"])
    id_col    = find_col(df, ["video id", "external video id", "контент", "content", "id видео"])
    return title_col, id_col

def shorten(text: str, n: int = 40) -> str:
    t = str(text) if text is not None else ""
    return (t[:n] + "…") if len(t) > n else t

# -------------------- Основная логика ---------------------
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    # аккуратно приводим названия колонок
    df.columns = [c.strip() for c in df.columns]

    # определяем основные колонки
    title_col, id_col = detect_title_and_id(df)

    # сортируем по дате публикации, если есть
    publish_col = find_col(df, ["время публикации видео", "video publish time", "publish time", "дата публикации"])
    if publish_col:
        df[publish_col] = pd.to_datetime(df[publish_col], errors="coerce")
        df = df.sort_values(publish_col, ascending=False)

    # берём последние N
    df = df.head(n_videos).copy()

    # добавляем ссылку на YouTube, если есть ID
    if id_col:
        df_link_col = "YouTube Link"
        df[df_link_col] = df[id_col].apply(lambda x: f"https://www.youtube.com/watch?v={x}")
    else:
        df_link_col = None

    # сформируем базовые колонки для таблицы — только те, что реально есть
    base_cols = []
    if title_col and title_col in df.columns: base_cols.append(title_col)
    if id_col and id_col in df.columns:       base_cols.append(id_col)
    if df_link_col and df_link_col in df.columns: base_cols.append(df_link_col)

    # найдём реальные колонки для выбранных метрик
    metric_cols = []
    for m in selected_metrics:
        col = find_col(df, METRICS_MAP[m])
        if col and col not in metric_cols:
            metric_cols.append(col)

    # ---- Таблица (безопасно) ----
    st.subheader("📋 Таблица всех метрик")
    available_cols = [c for c in (base_cols + metric_cols) if c in df.columns]
    if available_cols:
        st.dataframe(df[available_cols], use_container_width=True)
    else:
        st.warning("Не нашёл колонок для отображения (проверь файл и выбор метрик).")

    # ---- Графики по выбранным метрикам ----
    # для читаемости подписей сделаем сокращённый заголовок
    if title_col and title_col in df.columns:
        df["__title_short__"] = df[title_col].apply(lambda x: shorten(x, 38))
        x_axis = "__title_short__"
    elif id_col and id_col in df.columns:
        x_axis = id_col
    else:
        x_axis = None  # графики не строим без оси X

    if x_axis:
        for m in selected_metrics:
            y_col = find_col(df, METRICS_MAP[m])
            if y_col:
                st.subheader(f"{m} по видео")
                fig = px.bar(
                    df,
                    x=x_axis,
                    y=y_col,
                    text=y_col,
                    hover_data=[id_col] if id_col else None,
                )
                fig.update_traces(texttemplate="%{text}", textposition="outside", cliponaxis=False)
                fig.update_layout(xaxis_tickangle=-35, height=480, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"Метрика «{m}» в файле не найдена — пропускаю.")
    else:
        st.info("Не удалось определить колонку для оси X (ни названия, ни ID). Табличная часть доступна.")

    # ---- ТОП-5 видео по первой выбранной метрике ----
    if show_top and selected_metrics:
        first_metric_col = find_col(df, METRICS_MAP[selected_metrics[0]])
        if first_metric_col:
            st.subheader("🏆 ТОП-5 видео по выбранной метрике")
            top5 = df.sort_values(first_metric_col, ascending=False).head(5)
            cols_to_show = [c for c in [title_col, id_col, df_link_col, first_metric_col] if c and c in top5.columns]
            st.table(top5[cols_to_show])
        else:
            st.info("Для ТОП-5 не нашёл колонку по первой выбранной метрике.")

    # ---- Scatter: сравнение двух метрик ----
    if show_scatter and len(selected_metrics) >= 2:
        col_x = find_col(df, METRICS_MAP[selected_metrics[0]])
        col_y = find_col(df, METRICS_MAP[selected_metrics[1]])
        if col_x and col_y:
            st.subheader("🔗 Сравнение метрик (Scatter)")
            size_col = find_col(df, METRICS_MAP["Просмотры"])
            fig = px.scatter(
                df,
                x=col_x,
                y=col_y,
                size=size_col if size_col else None,
                color=df[title_col] if title_col else (df[id_col] if id_col else None),
                hover_name=df[title_col] if title_col else None,
                hover_data=[id_col] if id_col else None,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Для Scatter нужно выбрать как минимум две метрики, которые есть в файле.")

else:
    st.info("👆 Загрузите CSV-файл из YouTube Studio, затем включите нужные метрики слева.")
