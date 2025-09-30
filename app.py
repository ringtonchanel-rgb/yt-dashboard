# app.py — YouTube Analytics Tools
# Dashboard (группы + сохранение изменений) и Group Analytics (Year Mix)
# Дубликаты CSV РАЗРЕШЕНЫ: можно добавлять один и тот же файл сколько угодно раз.

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io, re, hashlib

# --------------------------- UI CONFIG ---------------------------
st.set_page_config(page_title="YouTube Analytics Tools", layout="wide")
USE_EMOJI = True
ICON_DASH  = "📊 " if USE_EMOJI else ""
ICON_GROUP = "🧩 " if USE_EMOJI else ""
ICON_BRAND = "📺 " if USE_EMOJI else ""

st.sidebar.markdown(
    f"<div style='font-weight:700;font-size:1.05rem;letter-spacing:.1px;'>{ICON_BRAND}YouTube Analytics Tools</div>",
    unsafe_allow_html=True,
)
st.sidebar.divider()
nav = st.sidebar.radio("Навигация", [f"{ICON_DASH}Dashboard", f"{ICON_GROUP}Group Analytics"])
st.sidebar.divider()

# --------------------------- HELPERS: columns / parsing ---------------------------
def _norm(s: str) -> str:
    return str(s).strip().lower()

MAP = {
    "publish_time": ["video publish time","publish time","время публикации видео","дата публикации","publish date"],
    "views": ["views","просмотры","просмторы","просмотры (views)"],
    "impressions": ["impressions","показы","показы (impressions)","показы значков","показы для значков"],
    "ctr": ["impressions click-through rate","ctr","ctr (%)","ctr for thumbnails (%)","ctr для значков",
            "ctr для значков видео (%)","ctr для значков (%)","ctr для значков видео","ctr видео"],
    "avd": ["average view duration","avg view duration","средняя продолжительность просмотра",
            "средняя продолжительность просмотра видео","average view duration (hh:mm:ss)"],
    "title": ["title","название видео","video title","видео","название"],
}

def find_col(df: pd.DataFrame, names) -> str | None:
    if isinstance(names, str):
        names = [names]
    by_norm = {_norm(c): c for c in df.columns}
    # точные совпадения
    for n in names:
        nn = _norm(n)
        if nn in by_norm:
            return by_norm[nn]
    # подстроки
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
    if s == "" or s.lower() in {"nan", "none"}:
        return np.nan
    s = s.replace(" ", "").replace("\u202f", "").replace("\xa0", "")
    is_percent = s.endswith("%")
    if is_percent:
        s = s[:-1]
    if "," in s and "." not in s:
        s = s.replace(",", ".")
    try:
        return float(s)
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
        return h * 3600 + m_ * 60 + s_
    m = re.match(r"^(\d+):(\d{2})$", s)
    if m:
        m_, s_ = map(int, m.groups())
        return m_ * 60 + s_
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

# --------------------------- FILE LOADER ---------------------------
def load_uploaded_file(uploaded_file):
    """
    Читаем файл стабильно:
    - единым байтовым буфером (getvalue/read)
    - md5-хэш (теперь лишь для информации, дубли разрешены)
    - пробы кодировок
    """
    raw = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
    h = hashlib.md5(raw).hexdigest()

    df = None
    for enc in (None, "utf-8-sig", "cp1251"):
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=enc) if enc else pd.read_csv(io.BytesIO(raw))
            break
        except Exception:
            df = None

    meta = "❌ не удалось прочитать CSV."
    if df is not None and not df.empty:
        df.columns = [c.strip() for c in df.columns]
        meta = f"✅ {uploaded_file.name}: {df.shape[0]} строк, {df.shape[1]} колонок."
    return {"name": uploaded_file.name, "hash": h, "df": df, "meta": meta}

# --------------------------- STORAGE ---------------------------
if "groups" not in st.session_state:
    st.session_state["groups"] = []   # [{name: str, files: [{name, hash, df, meta}, ...]}]

def concat_groups(indices):
    frames = []
    for i in indices:
        if 0 <= i < len(st.session_state["groups"]):
            for f in st.session_state["groups"][i]["files"]:
                if f["df"] is not None and not f["df"].empty:
                    frames.append(f["df"])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def kpis_for_group(group):
    total_impr = 0.0
    total_views = 0.0
    ctr_vals = []
    avd_vals = []
    for f in group["files"]:
        df = f["df"]
        if df is None or df.empty:
            continue
        C = detect_columns(df)
        if C["impressions"] and C["impressions"] in df.columns:
            total_impr += pd.to_numeric(df[C["impressions"]].apply(to_number), errors="coerce").fillna(0).sum()
        if C["views"] and C["views"] in df.columns:
            total_views += pd.to_numeric(df[C["views"]].apply(to_number), errors="coerce").fillna(0).sum()
        if C["ctr"] and C["ctr"] in df.columns:
            ctr_vals += list(df[C["ctr"]].apply(to_number).dropna().values)
        if C["avd"] and C["avd"] in df.columns:
            avd_vals += list(df[C["avd"]].apply(parse_duration_to_seconds).dropna().values)
    avg_ctr = float(np.nanmean(ctr_vals)) if ctr_vals else np.nan
    avg_avd = float(np.nanmean(avd_vals)) if avd_vals else np.nan
    return dict(impressions=int(total_impr), views=int(total_views), ctr=avg_ctr, avd_sec=avg_avd)

# --------------------------- DASHBOARD ---------------------------
if nav.endswith("Dashboard"):
    st.header("Dashboard")

    # --- Добавить новую группу
    with st.sidebar.expander("➕ Добавить группу данных", expanded=True):
        group_name = st.text_input("Название группы (канала)", value=f"Group {len(st.session_state['groups'])+1}")
        files = st.file_uploader("Загрузите один или несколько CSV", type=["csv"], accept_multiple_files=True, key="add_group_files")
        if st.button("Добавить группу"):
            if not group_name.strip():
                st.warning("Введите название группы.")
            elif not files:
                st.warning("Загрузите хотя бы один CSV.")
            else:
                new_files = []
                for uf in files:
                    pack = load_uploaded_file(uf)
                    if pack["df"] is None or pack["df"].empty:
                        continue
                    # Дубликаты РАЗРЕШЕНЫ — просто добавляем всё
                    new_files.append(pack)
                if new_files:
                    st.session_state["groups"].append({"name": group_name.strip(), "files": new_files})
                    st.success(f"Группа добавлена. Загружено файлов: {len(new_files)}.")
                    st.rerun()
                else:
                    st.error("Не удалось добавить файлы (возможно пустые/повреждены).")

    if not st.session_state["groups"]:
        st.info("Добавьте хотя бы одну группу в сайдбаре.")
    else:
        # --- Управление группами
        st.markdown("### Управление группами")
        for gi, g in enumerate(st.session_state["groups"]):
            with st.expander(f"Группа: {g['name']}", expanded=False):
                # Поля редактирования
                new_name = st.text_input("Название", value=g["name"], key=f"rename_{gi}")
                add_more = st.file_uploader(
                    "Добавить отчёты в эту группу",
                    type=["csv"], accept_multiple_files=True, key=f"append_files_{gi}"
                )

                # ЕДИНАЯ КНОПКА СОХРАНЕНИЯ
                if st.button("Сохранить изменения", key=f"save_group_{gi}"):
                    changed = False

                    # Переименование
                    if new_name.strip() and new_name.strip() != g["name"]:
                        g["name"] = new_name.strip()
                        changed = True

                    # Добавление файлов (дубликаты разрешены)
                    if add_more:
                        added = 0
                        for uf in add_more:
                            pack = load_uploaded_file(uf)
                            if pack["df"] is None or pack["df"].empty:
                                continue
                            g["files"].append(pack)
                            added += 1
                        if added:
                            st.success(f"Добавлено файлов: {added}.")
                            changed = True

                    if changed:
                        st.rerun()
                    else:
                        st.info("Изменений нет — нечего сохранять.")

                # Список файлов + удаление
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
                                st.rerun()

                st.divider()
                # Удаление группы
                if st.button("Удалить группу", key=f"del_group_{gi}"):
                    st.session_state["groups"].pop(gi)
                    st.rerun()

        st.divider()

        # --- KPI по группам
        st.markdown("### Сводка по группам")
        kpi_rows = []
        for gi, g in enumerate(st.session_state["groups"]):
            kp = kpis_for_group(g)
            st.subheader(f"Группа: {g['name']}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Показы (сумма)", f"{kp['impressions']:,}".replace(",", " "))
            c2.metric("Просмотры (сумма)", f"{kp['views']:,}".replace(",", " "))
            c3.metric("Средний CTR по видео", "—" if np.isnan(kp["ctr"]) else f"{kp['ctr']:.2f}%")
            c4.metric("Средний AVD", seconds_to_hhmmss(kp["avd_sec"]))
            kpi_rows.append({
                "Группа": g["name"],
                "Показы": kp["impressions"],
                "Просмотры": kp["views"],
                "CTR, % (среднее)": None if np.isnan(kp["ctr"]) else round(kp["ctr"], 2),
                "AVD (ср.)": seconds_to_hhmmss(kp["avd_sec"]),
            })
            st.divider()

        if kpi_rows:
            st.markdown("### Сравнение групп")
            comp_df = pd.DataFrame(kpi_rows)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

# --------------------------- GROUP ANALYTICS ---------------------------
else:
    st.header("Group Analytics")
    tool = st.sidebar.selectbox("Выберите инструмент анализа", ["Сравнение по годам (Year Mix)"])

    if tool.startswith("Сравнение по годам"):
        st.subheader("Сравнение по годам (Year Mix)")
        source_mode = st.sidebar.radio("Источник данных", ["Группы из Dashboard", "Загрузить файлы"])

        # Источник: группы
        if source_mode == "Группы из Dashboard":
            if not st.session_state["groups"]:
                st.info("Нет групп данных. Сначала добавьте их в Dashboard.")
                st.stop()
            names = [g["name"] for g in st.session_state["groups"]]
            selected = st.sidebar.multiselect("Выберите группы", names, default=names[:1])
            if not selected:
                st.info("Выберите хотя бы одну группу.")
                st.stop()
            idxs = [names.index(n) for n in selected]
            df = concat_groups(idxs)

        # Источник: новые файлы (с опцией сохранить в группу, дубликаты разрешены)
        else:
            up_files = st.sidebar.file_uploader("Загрузите CSV (можно несколько)", type=["csv"], accept_multiple_files=True, key="ga_upload")
            df_list = []
            if up_files:
                for uf in up_files:
                    pack = load_uploaded_file(uf)
                    if pack["df"] is not None and not pack["df"].empty:
                        df_list.append(pack["df"])
            if not df_list:
                st.info("Загрузите хотя бы один CSV.")
                st.stop()
            df = pd.concat(df_list, ignore_index=True)

            # Сохранение в группу при желании (без дедупа)
            if st.sidebar.checkbox("Сохранить эти файлы в группу"):
                mode = st.sidebar.radio("Куда сохранить", ["В существующую группу", "Создать новую"])
                if mode == "В существующую группу":
                    if not st.session_state["groups"]:
                        st.warning("Нет групп. Создайте новую ниже.")
                    else:
                        names = [g["name"] for g in st.session_state["groups"]]
                        gi = st.sidebar.selectbox("Группа", list(range(len(names))), format_func=lambda i: names[i])
                        if st.sidebar.button("Сохранить"):
                            added = 0
                            for uf in up_files:
                                pack = load_uploaded_file(uf)
                                if pack["df"] is None or pack["df"].empty:
                                    continue
                                st.session_state["groups"][gi]["files"].append(pack)
                                added += 1
                            if added:
                                st.success(f"Добавлено файлов: {added}.")
                                st.rerun()
                            else:
                                st.info("Новые файлы не добавлены (пустые/повреждены).")
                else:
                    new_name = st.sidebar.text_input("Название новой группы", value=f"GA Group {len(st.session_state['groups'])+1}")
                    if st.sidebar.button("Создать и сохранить"):
                        new_files = []
                        for uf in up_files:
                            pack = load_uploaded_file(uf)
                            if pack["df"] is None or pack["df"].empty:
                                continue
                            new_files.append(pack)
                        if new_files:
                            st.session_state["groups"].append({"name": new_name.strip() or "New Group", "files": new_files})
                            st.success("Группа создана.")
                            st.rerun()
                        else:
                            st.info("Новые файлы не добавлены (пустые/повреждены).")

        # Очистка строк «ИТОГО»
        try:
            df = df[~df.apply(lambda r: r.astype(str).str.contains("итог", case=False).any(), axis=1)]
        except Exception:
            pass

        C = detect_columns(df)
        pub_col = C["publish_time"]
        views_col = C["views"]
        missing = []
        if not (pub_col and pub_col in df.columns):
            missing.append("Дата публикации")
        if not (views_col and views_col in df.columns):
            missing.append("Просмотры")
        if missing:
            st.error("Не хватает колонок: " + ", ".join(missing))
            st.stop()

        # Приведение типов
        df[pub_col] = pd.to_datetime(df[pub_col], errors="coerce")
        df = df[df[pub_col].notna()].copy()
        df["_views_num"] = pd.to_numeric(df[views_col].apply(to_number), errors="coerce")
        df["_year"] = df[pub_col].dt.year

        # Агрегации
        views_year = (
            df.groupby("_year", as_index=False)["_views_num"].sum()
              .rename(columns={"_year":"Год","_views_num":"Суммарное количество просмотров"})
              .sort_values("Год")
        )
        count_year = (
            df.groupby("_year", as_index=False).size()
              .rename(columns={"_year":"Год","size":"Количество видео"})
              .sort_values("Год")
        )

        if views_year.empty or count_year.empty:
            st.info("Недостаточно данных для построения графиков по годам.")
            st.stop()

        years_list = sorted(views_year["Год"].dropna().astype(int).unique())
        default_ref = 2024 if 2024 in years_list else int(max(years_list))
        ref_year = st.selectbox("Опорный год для текста-аналитики", years_list, index=years_list.index(default_ref))

        # Графики
        c1, c2 = st.columns(2)
        fig1 = px.bar(
            views_year, x="Год", y="Суммарное количество просмотров",
            text="Суммарное количество просмотров", template="simple_white"
        )
        fig1.update_traces(marker_color="#4e79a7", texttemplate="%{text:,}", textposition="outside")
        fig1.update_layout(
            title="Суммарное количество просмотров по годам",
            xaxis_title="Год публикации", yaxis_title="Суммарное количество просмотров",
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
            xaxis_title="Год публикации", yaxis_title="Количество видео",
            showlegend=False, margin=dict(l=10, r=10, t=50, b=10), height=430
        )
        fig2.update_xaxes(type="category", categoryorder="category ascending")
        c2.plotly_chart(fig2, use_container_width=True)

        # Автотекст
        st.markdown("### 🧠 Автокомментарий")
        vy = dict(zip(views_year["Год"], views_year["Суммарное количество просмотров"]))
        cy = dict(zip(count_year["Год"], count_year["Количество видео"]))
        ranking = sorted(vy.items(), key=lambda x: x[1], reverse=True)
        ranking_years = [str(int(y)) for y, _ in ranking[:5]]
        older_sum = sum(v for y, v in vy.items() if y < ref_year)
        ref_sum = vy.get(ref_year, np.nan)
        prev_year = ref_year - 1 if (ref_year - 1) in vy else None
        views_ref = vy.get(ref_year, np.nan)
        views_prev = vy.get(prev_year, np.nan) if prev_year else np.nan
        cnt_ref = cy.get(ref_year, np.nan)
        cnt_prev = cy.get(prev_year, np.nan) if prev_year else np.nan

        def close_enough(a, b, tol=0.12):
            if pd.isna(a) or pd.isna(b):
                return False
            return abs(a - b) / max(abs(b), 1e-9) <= tol

        parts = [f"Опорная точка — **{ref_year}**."]
        if ranking_years:
            parts.append("Лидируют по просмотрам: **" + " → ".join(ranking_years) + "**.")
        if not pd.isna(ref_sum) and older_sum > ref_sum:
            total_pair = older_sum + ref_sum
            share_old = f" (≈{older_sum/total_pair*100:.0f}% от «старый+{ref_year}»)" if total_pair > 0 else ""
            parts.append(f"**Старый контент** (до {ref_year}) собрал больше просмотров, чем {ref_year}-й год{share_old}.")
        frame = [y for y in [2022, 2023, 2024] if y in vy]
        if len(frame) >= 2:
            vals = [vy[y] for y in frame]
            mx = max(vals); mn = min(vals)
            if mx > 0 and (mx - mn) / mx <= 0.15:
                parts.append("В **2022–2024** суммарные просмотры держались примерно на одном уровне (±15%).")
        if prev_year and not any(pd.isna(x) for x in [views_ref, views_prev, cnt_ref, cnt_prev]):
            if close_enough(views_ref, views_prev, 0.12) and cnt_ref > cnt_prev:
                times = cnt_ref / max(cnt_prev, 1)
                parts.append(
                    f"При близких просмотрах у {prev_year} и {ref_year} в {ref_year}-м понадобилось больше видео (≈×{times:.1f})."
                )

        for s in parts:
            st.markdown("• " + s)
            # ---------- Чат с n8n ----------
import os, requests, json
import streamlit as st

N8N_CHAT_URL = os.environ.get("N8N_CHAT_URL")  # ты уже сохранил секрет в App settings → Secrets

st.divider()
st.subheader("🤖 Chat (n8n webhook)")

if not N8N_CHAT_URL:
    st.warning("Секрет N8N_CHAT_URL не задан. Задай его в App settings → Secrets и перезапусти приложение.")
else:
    # Память чата в сессии
    if "chat" not in st.session_state:
        st.session_state.chat = []  # список кортежей: ("user"/"bot", text)

    # Рисуем историю
    for role, text in st.session_state.chat:
        st.chat_message("user" if role == "user" else "assistant").write(text)

    # Ввод нового сообщения
    prompt = st.chat_input("Напишите вопрос…")
    if prompt:
        st.session_state.chat.append(("user", prompt))
        with st.chat_message("assistant"):
            with st.spinner("Отправляю в n8n…"):
                try:
                    payload = {"prompt": prompt, "meta": {"source": "streamlit", "user": "anon"}}
                    r = requests.post(N8N_CHAT_URL, json=payload, timeout=30)
                    r.raise_for_status()
                    # пробуем JSON, иначе текст
                    if r.headers.get("content-type", "").startswith("application/json"):
                        data = r.json()
                        answer = data.get("answer", json.dumps(data, ensure_ascii=False))
                    else:
                        answer = r.text
                except Exception as e:
                    answer = f"Ошибка запроса: {e}"
            st.write(answer)
        st.session_state.chat.append(("bot", answer))


# === 1) ДОБАВЬ / ПРОВЕРЬ ИМПОРТЫ ВВЕРХУ ФАЙЛА ===
import os
import uuid
import requests

# === 2) ДОБАВЬ ПОСЛЕ СВОЕЙ НАВИГАЦИИ В SIDEBAR ===
# Было, например:
# nav = st.sidebar.radio("Навигация", [f"{ICON_DASH}Dashboard", f"{ICON_GROUP}Group Analytics"])
# Заменяем на:
nav = st.sidebar.radio(
    "Навигация",
    [f"{ICON_DASH}Dashboard", f"{ICON_GROUP}Group Analytics", "🤖 Assistant"]
)

# === 3) ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ ЧАТА ===
def _get_n8n_urls_and_headers():
    """
    Берём URL из секрета или env и готовим заголовки.
    Если в n8n включена авторизация по Bearer токену — тоже подставим.
    """
    n8n_url = st.secrets.get("N8N_CHAT_URL") or os.getenv("N8N_CHAT_URL")
    if not n8n_url:
        st.error("Не задан N8N_CHAT_URL в Secrets / переменных окружения.")
        st.stop()

    headers = {"Content-Type": "application/json"}
    token = st.secrets.get("N8N_TOKEN") or os.getenv("N8N_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return n8n_url, headers

def ask_n8n(question: str, history: list[dict] | None = None, user_id: str | None = None) -> dict:
    """
    Делает POST в твой n8n вебхук и возвращает JSON-ответ.
    Ожидается, что твой n8n workflow возвращает, например:
      { "answer": "текст ответа", "meta": { ... } }
    """
    n8n_url, headers = _get_n8n_urls_and_headers()

    payload = {
        "question": question,
        "history": history or [],  # можно пробросить историю, если используешь её в n8n
        "user_id": user_id or str(uuid.uuid4()),
        # можно добавить любые кастомные поля, которые ждёт твой Prepare node
    }

    try:
        resp = requests.post(n8n_url, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        return resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {"answer": resp.text}
    except requests.HTTPError as e:
        return {"answer": f"HTTP error: {e} — {getattr(e.response, 'text', '')}"}
    except requests.RequestException as e:
        return {"answer": f"Network error: {e}"}
    except Exception as e:
        return {"answer": f"Unexpected error: {e}"}

def render_chat_page():
    st.title("🤖 Assistant")
    st.caption("Чат идёт через n8n → OpenAI (Message a model).")

    # Инициализируем историю
    if "chat_msgs" not in st.session_state:
        st.session_state.chat_msgs = []

    # Рисуем накопленные сообщения
    for m in st.session_state.chat_msgs:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # Поле ввода
    user_text = st.chat_input("Напишите вопрос…")
    if user_text:
        # 1) локально отрисуем пользователя
        st.session_state.chat_msgs.append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.markdown(user_text)

        # 2) вызовем n8n
        with st.chat_message("assistant"):
            with st.spinner("Думаю…"):
                # Если в n8n ты используешь историю – можно передать её целиком
                # (или сжать до нужного формата внутри Prepare Messages)
                n8n_resp = ask_n8n(
                    question=user_text,
                    history=st.session_state.chat_msgs,  # можно заменить на [] если история не нужна
                    user_id=st.session_state.get("user_id") or str(uuid.uuid4()),
                )
                answer = n8n_resp.get("answer", "Пустой ответ 🤖")
                st.markdown(answer)

        # 3) докинем ассистента в историю
        st.session_state.chat_msgs.append({"role": "assistant", "content": answer})

        # Кнопка сброса сессии чата (по желанию)
    cols = st.columns([1,1,6])
    with cols[0]:
        if st.button("Очистить диалог"):
            st.session_state.chat_msgs = []
            st.rerun()

# === 4) ВЕТКА РЕНДЕРА ЧАТА ===
if nav == "🤖 Assistant":
    render_chat_page()
