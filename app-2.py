import urllib.parse

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import roc_auc_score, roc_curve

st.set_page_config(page_title="AI-детектор / AI-детектор", page_icon="🧬", layout="centered")

NEAR_MARGIN = 0.10  # |p - threshold| below this = low-confidence zone
MIN_COVERAGE = 0.70  # uploaded profile must cover this share of model genes


@st.cache_resource
def load_artifacts():
    return joblib.load("multi_drug_artifacts.pkl")


artifacts = load_artifacts()

# ----------------------------------------------------------------------------
# Texts (kk / ru)
# ----------------------------------------------------------------------------
TXT = {
    "ru": {
        "title": "AI-детектор фармакорезистентности",
        "caption": "Прогноз чувствительности опухолевых клеток к химиопрепаратам по профилю экспрессии генов",
        "warn": "Исследовательский прототип — не для принятия клинических решений",
        "tab_pred": "Прогноз",
        "tab_all": "Все препараты",
        "tab_q": "Качество модели",
        "tab_about": "О проекте",
        "drug": "Выберите препарат:",
        "line": "Выберите клеточную линию (не участвовала в обучении):",
        "metrics": "Метрики модели: F1={f1:.3f}, ROC-AUC={auc:.3f}, порог классификации={thr:.2f}",
        "sens": "чувствительна",
        "res": "устойчива",
        "pred": "Прогноз",
        "true": "Настоящий класс",
        "match": "✓ Совпадает",
        "miss": "✗ Модель ошиблась",
        "prob": "Вероятность чувствительности",
        "thr_note": "Порог классификации: {thr:.0%}",
        "near": "Результат близок к порогу: прогноз неустойчив, уверенность низкая.",
        "clear": "Результат заметно отличается от порога.",
        "genes_title": "Гены, наиболее важные для модели",
        "genes_cap": "Это важность гена для всей модели (Random Forest), а не причина ответа. В колонке «У этой линии» показано, насколько экспрессия гена в выбранной линии выше или ниже среднего по обучающим данным (z-score).",
        "gene": "Ген",
        "imp": "Важность",
        "level": "У этой линии",
        "higher": "выше среднего",
        "lower": "ниже среднего",
        "tox_title": "Известные побочные эффекты {drug}",
        "tox_cap": "Справочная информация из инструкций по применению (FDA), не зависит от введённого генетического профиля",
        "all_title": "Одна линия, пять препаратов",
        "all_line": "Клеточная линия:",
        "all_cap": "Столбец — вероятность чувствительности, чёрная метка — порог этого препарата. Выше метки: прогноз «чувствительна».",
        "all_note": "Для некоторых препаратов эта линия может отсутствовать в тестовой выборке; такие препараты не показаны.",
        "q_title": "Качество модели",
        "q_cap": "ROC-AUC на тестовых линиях с 95% доверительным интервалом (bootstrap, 1000 повторов). Красная линия — случайное угадывание (0.5), жёлтая — порог практической пользы (0.70).",
        "q_small": "Тестовых линий всего 14–15 на препарат (чувствительных линий от 2 до 8), поэтому интервалы широкие. Эти оценки показывают тенденцию, но не окончательное качество.",
        "q_drug": "Препарат",
        "q_n": "Линий",
        "q_pos": "Чувств.",
        "q_ci": "95% ДИ",
        "q_roc": "ROC-кривая выбранного препарата",
        "up_title": "Загрузить свой профиль экспрессии (экспериментально)",
        "up_help": "CSV: либо одна строка с названиями генов в заголовке, либо два столбца «gene, value». Значения должны быть в том же масштабе, что и данные GDSC.",
        "up_cov": "Покрытие генов модели: {c:.0%}",
        "up_low": "Покрытие меньше {m:.0%}: прогноз ненадёжен и не показан.",
        "up_fill": "Недостающие гены заменены средним по обучающим данным.",
        "up_err": "Не удалось прочитать файл: {e}",
        "up_res": "Результат по загруженному профилю",
        "col_drug": "Препарат",
        "col_prob": "Вероятность",
        "col_pred": "Прогноз",
        "col_cov": "Покрытие",
        "about": """
**О проекте.** Прототип показывает, как модель машинного обучения (Random Forest) по профилю экспрессии генов оценивает чувствительность клеточных линий рака лёгкого к пяти препаратам.

**Данные.** GDSC (Genomics of Drug Sensitivity in Cancer). Информация о побочных эффектах взята из инструкций по применению (FDA).

**Что это не так.** Это не диагностика и не рекомендация по лечению. Модель обучена на клеточных линиях, а не на пациентах, и проверена на небольшой тестовой выборке.

**Чем полезно.** Работает в браузере без установки, на казахском и русском; показывает вероятность и уровень уверенности, гены с пояснением, все 5 препаратов сразу и справку по побочным эффектам.

**Авторы.** Оразақын Айзере, Оразбек Адия. Руководитель: Айдаркулова Р.С. NIS Алматы, проект №62.
""",
        "footer": "Данные: GDSC (Genomics of Drug Sensitivity in Cancer). Побочные эффекты: инструкции по применению (FDA).",
    },
    "kk": {
        "title": "Фармакотөзімділік AI-детекторы",
        "caption": "Ісік жасушаларының химиопрепараттарға сезімталдығын гендер экспрессиясы профилі бойынша болжау",
        "warn": "Зерттеу прототипі — клиникалық шешім қабылдауға арналмаған",
        "tab_pred": "Болжам",
        "tab_all": "Барлық препараттар",
        "tab_q": "Модель сапасы",
        "tab_about": "Жоба туралы",
        "drug": "Препаратты таңдаңыз:",
        "line": "Жасушалық желіні таңдаңыз (оқытуға қатыспаған):",
        "metrics": "Модель метрикалары: F1={f1:.3f}, ROC-AUC={auc:.3f}, классификация шегі={thr:.2f}",
        "sens": "сезімтал",
        "res": "төзімді",
        "pred": "Болжам",
        "true": "Нақты класс",
        "match": "✓ Сәйкес келеді",
        "miss": "✗ Модель қателесті",
        "prob": "Сезімталдық ықтималдығы",
        "thr_note": "Классификация шегі: {thr:.0%}",
        "near": "Нәтиже шекке жақын: болжам тұрақсыз, сенімділік төмен.",
        "clear": "Нәтиже шектен анық ерекшеленеді.",
        "genes_title": "Модель үшін ең маңызды гендер",
        "genes_cap": "Бұл геннің бүкіл модель (Random Forest) үшін маңыздылығы, жауаптың себебі емес. «Осы желіде» бағанында таңдалған желідегі ген экспрессиясы оқыту деректерінің орташасынан жоғары немесе төмен екені көрсетілген (z-score).",
        "gene": "Ген",
        "imp": "Маңыздылығы",
        "level": "Осы желіде",
        "higher": "орташадан жоғары",
        "lower": "орташадан төмен",
        "tox_title": "{drug} препаратының белгілі жанама әсерлері",
        "tox_cap": "Қолдану нұсқаулықтарынан (FDA) алынған анықтамалық ақпарат, енгізілген генетикалық профильге тәуелсіз",
        "all_title": "Бір желі, бес препарат",
        "all_line": "Жасушалық желі:",
        "all_cap": "Баған — сезімталдық ықтималдығы, қара белгі — осы препараттың шегі. Белгіден жоғары: болжам «сезімтал».",
        "all_note": "Кейбір препарат үшін бұл желі тест таңдамасында болмауы мүмкін; ондай препараттар көрсетілмейді.",
        "q_title": "Модель сапасы",
        "q_cap": "Тест желілеріндегі ROC-AUC және 95% сенім аралығы (bootstrap, 1000 қайталау). Қызыл сызық — кездейсоқ болжам (0.5), сары сызық — тәжірибелік пайда шегі (0.70).",
        "q_small": "Әр препарат бойынша тест желілері небәрі 14–15 (сезімтал желі 2-ден 8-ге дейін), сондықтан аралықтар кең. Бұл бағалар үрдісті көрсетеді, түпкілікті сапаны емес.",
        "q_drug": "Препарат",
        "q_n": "Желілер",
        "q_pos": "Сезімтал",
        "q_ci": "95% СА",
        "q_roc": "Таңдалған препараттың ROC-қисығы",
        "up_title": "Өз экспрессия профиліңізді жүктеу (эксперименттік)",
        "up_help": "CSV: не гендер атаулары тақырыпта болатын бір жол, не «gene, value» екі баған. Мәндер GDSC деректерімен бірдей масштабта болуы керек.",
        "up_cov": "Модель гендерін қамту: {c:.0%}",
        "up_low": "Қамту {m:.0%}-дан аз: болжам сенімсіз, сондықтан көрсетілмейді.",
        "up_fill": "Жетіспейтін гендер оқыту деректерінің орташасымен толтырылды.",
        "up_err": "Файлды оқу мүмкін болмады: {e}",
        "up_res": "Жүктелген профиль бойынша нәтиже",
        "col_drug": "Препарат",
        "col_prob": "Ықтималдық",
        "col_pred": "Болжам",
        "col_cov": "Қамту",
        "about": """
**Жоба туралы.** Прототип машиналық оқыту моделі (Random Forest) гендер экспрессиясы профилі бойынша өкпе обыры жасушалық желілерінің бес препаратқа сезімталдығын қалай бағалайтынын көрсетеді.

**Деректер.** GDSC (Genomics of Drug Sensitivity in Cancer). Жанама әсерлер туралы ақпарат қолдану нұсқаулықтарынан (FDA) алынды.

**Бұл не емес.** Бұл диагностика да, емдеу ұсынысы да емес. Модель пациенттерде емес, жасушалық желілерде оқытылған және шағын тест таңдамасында тексерілген.

**Артықшылығы.** Браузерде орнатусыз жұмыс істейді, қазақ және орыс тілінде; ықтималдықты және сенімділік деңгейін, түсіндірмесі бар гендерді, барлық 5 препаратты бірден және жанама әсерлер анықтамасын көрсетеді.

**Авторлар.** Оразақын Айзере, Оразбек Адия. Жетекші: Айдаркулова Р.С. NIS Алматы, №62 жоба.
""",
        "footer": "Деректер: GDSC (Genomics of Drug Sensitivity in Cancer). Жанама әсерлер: қолдану нұсқаулықтары (FDA).",
    },
}

ORGAN = {
    "ЖКТ": {"ru": "ЖКТ", "kk": "АІЖ"},
    "Кожа": {"ru": "Кожа", "kk": "Тері"},
    "Слизистые": {"ru": "Слизистые", "kk": "Кілегей қабықтар"},
    "Печень": {"ru": "Печень", "kk": "Бауыр"},
    "Лёгкие": {"ru": "Лёгкие", "kk": "Өкпе"},
    "Костный мозг": {"ru": "Костный мозг", "kk": "Сүйек кемігі"},
    "Почки": {"ru": "Почки", "kk": "Бүйрек"},
    "Периферические нервы": {"ru": "Периферические нервы", "kk": "Шеткі жүйкелер"},
}

# organ -> (risk, ru note, kk note)
TOX = {
    "Afatinib": {
        "ЖКТ": ("high", "Диарея — наиболее частый дозолимитирующий эффект", "Диарея — дозаны шектейтін ең жиі әсер"),
        "Кожа": ("high", "Акнеформная сыпь, поражение ногтевых валиков", "Безеуге ұқсас бөртпе, тырнақ қыртыстарының зақымдалуы"),
        "Слизистые": ("medium", "Стоматит, мукозит", "Стоматит, мукозит"),
        "Печень": ("low", "Редкое повышение печёночных ферментов", "Бауыр ферменттерінің сирек жоғарылауы"),
    },
    "Gefitinib": {
        "Лёгкие": ("high", "Интерстициальная болезнь лёгких — редкая, но потенциально фатальная (частота ~1-6%)", "Интерстициалды өкпе ауруы — сирек, бірақ өліммен аяқталуы мүмкін (жиілігі ~1-6%)"),
        "Печень": ("medium", "Повышение печёночных трансаминаз, описаны случаи гепатотоксичности", "Бауыр трансаминазаларының жоғарылауы, гепатотоксикалық жағдайлар сипатталған"),
        "ЖКТ": ("medium", "Диарея (~49% пациентов)", "Диарея (~49% пациент)"),
        "Кожа": ("medium", "Акнеформная сыпь (~44% пациентов)", "Безеуге ұқсас бөртпе (~44% пациент)"),
    },
    "Pemetrexed": {
        "Костный мозг": ("high", "Миелосупрессия — основная дозолимитирующая токсичность", "Миелосупрессия — дозаны шектейтін негізгі уыттылық"),
        "Почки": ("medium", "Описаны случаи нефротоксичности вплоть до необратимого повреждения почек", "Нефротоксичность жағдайлары, бүйректің қайтымсыз зақымдалуына дейін, сипатталған"),
        "Печень": ("medium", "Частое повышение печёночных ферментов", "Бауыр ферменттерінің жиі жоғарылауы"),
        "Слизистые": ("medium", "Мукозит, кожные реакции", "Мукозит, тері реакциялары"),
    },
    "Olaparib": {
        "Костный мозг": ("high", "Анемия, миелосупрессия — частые дозолимитирующие эффекты", "Анемия, миелосупрессия — дозаны шектейтін жиі әсерлер"),
        "ЖКТ": ("medium", "Тошнота, рвота, диарея", "Жүрек айну, құсу, диарея"),
        "Печень": ("low", "Редкое повышение печёночных ферментов", "Бауыр ферменттерінің сирек жоғарылауы"),
    },
    "Vinorelbine": {
        "Костный мозг": ("high", "Нейтропения — основная дозолимитирующая токсичность", "Нейтропения — дозаны шектейтін негізгі уыттылық"),
        "Периферические нервы": ("medium", "Периферическая нейропатия, сниженные рефлексы", "Шеткі нейропатия, рефлекстердің төмендеуі"),
        "ЖКТ": ("medium", "Запор, тошнота, рвота, стоматит", "Іш қату, жүрек айну, құсу, стоматит"),
        "Лёгкие": ("low", "Редкие случаи одышки и интерстициальных изменений", "Ентігу мен интерстициалды өзгерістердің сирек жағдайлары"),
    },
}

RISK_ICON = {"high": "🔴", "medium": "🟡", "low": "🟢"}
RISK_COLOR = {"high": "#E24B4A", "medium": "#F0B400", "low": "#1D9E75"}
ORGAN_POS = {
    "Лёгкие": (135, 95),
    "Печень": (172, 132),
    "Почки": (165, 160),
    "ЖКТ": (140, 165),
    "Кожа": (75, 130),
    "Костный мозг": (125, 240),
    "Периферические нервы": (170, 240),
    "Слизистые": (150, 62),
}


def body_svg(tox, lang):
    circles = ""
    for organ, (risk, ru, kk) in tox.items():
        if organ not in ORGAN_POS:
            continue
        x, y = ORGAN_POS[organ]
        note = ru if lang == "ru" else kk
        circles += (
            f'<circle cx="{x}" cy="{y}" r="9" fill="{RISK_COLOR[risk]}" stroke="white" stroke-width="2">'
            f"<title>{ORGAN[organ][lang]}: {note}</title></circle>"
        )
    return f"""<svg width="260" height="320" viewBox="0 0 300 320" xmlns="http://www.w3.org/2000/svg">
        <circle cx="150" cy="35" r="24" fill="#E8E8EA" stroke="#B0B0B5" stroke-width="1.5"/>
        <rect x="138" y="56" width="24" height="12" fill="#E8E8EA" stroke="#B0B0B5" stroke-width="1.5"/>
        <path d="M 108 68 Q 150 60 192 68 L 200 100 Q 205 150 195 195 Q 150 210 105 195 Q 95 150 100 100 Z" fill="#E8E8EA" stroke="#B0B0B5" stroke-width="1.5"/>
        <path d="M 100 75 Q 75 85 68 140 Q 66 165 72 185 L 88 182 Q 84 155 88 135 Q 92 105 105 90 Z" fill="#E8E8EA" stroke="#B0B0B5" stroke-width="1.5"/>
        <path d="M 200 75 Q 225 85 232 140 Q 234 165 228 185 L 212 182 Q 216 155 212 135 Q 208 105 195 90 Z" fill="#E8E8EA" stroke="#B0B0B5" stroke-width="1.5"/>
        <path d="M 118 195 Q 112 240 108 290 Q 108 300 120 300 Q 128 300 128 290 Q 132 245 138 200 Z" fill="#E8E8EA" stroke="#B0B0B5" stroke-width="1.5"/>
        <path d="M 182 195 Q 188 240 192 290 Q 192 300 180 300 Q 172 300 172 290 Q 168 245 162 200 Z" fill="#E8E8EA" stroke="#B0B0B5" stroke-width="1.5"/>
        {circles}
    </svg>"""


# ----------------------------------------------------------------------------
# Model helpers
# ----------------------------------------------------------------------------
def thr_of(d):
    return float(d.get("threshold", 0.5))


def s_index(d):
    return list(d["label_encoder"].classes_).index("S")


def predict_prob(d, row_values):
    xs = d["scaler"].transform(np.asarray(row_values, dtype=float).reshape(1, -1))
    return float(d["model"].predict_proba(xs)[0][s_index(d)]), xs[0]


def is_sens(label):
    return str(label).lower().startswith("s")


@st.cache_data
def test_scores(drug):
    d = artifacts[drug]
    xs = d["scaler"].transform(d["demo_profiles"].values)
    p = d["model"].predict_proba(xs)[:, s_index(d)]
    y = np.array([1 if is_sens(t) else 0 for t in d["demo_true_labels"]])
    return y, p


@st.cache_data
def auc_with_ci(drug, n_boot=1000, seed=42):
    y, p = test_scores(drug)
    auc = roc_auc_score(y, p)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        vals.append(roc_auc_score(y[idx], p[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(auc), float(lo), float(hi), int(len(y)), int(y.sum())


def read_uploaded(file):
    df = pd.read_csv(file)
    if df.shape[1] == 2 and df.shape[0] > 1:
        s = pd.Series(df.iloc[:, 1].values, index=df.iloc[:, 0].astype(str))
    else:
        s = df.iloc[0]
        s.index = s.index.astype(str)
    s = pd.to_numeric(s, errors="coerce").dropna()
    return s[~s.index.duplicated()]


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
lang_label = st.radio("", ["Қазақша", "Русский"], horizontal=True, label_visibility="collapsed")
L = "kk" if lang_label == "Қазақша" else "ru"
T = TXT[L]

st.title(T["title"])
st.caption(T["caption"])
st.warning(T["warn"])

tab_pred, tab_all, tab_q, tab_about = st.tabs([T["tab_pred"], T["tab_all"], T["tab_q"], T["tab_about"]])

drugs = list(artifacts.keys())

# ---- Tab 1: prediction for one drug ----------------------------------------
with tab_pred:
    drug = st.selectbox(T["drug"], drugs, key="drug")
    d = artifacts[drug]
    thr = thr_of(d)
    st.caption(T["metrics"].format(f1=d["metrics"]["f1"], auc=float(d["metrics"]["roc_auc"]), thr=thr))

    name = st.selectbox(T["line"], sorted(d["demo_names"]), key="line")
    idx = list(d["demo_names"]).index(name)
    profile = d["demo_profiles"].iloc[idx]
    true_label = d["demo_true_labels"][idx]
    prob, z = predict_prob(d, profile.values)
    pred_sens = prob >= thr

    c1, c2 = st.columns(2)
    with c1:
        txt = f"{T['pred']}: **{(T['sens'] if pred_sens else T['res']).upper()}**"
        (st.success if pred_sens else st.error)(txt)
        st.metric(T["prob"], f"{prob:.0%}")
        st.progress(min(max(prob, 0.0), 1.0))
        st.caption(T["thr_note"].format(thr=thr))
        if abs(prob - thr) < NEAR_MARGIN:
            st.warning(T["near"])
        else:
            st.caption(T["clear"])
    with c2:
        st.info(f"{T['true']}: **{T['sens'] if is_sens(true_label) else T['res']}**")
        st.write(T["match"] if pred_sens == is_sens(true_label) else T["miss"])

    st.subheader(T["genes_title"])
    st.caption(T["genes_cap"])
    feats = list(d["scaler"].feature_names_in_)
    top5 = d["top_10_genes"].head(5)
    rows = []
    for g, imp in top5.items():
        zi = float(z[feats.index(g)]) if g in feats else float("nan")
        rows.append(
            {
                T["gene"]: g,
                T["imp"]: round(float(imp), 4),
                T["level"]: f"{T['higher'] if zi > 0 else T['lower']} (z={zi:+.1f})",
                "NCBI": "https://www.ncbi.nlm.nih.gov/gene/?term=" + urllib.parse.quote(f"{g}[Sym] AND human[Organism]"),
            }
        )
    gdf = pd.DataFrame(rows)
    chart = (
        alt.Chart(gdf)
        .mark_bar(color="#2B66C3")
        .encode(
            x=alt.X(f"{T['gene']}:N", sort=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y(f"{T['imp']}:Q"),
            tooltip=[T["gene"], T["imp"], T["level"]],
        )
        .properties(height=240)
    )
    st.altair_chart(chart, use_container_width=True)
    st.dataframe(
        gdf,
        hide_index=True,
        use_container_width=True,
        column_config={"NCBI": st.column_config.LinkColumn("NCBI", display_text="NCBI")},
    )

    st.divider()
    st.subheader(T["tox_title"].format(drug=drug))
    st.caption(T["tox_cap"])
    tox = TOX.get(drug, {})
    ca, cb = st.columns(2)
    with ca:
        st.markdown(body_svg(tox, L), unsafe_allow_html=True)
    with cb:
        st.write("")
        for organ, (risk, ru, kk) in tox.items():
            st.write(f"{RISK_ICON[risk]} **{ORGAN[organ][L]}** — {ru if L == 'ru' else kk}")

    st.divider()
    with st.expander(T["up_title"]):
        st.caption(T["up_help"])
        up = st.file_uploader("CSV", type=["csv"], key="upload")
        if up is not None:
            try:
                sample = read_uploaded(up)
                out = []
                for dn in drugs:
                    dd = artifacts[dn]
                    f = list(dd["scaler"].feature_names_in_)
                    cov = float(np.mean([g in sample.index for g in f]))
                    row = {T["col_drug"]: dn, T["col_cov"]: f"{cov:.0%}"}
                    if cov >= MIN_COVERAGE:
                        vals = np.array([sample[g] if g in sample.index else m for g, m in zip(f, dd["scaler"].mean_)], dtype=float)
                        p, _ = predict_prob(dd, vals)
                        row[T["col_prob"]] = f"{p:.0%}"
                        row[T["col_pred"]] = T["sens"] if p >= thr_of(dd) else T["res"]
                    else:
                        row[T["col_prob"]] = "—"
                        row[T["col_pred"]] = "—"
                    out.append(row)
                st.subheader(T["up_res"])
                st.dataframe(pd.DataFrame(out), hide_index=True, use_container_width=True)
                st.caption(T["up_fill"] + " " + T["up_low"].format(m=MIN_COVERAGE))
            except Exception as e:  # noqa: BLE001
                st.error(T["up_err"].format(e=e))

# ---- Tab 2: one line, all drugs ---------------------------------------------
with tab_all:
    st.subheader(T["all_title"])
    all_names = sorted({n for dn in drugs for n in artifacts[dn]["demo_names"]})
    pick = st.selectbox(T["all_line"], all_names, key="all_line")
    rows = []
    for dn in drugs:
        dd = artifacts[dn]
        if pick in list(dd["demo_names"]):
            i = list(dd["demo_names"]).index(pick)
            p, _ = predict_prob(dd, dd["demo_profiles"].iloc[i].values)
            rows.append({"drug": dn, "prob": p, "thr": thr_of(dd)})
    if rows:
        adf = pd.DataFrame(rows)
        base = alt.Chart(adf).encode(x=alt.X("drug:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)))
        bars = base.mark_bar(color="#2B66C3").encode(y=alt.Y("prob:Q", scale=alt.Scale(domain=[0, 1]), title=T["prob"]))
        ticks = base.mark_tick(color="black", thickness=3, size=40).encode(y="thr:Q")
        st.altair_chart((bars + ticks).properties(height=300), use_container_width=True)
        st.caption(T["all_cap"])
    st.caption(T["all_note"])

# ---- Tab 3: model quality ----------------------------------------------------
with tab_q:
    st.subheader(T["q_title"])
    st.caption(T["q_cap"])
    qrows = []
    for dn in drugs:
        auc, lo, hi, n, pos = auc_with_ci(dn)
        qrows.append({"drug": dn, "auc": auc, "lo": lo, "hi": hi, "n": n, "pos": pos})
    qdf = pd.DataFrame(qrows)
    pts = alt.Chart(qdf).mark_point(filled=True, size=140, color="#12274D").encode(
        y=alt.Y("drug:N", sort=None, title=None),
        x=alt.X("auc:Q", scale=alt.Scale(domain=[0, 1]), title="ROC-AUC"),
        tooltip=["drug", "auc", "lo", "hi", "n", "pos"],
    )
    err = alt.Chart(qdf).mark_rule(color="#12274D", strokeWidth=3).encode(
        y=alt.Y("drug:N", sort=None), x="lo:Q", x2="hi:Q"
    )
    rule05 = alt.Chart(pd.DataFrame({"x": [0.5]})).mark_rule(color="#C23B4A", strokeWidth=2).encode(x="x:Q")
    rule07 = alt.Chart(pd.DataFrame({"x": [0.7]})).mark_rule(color="#E0A100", strokeWidth=2).encode(x="x:Q")
    st.altair_chart((err + pts + rule05 + rule07).properties(height=280), use_container_width=True)
    st.dataframe(
        pd.DataFrame(
            {
                T["q_drug"]: qdf["drug"],
                "ROC-AUC": qdf["auc"].round(3),
                T["q_ci"]: [f"{a:.2f}–{b:.2f}" for a, b in zip(qdf["lo"], qdf["hi"])],
                T["q_n"]: qdf["n"],
                T["q_pos"]: qdf["pos"],
            }
        ),
        hide_index=True,
        use_container_width=True,
    )
    st.info(T["q_small"])

    rd = st.selectbox(T["drug"], drugs, key="roc_drug")
    y, p = test_scores(rd)
    fpr, tpr, _ = roc_curve(y, p)
    rdf = pd.DataFrame({"fpr": fpr, "tpr": tpr})
    diag = pd.DataFrame({"fpr": [0, 1], "tpr": [0, 1]})
    st.caption(T["q_roc"])
    roc = alt.Chart(rdf).mark_line(color="#2B66C3", strokeWidth=3, interpolate="step-after").encode(
        x=alt.X("fpr:Q", title="1 − specificity", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("tpr:Q", title="sensitivity", scale=alt.Scale(domain=[0, 1])),
    )
    dline = alt.Chart(diag).mark_line(color="#C23B4A", strokeDash=[6, 4]).encode(x="fpr:Q", y="tpr:Q")
    st.altair_chart((roc + dline).properties(height=320), use_container_width=True)

# ---- Tab 4: about -------------------------------------------------------------
with tab_about:
    st.markdown(T["about"])

st.divider()
st.caption(T["footer"])
