import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="AI-детектор фармакорезистентности", page_icon="🧬", layout="centered")

artifacts = joblib.load("multi_drug_artifacts.pkl")

TOXICITY_DATA = {
    "Erlotinib": {
        "Кожа": {"risk": "high", "note": "Сыпь у 75% пациентов (данные исследования BR.21)"},
        "ЖКТ": {"risk": "high", "note": "Диарея — дозолимитирующая токсичность"},
        "Печень": {"risk": "medium", "note": "Редкие случаи гепатотоксичности"},
        "Лёгкие": {"risk": "medium", "note": "Редкие случаи интерстициального заболевания лёгких"},
        "Почки": {"risk": "low", "note": "Редкая почечная недостаточность, чаще на фоне обезвоживания"},
    },
    "Doxorubicin": {
        "Сердце": {"risk": "high", "note": "Кардиомиопатия — основная дозолимитирующая токсичность"},
        "Костный мозг": {"risk": "high", "note": "Миелосупрессия"},
        "ЖКТ": {"risk": "medium", "note": "Тошнота, рвота, мукозит"},
        "Кожа": {"risk": "medium", "note": "Алопеция; при экстравазации — некроз тканей"},
        "Печень": {"risk": "low", "note": "Усиление токсичности при сопутствующей лучевой терапии"},
    },
    "Gemcitabine": {
        "Костный мозг": {"risk": "high", "note": "Миелосупрессия — дозолимитирующая токсичность"},
        "Лёгкие": {"risk": "medium", "note": "Требует немедленной отмены при тяжёлом течении"},
        "Печень": {"risk": "medium", "note": "Редкие случаи серьёзной гепатотоксичности"},
        "Почки": {"risk": "medium", "note": "Редкие случаи гемолитико-уремического синдрома"},
        "ЖКТ": {"risk": "medium", "note": "Тошнота и рвота — частые эффекты"},
    },
    "Cisplatin": {
        "Почки": {"risk": "high", "note": "Нефротоксичность — основная дозолимитирующая токсичность"},
        "Периферические нервы": {"risk": "high", "note": "Периферическая нейропатия, часто необратимая"},
        "ЖКТ": {"risk": "high", "note": "Выраженная тошнота и рвота"},
        "Костный мозг": {"risk": "medium", "note": "Умеренная миелосупрессия"},
        "Лёгкие": {"risk": "low", "note": "Редкие случаи лёгочной токсичности"},
    },
    "Paclitaxel": {
        "Периферические нервы": {"risk": "high", "note": "Периферическая нейропатия — частый дозолимитирующий эффект"},
        "Костный мозг": {"risk": "high", "note": "Нейтропения — дозолимитирующая токсичность"},
        "Сердце": {"risk": "medium", "note": "Редкие нарушения сердечного ритма"},
        "Кожа": {"risk": "medium", "note": "Алопеция, реакции гиперчувствительности"},
        "ЖКТ": {"risk": "low", "note": "Умеренная тошнота"},
    },
}

RISK_COLORS = {"high": "🔴", "medium": "🟡", "low": "🟢"}
COLOR_MAP = {"high": "#E24B4A", "medium": "#F0B400", "low": "#1D9E75"}

ORGAN_POSITIONS = {
    "Мозг и ЦНС": (150, 35),
    "Лёгкие": (135, 95),
    "Сердце": (140, 118),
    "Печень": (172, 132),
    "Почки": (165, 160),
    "ЖКТ": (140, 165),
    "Кожа": (75, 130),
    "Костный мозг": (125, 240),
    "Периферические нервы": (170, 240),
}

def make_body_svg(drug_toxicity):
    circles = ""
    for organ, info in drug_toxicity.items():
        if organ not in ORGAN_POSITIONS:
            continue
        x, y = ORGAN_POSITIONS[organ]
        color = COLOR_MAP[info["risk"]]
        circles += f'<circle cx="{x}" cy="{y}" r="9" fill="{color}" stroke="white" stroke-width="2"><title>{organ}: {info["note"]}</title></circle>'

    return f'''<svg width="260" height="320" viewBox="0 0 300 320" xmlns="http://www.w3.org/2000/svg">
        <circle cx="150" cy="35" r="24" fill="#E8E8EA" stroke="#B0B0B5" stroke-width="1.5"/>
        <rect x="138" y="56" width="24" height="12" fill="#E8E8EA" stroke="#B0B0B5" stroke-width="1.5"/>
        <path d="M 108 68 Q 150 60 192 68 L 200 100 Q 205 150 195 195 Q 150 210 105 195 Q 95 150 100 100 Z" fill="#E8E8EA" stroke="#B0B0B5" stroke-width="1.5"/>
        <path d="M 100 75 Q 75 85 68 140 Q 66 165 72 185 L 88 182 Q 84 155 88 135 Q 92 105 105 90 Z" fill="#E8E8EA" stroke="#B0B0B5" stroke-width="1.5"/>
        <path d="M 200 75 Q 225 85 232 140 Q 234 165 228 185 L 212 182 Q 216 155 212 135 Q 208 105 195 90 Z" fill="#E8E8EA" stroke="#B0B0B5" stroke-width="1.5"/>
        <path d="M 118 195 Q 112 240 108 290 Q 108 300 120 300 Q 128 300 128 290 Q 132 245 138 200 Z" fill="#E8E8EA" stroke="#B0B0B5" stroke-width="1.5"/>
        <path d="M 182 195 Q 188 240 192 290 Q 192 300 180 300 Q 172 300 172 290 Q 168 245 162 200 Z" fill="#E8E8EA" stroke="#B0B0B5" stroke-width="1.5"/>
        {circles}
    </svg>'''

st.title("AI-детектор фармакорезистентности")
st.caption("Прогноз чувствительности опухолевых клеток к химиопрепаратам по профилю экспрессии генов")
st.warning("Research prototype — not for clinical decision-making")

drug_name = st.selectbox("Выберите препарат:", list(artifacts.keys()))

drug_data = artifacts[drug_name]
model = drug_data["model"]
scaler = drug_data["scaler"]
le = drug_data["label_encoder"]
top_10_genes = drug_data["top_10_genes"]
threshold = drug_data.get("threshold", 0.5)

st.caption(f"Метрики модели: F1={drug_data['metrics']['f1']:.3f}, ROC-AUC={drug_data['metrics']['roc_auc']:.3f}")

name = st.selectbox("Выберите клеточную линию для проверки:", sorted(drug_data["demo_names"]))

if st.button("Предсказать чувствительность", type="primary"):
    idx = drug_data["demo_names"].index(name)
    profile = drug_data["demo_profiles"].iloc[idx]
    true_label = drug_data["demo_true_labels"][idx]

    profile_scaled = scaler.transform(profile.values.reshape(1, -1))
    pred_proba = model.predict_proba(profile_scaled)[0]
    sensitive_idx = list(le.classes_).index("S")
    prob_sensitive = pred_proba[sensitive_idx]

    prediction = "sensitive" if prob_sensitive >= threshold else "resistant"

    col1, col2 = st.columns(2)
    with col1:
        if prediction == "sensitive":
            st.success(f"Прогноз: **{prediction.upper()}**")
        else:
            st.error(f"Прогноз: **{prediction.upper()}**")
        st.metric("Вероятность sensitive", f"{prob_sensitive:.0%}")
    with col2:
        st.info(f"Настоящий класс: **{true_label}**")
        st.write("✓ Совпадает" if prediction == true_label else "✗ Модель ошиблась")

    st.subheader("Топ-5 генов, повлиявших на прогноз")
    st.bar_chart(top_10_genes.head(5))

    st.divider()
    st.subheader(f"Известные побочные эффекты {drug_name}")
    st.caption("Справочная информация из инструкций по применению (FDA), не зависит от введённого генетического профиля")

    col_svg, col_list = st.columns([1, 1])
    with col_svg:
        st.markdown(make_body_svg(TOXICITY_DATA.get(drug_name, {})), unsafe_allow_html=True)
    with col_list:
        st.write("")
        for organ, info in TOXICITY_DATA.get(drug_name, {}).items():
            st.write(f"{RISK_COLORS[info['risk']]} **{organ}** — {info['note']}")

st.divider()
st.caption("Данные: GDSC (Genomics of Drug Sensitivity in Cancer). Информация о побочных эффектах — из инструкций по применению препаратов (FDA prescribing information).")
