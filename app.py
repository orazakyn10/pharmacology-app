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
}

RISK_COLORS = {"high": "🔴", "medium": "🟡", "low": "🟢"}

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
    for organ, info in TOXICITY_DATA.get(drug_name, {}).items():
        st.write(f"{RISK_COLORS[info['risk']]} **{organ}** — {info['note']}")

st.divider()
st.caption("Данные: GDSC (Genomics of Drug Sensitivity in Cancer). Информация о побочных эффектах — из инструкций по применению препаратов (FDA prescribing information).")
