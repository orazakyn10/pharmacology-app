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
COLOR_MAP = {"high": "#E24B4A", "medium": "#F0B400", "low": "#1D9E75"}

ORGAN_POSITIONS = {
    "Мозг и ЦНС": (150, 40),
    "Лёгкие": (150, 108),
    "Сердце": (137, 118),
    "Печень": (168, 148),
    "Почки": (150, 165),
    "ЖКТ": (150, 178),
    "Кожа": (95, 115),
    "Костный мозг": (150, 255),
}

def make_body_svg(drug_toxicity):
    circles = ""
    for organ, info in drug_toxicity.items():
        if organ not in ORGAN_POSITIONS:
            continue
        x, y = ORGAN_POSITIONS[organ]
        color = COLOR_MAP[info["risk"]]
        circles += f'''<circle cx="{x}" cy="{y}" r="9" fill="{color}" stroke="white" stroke-width="2">
            <title>{organ}: {info["note"]}</title>
        </circle>'''

    return f'''
    <svg width="260" height="320" viewBox="0 0 300 320" xmlns="http://www.w3.org/2000/svg">
        <!-- голова -->
        <circle cx="150" cy="35" r="24" fill="#E8E8EA" stroke="#B0B0B5" stroke-width="1.5"/>
        <!-- шея -->
        <rect x="138" y="56" width="24" height="12" fill="#E8E8EA" stroke="#B0B0B5" stroke-width="1.5"/>
        <!-- торс -->
        <path d="M 108 68
                 Q 150 60 192 68
                 L 200 100
                 Q 205 150 195 195
                 Q 150 210 105 195
                 Q 95 150 100 100
                 Z"
              fill="#E8E8EA" stroke="#B0B0B5" stroke-width="1.5"/>
        <!-- левая рука -->
        <path d="M 100 75 Q 75 85 68 140 Q 66 165 72 185
                 L 88 182 Q 84 155 88 135 Q 92 105 105 90 Z"
              fill="#E8E8EA" stroke="#B0B0B5" stroke-width="1.5"/>
        <!-- правая рука -->
        <path d="M 200 75 Q 225 85 232 140 Q 234 165 228 185
                 L 212 182 Q 216 155 212 135 Q 208 105 195 90 Z"
              fill="#E8E8EA" stroke="#B0B0B5" stroke-width="1.5"/>
        <!-- ноги -->
        <path d="M 118 195 Q 112 240 108 290 Q 108 300 120 300 Q 128 300 128 290
                 Q 132 245 138 200 Z"
              fill="#E8E8EA" stroke="#B0B0B5" stroke-width="1.5"/>
        <path d="M 182 195 Q 188 240 192 290 Q 192 300 180 300 Q 172 300 172 290
                 Q 168 245 162 200 Z"
              fill="#E8E8EA" stroke="#B0B0B5" stroke-width="1.5"/>
        {circles}
    </svg>
    '''
