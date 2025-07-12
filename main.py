import streamlit as st
import pandas as pd
import numpy as np

from config import SECTOR_LIST, INDIVIDUAL_CATEGORIES, PRODUCT_TYPES, PRODUCT_CATEGORIES, PROMOTIONS
from data_generator import generate_customers, generate_transactions
from features import aggregate_transactions
from model_train import train_model
from viz import (
    plot_twin_distribution, plot_segment_heatmap, plot_sector_heatmap,
    plot_pie_twin_response, plot_segment_interest_heatmap
)
from chatbot import parse_with_mistral

st.set_page_config(page_title="Digital Twin & AI Customer Simulation Demo", layout="wide")
st.title("Enterprise-scale Digital Twin & AI Segmentation Demo")

# ----------- Normalization Functions -----------

def normalize_list_value(val_list, valid_list):
    if isinstance(val_list, str):
        val_list = [val_list]
    if not isinstance(val_list, list):
        val_list = []
    normalized = []
    valid_set = {v.lower(): v for v in valid_list}
    for p in val_list:
        if not isinstance(p, str):
            continue
        p_clean = p.strip().lower()
        if p_clean in valid_set:
            normalized.append(valid_set[p_clean])
        else:
            # Fuzzy match for common LLM errors
            for v in valid_list:
                if p_clean == v.strip().lower():
                    normalized.append(v)
                    break
    return normalized

def normalize_single_value(val, valid_list, default_val):
    if not isinstance(val, str):
        return default_val
    val_clean = val.strip().lower()
    valid_map = {v.lower(): v for v in valid_list}
    if val_clean in valid_map:
        return valid_map[val_clean]
    for v in valid_list:
        if val_clean == v.lower():
            return v
    return default_val

def normalize_channel(val):
    val = (val or "").strip().lower()
    if val in ["digital", "digital only"]:
        return "Digital"
    if val in ["branch", "branch only"]:
        return "Branch"
    if val in ["digital and branch", "both", "both channels"]:
        return "Both"
    return "Digital"

def normalize_int(val, default):
    try:
        v = int(val)
        if 1 <= v <= 60:
            return v
        else:
            return default
    except:
        return default

def normalize_year(val):
    valid_years = [2020, 2021, 2022, 2023, 2024]
    try:
        y = int(val)
        if y in valid_years:
            return y
        else:
            return 2024
    except:
        return 2024

def normalize_product_type(val):
    # Her türlü POS vurgusunu veya karışıklığını yakala!
    val = (val or "").strip().lower()
    if "pos" in val:
        return "POS"
    # "loan" ifadesi varsa ve "pos" da varsa, yine POS
    if "loan" in val and "pos" in val:
        return "POS"
    for option in PRODUCT_TYPES:
        if val == option.lower():
            return option
    return PRODUCT_TYPES[0]  # default

# ----------- SESSION STATE INIT -----------

if 'parsed' not in st.session_state:
    st.session_state['parsed'] = None
if 'run_simulation' not in st.session_state:
    st.session_state['run_simulation'] = False
if 'filters' not in st.session_state:
    st.session_state['filters'] = {}

# ----------- CHATBOT INPUT -----------

st.header("💬 Digital Twin Chatbot Assistant")

user_input = st.text_area(
    "Describe your product/campaign below. The AI assistant will extract relevant filters and you can further edit them.",
    placeholder="Example: 18-month fixed interest POS loan for textile SMEs with cashback and loyalty program.",
    height=100
)

if st.button("Ask Bot"):
    if user_input.strip() == "":
        st.warning("Please enter a product/campaign description.")
        st.stop()
    parsed = parse_with_mistral(user_input)
    st.session_state['parsed'] = parsed
    st.session_state['run_simulation'] = False
    st.session_state['filters'] = {}

# ----------- FILTERS SECTION -----------

if st.session_state['parsed'] is not None:
    parsed = st.session_state['parsed']

    segments_raw = st.session_state['filters'].get('segment', parsed.get('segment', []))
    segments = normalize_list_value(segments_raw, ['Individual', 'SME', 'Corporate'])

    sectors_raw = st.session_state['filters'].get('sector', parsed.get('sector', []))
    sectors = normalize_list_value(sectors_raw, SECTOR_LIST)

    product_type_raw = st.session_state['filters'].get('product_type', parsed.get('product_type', PRODUCT_TYPES[0]))
    product_type = normalize_product_type(product_type_raw)

    product_category_raw = st.session_state['filters'].get('product_category', parsed.get('product_category', PRODUCT_CATEGORIES[0]))
    product_category = normalize_single_value(product_category_raw, PRODUCT_CATEGORIES, PRODUCT_CATEGORIES[0])

    promotions_raw = st.session_state['filters'].get('promotion', parsed.get('promotion', []))
    promotions = normalize_list_value(promotions_raw, PROMOTIONS)

    channel_raw = st.session_state['filters'].get('channel', parsed.get('channel', "Digital"))
    channel = normalize_channel(channel_raw)

    term_raw = st.session_state['filters'].get('term', parsed.get('term', 12))
    term = normalize_int(term_raw, 12)

    interest_type_raw = st.session_state['filters'].get('interest_type', parsed.get('interest_type', 'Fixed'))
    interest_type = normalize_single_value(interest_type_raw, ['Fixed', 'Variable', 'None'], 'Fixed')

    risk_level_raw = st.session_state['filters'].get('risk_level', parsed.get('risk_level', 'Medium'))
    risk_level = normalize_single_value(risk_level_raw, ['High', 'Medium', 'Low'], 'Medium')

    innovation_level_raw = st.session_state['filters'].get('innovation_level', parsed.get('innovation_level', 'Medium'))
    innovation_level = normalize_single_value(innovation_level_raw, ['High', 'Medium', 'Low'], 'Medium')

    launch_year_raw = st.session_state['filters'].get('launch_year', parsed.get('launch_year', 2024))
    launch_year = normalize_year(launch_year_raw)

    segment = st.multiselect("Segment", ['Individual', 'SME', 'Corporate'], default=segments)
    sector = st.multiselect("Sector", SECTOR_LIST, default=sectors)
    product_type = st.selectbox("Product Type", PRODUCT_TYPES, index=PRODUCT_TYPES.index(product_type))
    product_category = st.selectbox("Product Category", PRODUCT_CATEGORIES, index=PRODUCT_CATEGORIES.index(product_category))
    promotion = st.multiselect("Promotion", PROMOTIONS, default=promotions)
    channel = st.selectbox("Channel", ["Digital", "Branch", "Both"], index=["Digital", "Branch", "Both"].index(channel))
    term = st.slider("Term (Months)", 1, 60, value=term)
    interest_type = st.selectbox("Interest Type", ['Fixed', 'Variable', 'None'], index=['Fixed', 'Variable', 'None'].index(interest_type))
    risk_level = st.radio("Risk Level", ['High', 'Medium', 'Low'], index=['High', 'Medium', 'Low'].index(risk_level))
    innovation_level = st.radio("Innovation Level", ['High', 'Medium', 'Low'], index=['High', 'Medium', 'Low'].index(innovation_level))
    launch_year = st.selectbox("Launch Year", [2020, 2021, 2022, 2023, 2024], index=[2020, 2021, 2022, 2023, 2024].index(launch_year))

    st.session_state['filters'] = {
        'segment': segment,
        'sector': sector,
        'product_type': product_type,
        'product_category': product_category,
        'promotion': promotion,
        'channel': channel,
        'term': term,
        'interest_type': interest_type,
        'risk_level': risk_level,
        'innovation_level': innovation_level,
        'launch_year': launch_year,
    }

    st.info("Edit any filter if you want, then click **Run Simulation** below!")

    if st.button("Run Simulation"):
        st.session_state['run_simulation'] = True

# ----------- SIMULATION AND VISUALS -----------

if st.session_state['run_simulation'] and st.session_state['filters']:
    filters = st.session_state['filters']

    n_individual, n_sme, n_corporate = 650, 220, 130
    df_customers = generate_customers(n_individual, n_sme, n_corporate)
    df_transactions = generate_transactions(df_customers, months=6, max_sample=15000)
    df_main = aggregate_transactions(df_customers, df_transactions)

    def product_effect_score(row):
        score = 1
        if row['segment'] not in filters['segment']: score *= 0.8
        if row['segment'] in ['SME', 'Corporate']:
            if row['sector'] not in filters['sector']: score *= 0.85
        if row['segment'] == 'Individual' and row['category'] not in INDIVIDUAL_CATEGORIES: score *= 0.85
        if filters['channel'] == "Digital" and row['digital_openness'] < 0.4: score *= 0.7
        if "Cashback" in filters['promotion'] and row['promotion_sensitivity'] > 0.5: score *= 1.12
        if "Digital Convenience" in filters['promotion'] and row['digital_openness'] > 0.7: score *= 1.10
        if filters['innovation_level'] == 'High' and row['innovation_openness'] > 0.7: score *= 1.08
        if filters['risk_level'] == 'High': score *= 0.85
        if filters['term'] > 36: score *= 0.92
        if filters['launch_year'] == 2024: score *= 1.06
        if row['tx_category_count'] > 8: score *= 1.04
        return score

    def twin_response_func(x):
        if x > 0.78: return 'apply/purchase'
        elif x > 0.55: return 'high interest'
        elif x > 0.35: return 'medium interest'
        elif x > 0.18: return 'neutral'
        else: return 'negative response'

    proba = train_model(df_main, "RandomForest")
    df_main = df_main.copy()
    df_main['product_score'] = df_main.apply(product_effect_score, axis=1)
    df_main['product_interest_probability'] = proba * df_main['product_score']
    df_main['twin_response'] = df_main['product_interest_probability'].apply(twin_response_func)

    st.subheader("First 100 Customers' Simulation Results")
    st.dataframe(df_main[['customer_id', 'segment', 'sector', 'category', 'avg_amount', 'tx_category_count', 'twin_response', 'product_interest_probability']].head(100))

    st.subheader("Twin Response Distribution")
    plot_twin_distribution(df_main)

    st.subheader("Twin Response (Pie Chart)")
    plot_pie_twin_response(df_main)

    st.subheader("Segment-based Twin Response Heatmap")
    plot_segment_heatmap(df_main)

    st.subheader("Sector-based Twin Response Heatmap")
    plot_sector_heatmap(df_main)

    st.subheader("Mean Product Interest Score by Segment/Response (Heatmap)")
    plot_segment_interest_heatmap(df_main)

    st.subheader("Individual Twin Analysis Panel")
    selected_customer = st.selectbox("Select a customer", df_main['customer_id'].sample(n=50, random_state=42).tolist())
    st.write(df_main[df_main['customer_id'] == selected_customer].T)
