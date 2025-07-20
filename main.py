import streamlit as st
import pandas as pd
import numpy as np

from config import SECTOR_LIST, INDIVIDUAL_CATEGORIES, PRODUCT_TYPES, PRODUCT_CATEGORIES, PROMOTIONS
from data_generator import generate_customers, generate_transactions
from features import aggregate_transactions, product_effect_score
from model_train import train_model
from viz import (
    plot_pie_twin_response, plot_twin_distribution, plot_segment_heatmap,
    plot_sector_heatmap, plot_segment_interest_heatmap
)
from chatbot import parse_with_mistral

st.set_page_config(page_title="Digital Twin & AI Customer Simulation Demo", layout="wide")
st.title("Enterprise-scale Digital Twin & AI Segmentation Demo")

st.sidebar.header("Test Mode")
test_mode = st.sidebar.radio("Choose test type:", ["Single Scenario", "A/B Test"])

CHANNELS = ["Digital", "Branch", "Digital and Branch"]
RISK_LEVELS = ['High', 'Medium', 'Low']
INNOVATION_LEVELS = ['High', 'Medium', 'Low']

def normalize_channels(ch_val):
    if isinstance(ch_val, str):
        if ch_val.lower() == "digital and branch":
            return ["Digital", "Branch"]
        return [ch_val]
    if isinstance(ch_val, list):
        out = []
        for c in ch_val:
            if c.lower() == "digital and branch":
                out += ["Digital", "Branch"]
            else:
                out.append(c)
        return sorted(list(set(out)))
    return ["Digital"]

def get_filter_defaults(parsed):
    seg = parsed.get('segment') or []
    if isinstance(seg, str):
        seg = [seg] if seg else []
    sector = parsed.get('sector') or []
    if isinstance(sector, str):
        sector = [sector] if sector else []
    promotion = parsed.get('promotion') or []
    if isinstance(promotion, str):
        promotion = [promotion] if promotion else []
    channels = normalize_channels(parsed.get('channel'))
    term = int(parsed.get('term') or 12)
    risk_level = parsed.get('risk_level', 'Medium')
    if risk_level not in RISK_LEVELS:
        risk_level = 'Medium'
    innovation_level = parsed.get('innovation_level', 'Medium')
    if innovation_level not in INNOVATION_LEVELS:
        innovation_level = 'Medium'
    launch_year = int(parsed.get('launch_year', 2024) or 2024)
    return seg, sector, promotion, channels, term, risk_level, innovation_level, launch_year

# ------ A/B TESTI ------
if test_mode == "A/B Test":
    st.header("A/B Test: Compare Two Product/Campaign Variants")
    col_a, col_b = st.columns(2)
    with col_a:
        user_input_a = st.text_area(
            "Product/Campaign A",
            height=80,
            placeholder="Example: 24-month fixed interest consumer loan for retail sector SMEs with cashback and digital convenience promotions. Digital-only channel. High innovation level. Launch year:2024."
        )
        if st.button("Parse A"):
            parsed_a = parse_with_mistral(user_input_a)
            st.session_state['parsed_a'] = parsed_a
            st.session_state['filters_a'] = None
            st.session_state['run_a'] = False
    with col_b:
        user_input_b = st.text_area(
            "Product/Campaign B",
            height=80,
            placeholder="Example: Retail sector SMEs with loyalty points and low commission promotions. Available both on digital and branch channels. Medium innovation level. Launch year:2024."
        )
        if st.button("Parse B"):
            parsed_b = parse_with_mistral(user_input_b)
            st.session_state['parsed_b'] = parsed_b
            st.session_state['filters_b'] = None
            st.session_state['run_b'] = False

    parsed_a_ok = 'parsed_a' in st.session_state and st.session_state['parsed_a'] is not None
    parsed_b_ok = 'parsed_b' in st.session_state and st.session_state['parsed_b'] is not None

    if not (parsed_a_ok and parsed_b_ok):
        st.info("Please enter product/campaign descriptions for both A and B, and click 'Parse A' & 'Parse B'.")
    else:
        st.markdown("### Edit & Confirm Filters for Both A and B")
        col_a, col_b = st.columns(2)

        # --- FILTERS FOR A ---
        with col_a:
            st.markdown("#### Filters for A")
            filters_a = st.session_state.get('filters_a')
            if filters_a is None:
                seg, sector, promotion, channels, term, risk_level, innovation_level, launch_year = get_filter_defaults(st.session_state['parsed_a'])
            else:
                seg = filters_a.get('segment', [])
                sector = filters_a.get('sector', [])
                promotion = filters_a.get('promotion', [])
                channels = filters_a.get('channel', ["Digital"])
                term = filters_a.get('term', 12)
                risk_level = filters_a.get('risk_level', 'Medium')
                innovation_level = filters_a.get('innovation_level', 'Medium')
                launch_year = filters_a.get('launch_year', 2024)
            filters_a = {}
            filters_a['segment'] = st.multiselect("Segment (A)", ['Individual', 'SME', 'Corporate'], default=seg)
            filters_a['sector'] = st.multiselect("Sector (A)", SECTOR_LIST, default=sector)
            filters_a['promotion'] = st.multiselect("Promotion (A)", PROMOTIONS, default=promotion)
            filters_a['channel'] = st.multiselect("Channel (A)", ["Digital", "Branch", "Digital and Branch"], default=channels)
            filters_a['term'] = st.slider("Term (A, Months)", 1, 60, value=term)
            filters_a['risk_level'] = st.radio("Risk Level (A)", RISK_LEVELS, index=RISK_LEVELS.index(risk_level))
            filters_a['innovation_level'] = st.radio("Innovation Level (A)", INNOVATION_LEVELS, index=INNOVATION_LEVELS.index(innovation_level))
            filters_a['launch_year'] = st.selectbox("Launch Year (A)", [2020, 2021, 2022, 2023, 2024], index=[2020,2021,2022,2023,2024].index(launch_year))
            st.session_state['filters_a'] = filters_a

        # --- FILTERS FOR B ---
        with col_b:
            st.markdown("#### Filters for B")
            filters_b = st.session_state.get('filters_b')
            if filters_b is None:
                seg, sector, promotion, channels, term, risk_level, innovation_level, launch_year = get_filter_defaults(st.session_state['parsed_b'])
            else:
                seg = filters_b.get('segment', [])
                sector = filters_b.get('sector', [])
                promotion = filters_b.get('promotion', [])
                channels = filters_b.get('channel', ["Digital"])
                term = filters_b.get('term', 12)
                risk_level = filters_b.get('risk_level', 'Medium')
                innovation_level = filters_b.get('innovation_level', 'Medium')
                launch_year = filters_b.get('launch_year', 2024)
            filters_b = {}
            filters_b['segment'] = st.multiselect("Segment (B)", ['Individual', 'SME', 'Corporate'], default=seg)
            filters_b['sector'] = st.multiselect("Sector (B)", SECTOR_LIST, default=sector)
            filters_b['promotion'] = st.multiselect("Promotion (B)", PROMOTIONS, default=promotion)
            filters_b['channel'] = st.multiselect("Channel (B)", ["Digital", "Branch", "Digital and Branch"], default=channels)
            filters_b['term'] = st.slider("Term (B, Months)", 1, 60, value=term)
            filters_b['risk_level'] = st.radio("Risk Level (B)", RISK_LEVELS, index=RISK_LEVELS.index(risk_level))
            filters_b['innovation_level'] = st.radio("Innovation Level (B)", INNOVATION_LEVELS, index=INNOVATION_LEVELS.index(innovation_level))
            filters_b['launch_year'] = st.selectbox("Launch Year (B)", [2020, 2021, 2022, 2023, 2024], index=[2020,2021,2022,2023,2024].index(launch_year))
            st.session_state['filters_b'] = filters_b

        if st.button("Run A/B Simulation"):
            st.session_state['run_a'] = True
            st.session_state['run_b'] = True

        if st.session_state.get('run_a', False) and st.session_state.get('run_b', False):
            n_individual, n_sme, n_corporate = 650, 220, 130
            df_customers = generate_customers(n_individual, n_sme, n_corporate)
            df_transactions = generate_transactions(df_customers, months=6, max_sample=15000)

            # A
            filters_a = st.session_state['filters_a']
            df_main_a = aggregate_transactions(df_customers, df_transactions)
            df_main_a['product_score'] = df_main_a.apply(lambda row: product_effect_score(row, filters_a), axis=1)
            proba_a = train_model(df_main_a, "RandomForest")
            df_main_a['product_interest_probability'] = proba_a * df_main_a['product_score']
            df_main_a['twin_response'] = df_main_a['product_interest_probability'].apply(lambda x:
                'apply/purchase' if x > 0.78 else
                'high interest' if x > 0.55 else
                'medium interest' if x > 0.35 else
                'neutral' if x > 0.18 else
                'negative response'
            )

            # B
            filters_b = st.session_state['filters_b']
            df_main_b = aggregate_transactions(df_customers, df_transactions)
            df_main_b['product_score'] = df_main_b.apply(lambda row: product_effect_score(row, filters_b), axis=1)
            proba_b = train_model(df_main_b, "RandomForest")
            df_main_b['product_interest_probability'] = proba_b * df_main_b['product_score']
            df_main_b['twin_response'] = df_main_b['product_interest_probability'].apply(lambda x:
                'apply/purchase' if x > 0.78 else
                'high interest' if x > 0.55 else
                'medium interest' if x > 0.35 else
                'neutral' if x > 0.18 else
                'negative response'
            )

            st.markdown("### A/B Results Visualization")
            col1, col2 = st.columns(2)
            with col1:
                plot_pie_twin_response(df_main_a, variant_label="A")
                plot_twin_distribution(df_main_a, variant_label="A")
                plot_segment_heatmap(df_main_a, variant_label="A")
                plot_sector_heatmap(df_main_a, variant_label="A")
                plot_segment_interest_heatmap(df_main_a, variant_label="A")
            with col2:
                plot_pie_twin_response(df_main_b, variant_label="B")
                plot_twin_distribution(df_main_b, variant_label="B")
                plot_segment_heatmap(df_main_b, variant_label="B")
                plot_sector_heatmap(df_main_b, variant_label="B")
                plot_segment_interest_heatmap(df_main_b, variant_label="B")

# ------ SINGLE SCENARIO ------
if test_mode == "Single Scenario":
    st.header("Single Scenario Test")
    user_input = st.text_area(
        "Describe your product/campaign below.",
        placeholder="Example: 18-month fixed interest POS loan for textile SMEs with cashback and loyalty program.",
        height=100
    )

    if st.button("Ask Bot"):
        if user_input.strip() == "":
            st.warning("Please enter a product/campaign description.")
            st.stop()
        parsed = parse_with_mistral(user_input)
        st.session_state['parsed'] = parsed
        st.session_state['filters'] = None
        st.session_state['run_simulation'] = False

    filters = st.session_state.get('filters')
    if filters is None and 'parsed' in st.session_state and st.session_state['parsed'] is not None:
        filters = dict(st.session_state['parsed'])
    elif filters is None:
        filters = {}

    seg = filters.get('segment') or []
    if isinstance(seg, str):
        seg = [seg] if seg else []
    sector = filters.get('sector') or []
    if isinstance(sector, str):
        sector = [sector] if sector else []
    promotion = filters.get('promotion') or []
    if isinstance(promotion, str):
        promotion = [promotion] if promotion else []
    channels = normalize_channels(filters.get('channel'))
    term = int(filters.get('term') or 12)
    risk_level = filters.get('risk_level', 'Medium')
    if risk_level not in RISK_LEVELS:
        risk_level = 'Medium'
    innovation_level = filters.get('innovation_level', 'Medium')
    if innovation_level not in INNOVATION_LEVELS:
        innovation_level = 'Medium'
    launch_year = int(filters.get('launch_year', 2024) or 2024)

    filters['segment'] = st.multiselect("Segment", ['Individual', 'SME', 'Corporate'], default=seg)
    filters['sector'] = st.multiselect("Sector", SECTOR_LIST, default=sector)
    filters['promotion'] = st.multiselect("Promotion", PROMOTIONS, default=promotion)
    filters['channel'] = st.multiselect("Channel", CHANNELS, default=channels)
    filters['term'] = st.slider("Term (Months)", 1, 60, value=term)
    filters['risk_level'] = st.radio("Risk Level", RISK_LEVELS, index=RISK_LEVELS.index(risk_level))
    filters['innovation_level'] = st.radio("Innovation Level", INNOVATION_LEVELS, index=INNOVATION_LEVELS.index(innovation_level))
    filters['launch_year'] = st.selectbox("Launch Year", [2020, 2021, 2022, 2023, 2024], index=[2020,2021,2022,2023,2024].index(launch_year))

    st.session_state['filters'] = filters

    if st.button("Run Simulation"):
        st.session_state['run_simulation'] = True

    if st.session_state.get('run_simulation', False) and st.session_state['filters']:
        filters = st.session_state['filters']
        n_individual, n_sme, n_corporate = 650, 220, 130
        df_customers = generate_customers(n_individual, n_sme, n_corporate)
        df_transactions = generate_transactions(df_customers, months=6, max_sample=15000)
        df_main = aggregate_transactions(df_customers, df_transactions)
        df_main['product_score'] = df_main.apply(lambda row: product_effect_score(row, filters), axis=1)
        proba = train_model(df_main, "RandomForest")
        df_main['product_interest_probability'] = proba * df_main['product_score']
        df_main['twin_response'] = df_main['product_interest_probability'].apply(lambda x:
            'apply/purchase' if x > 0.78 else
            'high interest' if x > 0.55 else
            'medium interest' if x > 0.35 else
            'neutral' if x > 0.18 else
            'negative response'
        )
        st.subheader("Results")
        plot_pie_twin_response(df_main, variant_label="A")
        plot_twin_distribution(df_main, variant_label="A")
        plot_segment_heatmap(df_main, variant_label="A")
        plot_sector_heatmap(df_main, variant_label="A")
        plot_segment_interest_heatmap(df_main, variant_label="A")
