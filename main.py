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
from chatbot import parse_with_ollama

st.set_page_config(page_title="Digital Twin & AI Customer Simulation Demo", layout="wide")
st.title("Enterprise-scale Digital Twin & AI Segmentation Demo")

# ----- SESSION STATE INIT -----
if 'parsed' not in st.session_state:
    st.session_state['parsed'] = None
if 'run_simulation' not in st.session_state:
    st.session_state['run_simulation'] = False
if 'filters' not in st.session_state:
    st.session_state['filters'] = {}

# ----- CHATBOT INPUT -----
st.header("💬 Digital Twin Chatbot Assistant")

user_input = st.text_area(
    "Describe your product/campaign below. The AI assistant will extract relevant filters and you can further edit them.",
    placeholder="Example: 18-month fixed interest loan for textile SMEs with cashback and loyalty program.",
    height=100
)

if st.button("Ask Bot"):
    if user_input.strip() == "":
        st.warning("Please enter a product/campaign description.")
        st.stop()
    parsed = parse_with_ollama(user_input)
    st.session_state['parsed'] = parsed
    st.session_state['run_simulation'] = False
    st.session_state['filters'] = {}  # Reset any old filters

# ----- FILTERS SECTION -----
if st.session_state['parsed'] is not None:
    parsed = st.session_state['parsed']

    # Get current or default filter values
    segment = st.multiselect("Segment", ['Individual', 'SME', 'Corporate'],
                             default=st.session_state['filters'].get('segment', parsed.get('segment', [])))
    sector = st.multiselect("Sector", SECTOR_LIST,
                            default=st.session_state['filters'].get('sector', parsed.get('sector', [])))
    product_type = st.selectbox("Product Type", PRODUCT_TYPES,
                               index=PRODUCT_TYPES.index(
                                   st.session_state['filters'].get('product_type', parsed.get('product_type', PRODUCT_TYPES[0]))
                               ) if st.session_state['filters'].get('product_type', parsed.get('product_type')) in PRODUCT_TYPES else 0)
    product_category = st.selectbox("Product Category", PRODUCT_CATEGORIES,
                                   index=PRODUCT_CATEGORIES.index(
                                       st.session_state['filters'].get('product_category', parsed.get('product_category', PRODUCT_CATEGORIES[0]))
                                   ) if st.session_state['filters'].get('product_category', parsed.get('product_category')) in PRODUCT_CATEGORIES else 0)
    promotion = st.multiselect("Promotion", PROMOTIONS,
                               default=st.session_state['filters'].get('promotion', parsed.get('promotion', [])))
    channel = st.selectbox("Channel", ["Digital", "Branch", "Both"],
                           index=["Digital", "Branch", "Both"].index(
                               st.session_state['filters'].get('channel', parsed.get('channel', "Digital"))))
    term = st.slider("Term (Months)", 1, 60,
                     value=int(st.session_state['filters'].get('term', parsed.get('term', 12))))
    interest_type = st.selectbox("Interest Type", ['Fixed', 'Variable', 'None'],
                                 index=['Fixed', 'Variable', 'None'].index(
                                     st.session_state['filters'].get('interest_type', parsed.get('interest_type', 'Fixed'))))
    risk_level = st.radio("Risk Level", ['High', 'Medium', 'Low'],
                          index=['High', 'Medium', 'Low'].index(
                              st.session_state['filters'].get('risk_level', parsed.get('risk_level', 'Medium'))))
    innovation_level = st.radio("Innovation Level", ['High', 'Medium', 'Low'],
                                index=['High', 'Medium', 'Low'].index(
                                    st.session_state['filters'].get('innovation_level', parsed.get('innovation_level', 'Medium'))))
    launch_year = st.selectbox("Launch Year", [2020, 2021, 2022, 2023, 2024],
                               index=[2020, 2021, 2022, 2023, 2024].index(
                                   st.session_state['filters'].get('launch_year', parsed.get('launch_year', 2024))))

    # Save current filter values to session
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

# ----- SIMULATION AND VISUALS -----
if st.session_state['run_simulation'] and st.session_state['filters']:
    # Get filters from session
    filters = st.session_state['filters']

    n_individual, n_sme, n_corporate = 650, 220, 130
    df_main = aggregate_transactions(
        *[
            generate_customers(n_individual, n_sme, n_corporate),
            generate_transactions(generate_customers(n_individual, n_sme, n_corporate), months=6, max_sample=15000)
        ]
    )

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

    # --- RESULTS ---
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

