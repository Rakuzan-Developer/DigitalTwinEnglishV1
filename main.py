import streamlit as st
import pandas as pd
import numpy as np

from config import SECTOR_LIST, INDIVIDUAL_CATEGORIES, CATEGORY_LIST, PRODUCT_TYPES, PRODUCT_CATEGORIES, PROMOTIONS
from data_generator import generate_customers, generate_transactions
from features import aggregate_transactions
from model_train import train_model
from viz import (
    plot_twin_distribution, plot_segment_heatmap, plot_sector_heatmap,
    plot_scatter_interest, plot_pie_twin_response, plot_segment_interest_heatmap
)

st.set_page_config(page_title="Digital Twin & AI Customer Simulation Demo", layout="wide")
st.title("Enterprise-scale Digital Twin & AI Segmentation Demo")
st.markdown("""
This demo powerfully visualizes customer reactions, segment breakdowns, and data analytics for your product/campaign.
""")

# --- FIXED CUSTOMER COUNTS FOR SPEED ---
n_individual = 650   # Individual customers
n_sme = 220          # SME customers
n_corporate = 130    # Corporate customers

st.sidebar.markdown("**Customer counts are optimized for demo speed (Total: 1000: 650 Individual, 220 SME, 130 Corporate).**")

@st.cache_data
def load_data(n_individual, n_sme, n_corporate):
    df_customers = generate_customers(n_individual, n_sme, n_corporate)
    df_transactions = generate_transactions(df_customers, months=6, max_sample=15000)
    df_main = aggregate_transactions(df_customers, df_transactions)
    return df_main

df_main = load_data(n_individual, n_sme, n_corporate)

with st.sidebar:
    st.header("New Product/Campaign Features")

    product_type = st.selectbox("Product Type", PRODUCT_TYPES, help="Type of product offered")
    product_category = st.selectbox("Product Category", PRODUCT_CATEGORIES, help="Category (loan/payment/POS etc.)")
    target_segment = st.multiselect("Target Customer Segment", ['Individual', 'SME', 'Corporate'], default=['Individual', 'SME', 'Corporate'])
    target_sector = st.multiselect("Target Sector", sorted(SECTOR_LIST), default=sorted(SECTOR_LIST), help="For SME/Corporate")
    target_category = st.multiselect("Target Category (for Individuals)", INDIVIDUAL_CATEGORIES, default=INDIVIDUAL_CATEGORIES)
    term = st.slider("Term (Months)", 1, 60, 12, help="Product term")
    interest_type = st.selectbox("Interest Type", ['Fixed', 'Variable', 'None'], help="For loan or similar products")
    promotions = st.multiselect("Promotion/Campaign", PROMOTIONS, default=["Cashback"])
    innovation_level = st.radio("Product Innovation Level", ['High', 'Medium', 'Low'], index=1)
    risk_level = st.radio("Product Risk Level", ['High', 'Medium', 'Low'], index=1)
    launch_year = st.selectbox("Product Launch Year", [2020, 2021, 2022, 2023, 2024])
    channel = st.selectbox("Channel", ["Digital", "Branch", "Both"])

    model_choice = st.radio("Model Choice", ["RandomForest", "XGBoost", "DeepLearning - MLP", "DeepLearning - TabNet"])

def product_effect_score(row):
    score = 1
    if row['segment'] not in target_segment: score *= 0.8
    if row['segment'] in ['SME', 'Corporate']:
        if row['sector'] not in target_sector: score *= 0.85
    if row['segment'] == 'Individual' and row['category'] not in target_category: score *= 0.85
    if channel == "Digital" and row['digital_openness'] < 0.4: score *= 0.7
    if "Cashback" in promotions and row['promotion_sensitivity'] > 0.5: score *= 1.12
    if "Digital Convenience" in promotions and row['digital_openness'] > 0.7: score *= 1.10
    if innovation_level == 'High' and row['innovation_openness'] > 0.7: score *= 1.08
    if risk_level == 'High': score *= 0.85
    if term > 36: score *= 0.92
    if launch_year == 2024: score *= 1.06
    if row['tx_category_count'] > 8: score *= 1.04
    return score

def twin_response_func(x):
    if x > 0.78: return 'apply/purchase'
    elif x > 0.55: return 'high interest'
    elif x > 0.35: return 'medium interest'
    elif x > 0.18: return 'neutral'
    else: return 'negative response'

proba = train_model(df_main, model_choice)
df_main = df_main.copy()
df_main['product_score'] = df_main.apply(product_effect_score, axis=1)
df_main['product_interest_probability'] = proba * df_main['product_score']
df_main['twin_response'] = df_main['product_interest_probability'].apply(twin_response_func)

# --- New Charts & Analysis ---
st.subheader(f"Simulation Results for First 100 Customers ({model_choice})")
st.dataframe(df_main[['customer_id', 'segment', 'sector', 'category', 'avg_amount', 'tx_category_count', 'twin_response', 'product_interest_probability']].head(100))

st.subheader("Overall Twin Response Distribution")
plot_twin_distribution(df_main)

st.subheader("Twin Response Distribution (Pie Chart)")
plot_pie_twin_response(df_main)

st.subheader("Segment-based Twin Response Heatmap")
plot_segment_heatmap(df_main)

st.subheader("Sector-based Twin Response Heatmap")
plot_sector_heatmap(df_main)

st.subheader("Digital Openness vs. Product Interest Score Distribution")
plot_scatter_interest(df_main)

st.subheader("Average Product Interest Score by Segment/Response (Heatmap)")
plot_segment_interest_heatmap(df_main)

st.subheader("Individual Twin Analysis Panel")
selected_customer = st.selectbox("Select a customer", df_main['customer_id'].sample(n=50, random_state=42).tolist())
st.write(df_main[df_main['customer_id'] == selected_customer].T)
