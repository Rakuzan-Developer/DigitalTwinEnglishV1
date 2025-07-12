import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd

def plot_twin_distribution(df):
    fig, ax = plt.subplots(figsize=(7, 4))
    df['twin_response'].value_counts().reindex(
        ['apply/purchase', 'high interest', 'medium interest', 'neutral', 'negative response']
    ).plot(kind='bar', color=['green', 'limegreen', 'orange', 'skyblue', 'tomato'], ax=ax)
    ax.set_ylabel("Number of Customers")
    ax.set_xlabel("Response")
    ax.set_title("General Twin Response Distribution")
    plt.tight_layout()
    st.pyplot(fig)

def plot_segment_heatmap(df):
    seg_pivot = pd.pivot_table(df, values='customer_id', index='segment', columns='twin_response', aggfunc='count', fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 3))
    sns.heatmap(seg_pivot, annot=True, fmt="d", cmap="YlGnBu", ax=ax)
    ax.set_title("Segment-Response Heatmap")
    st.pyplot(fig)
    st.dataframe(seg_pivot)

def plot_sector_heatmap(df):
    if 'sector' in df.columns:
        sec_pivot = pd.pivot_table(df, values='customer_id', index='sector', columns='twin_response', aggfunc='count', fill_value=0)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(sec_pivot, annot=True, fmt="d", cmap="PuRd", ax=ax)
        ax.set_title("Sector-Response Heatmap")
        plt.xticks(rotation=30)
        st.pyplot(fig)
        st.dataframe(sec_pivot)

def plot_pie_twin_response(df):
    response_counts = df['twin_response'].value_counts().reindex(
        ['apply/purchase', 'high interest', 'medium interest', 'neutral', 'negative response'], fill_value=0
    )
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(response_counts, labels=response_counts.index, autopct='%1.1f%%',
           colors=['green', 'limegreen', 'orange', 'skyblue', 'tomato'], startangle=90)
    ax.set_title("Twin Response Distribution (Pie Chart)")
    st.pyplot(fig)

def plot_segment_interest_heatmap(df):
    pivot = pd.pivot_table(df, values='product_interest_probability', index='segment', columns='twin_response', aggfunc='mean', fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 3))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
    ax.set_title("Segment/Response-based Mean Product Interest Score")
    st.pyplot(fig)

