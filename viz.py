import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd

def plot_pie_twin_response(df, variant_label=""):
    response_counts = df['twin_response'].value_counts().reindex(
        ['apply/purchase', 'high interest', 'medium interest', 'neutral', 'negative response'], fill_value=0
    )
    nonzero = response_counts[response_counts > 0]
    labels = nonzero.index.tolist()
    sizes = nonzero.values.tolist()
    colors = ['green', 'limegreen', 'orange', 'skyblue', 'tomato'][:len(labels)]

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=None,
        autopct=lambda pct: f'{pct:.1f}%' if pct > 0 else '',
        colors=colors,
        startangle=90,
        pctdistance=0.7,
        textprops={'fontsize': 12, 'color': 'white'}
    )
    head = f"({variant_label}) " if variant_label else ""
    ax.set_title(f"{head}Twin Response Distribution (Pie Chart)", fontsize=13, pad=12)
    ax.set_aspect('equal')
    ax.legend(labels, loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=11, frameon=False)
    plt.tight_layout(pad=1)
    st.pyplot(fig)
    plt.close(fig)

def plot_twin_distribution(df, variant_label=""):
    response_counts = df['twin_response'].value_counts().reindex(
        ['apply/purchase', 'high interest', 'medium interest', 'neutral', 'negative response'], fill_value=0
    )
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    response_counts.plot(
        kind='bar',
        color=['green', 'limegreen', 'orange', 'skyblue', 'tomato'],
        ax=ax
    )
    head = f"({variant_label}) " if variant_label else ""
    ax.set_ylabel("Number of Customers", fontsize=11)
    ax.set_xlabel("Response", fontsize=11)
    ax.set_title(f"{head}General Twin Response Distribution", fontsize=13, pad=12)
    plt.tight_layout(pad=1)
    st.pyplot(fig)
    plt.close(fig)

def plot_segment_heatmap(df, variant_label=""):
    seg_pivot = pd.pivot_table(df, values='customer_id', index='segment', columns='twin_response', aggfunc='count', fill_value=0)
    all_responses = ['apply/purchase', 'high interest', 'medium interest', 'neutral', 'negative response']
    seg_pivot = seg_pivot.reindex(columns=all_responses, fill_value=0)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    sns.heatmap(seg_pivot, annot=True, fmt="d", cmap="YlGnBu", ax=ax, cbar=True)
    head = f"({variant_label}) " if variant_label else ""
    ax.set_title(f"{head}Segment-Response Heatmap", fontsize=13, pad=12)
    plt.tight_layout(pad=1)
    st.pyplot(fig)
    plt.close(fig)
    st.dataframe(seg_pivot)

def plot_sector_heatmap(df, variant_label=""):
    if 'sector' in df.columns:
        sec_pivot = pd.pivot_table(df, values='customer_id', index='sector', columns='twin_response', aggfunc='count', fill_value=0)
        all_responses = ['apply/purchase', 'high interest', 'medium interest', 'neutral', 'negative response']
        sec_pivot = sec_pivot.reindex(columns=all_responses, fill_value=0)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.heatmap(sec_pivot, annot=True, fmt="d", cmap="PuRd", ax=ax, cbar=True)
        head = f"({variant_label}) " if variant_label else ""
        ax.set_title(f"{head}Sector-Response Heatmap", fontsize=13, pad=12)
        plt.tight_layout(pad=1)
        st.pyplot(fig)
        plt.close(fig)
        st.dataframe(sec_pivot)

def plot_segment_interest_heatmap(df, variant_label=""):
    pivot = pd.pivot_table(df, values='product_interest_probability', index='segment', columns='twin_response', aggfunc='mean', fill_value=0)
    all_responses = ['apply/purchase', 'high interest', 'medium interest', 'neutral', 'negative response']
    pivot = pivot.reindex(columns=all_responses, fill_value=0)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax, cbar=True)
    head = f"({variant_label}) " if variant_label else ""
    ax.set_title(f"{head}Segment/Response-based Mean Product Interest Score", fontsize=13, pad=12)
    plt.tight_layout(pad=1)
    st.pyplot(fig)
    plt.close(fig)
