# data_generator.py
# Generates dummy customer and transaction data, supports sampling for large datasets and fast demo generation.

import numpy as np
import pandas as pd
from config import SECTOR_LIST, INDIVIDUAL_CATEGORIES, CATEGORY_LIST

def generate_customers(n_individual=10000, n_sme=2000, n_corporate=1000, seed=42):
    np.random.seed(seed)
    individual_df = pd.DataFrame({
        'customer_id': [f'INDIVIDUAL_{i+1}' for i in range(n_individual)],
        'segment': 'Individual',
        'category': np.random.choice(INDIVIDUAL_CATEGORIES, n_individual),
        'financial_performance': np.random.randint(1, 11, n_individual),
        'digital_openness': np.random.uniform(0, 1, n_individual),
        'promotion_sensitivity': np.random.uniform(0, 1, n_individual),
        'innovation_openness': np.random.uniform(0, 1, n_individual)
    })
    sme_df = pd.DataFrame({
        'customer_id': [f'SME_{i+1}' for i in range(n_sme)],
        'segment': 'SME',
        'sector': np.random.choice(SECTOR_LIST, n_sme),
        'financial_performance': np.random.randint(1, 11, n_sme),
        'digital_openness': np.random.uniform(0, 1, n_sme),
        'promotion_sensitivity': np.random.uniform(0, 1, n_sme),
        'innovation_openness': np.random.uniform(0, 1, n_sme)
    })
    corporate_df = pd.DataFrame({
        'customer_id': [f'CORPORATE_{i+1}' for i in range(n_corporate)],
        'segment': 'Corporate',
        'sector': np.random.choice(SECTOR_LIST, n_corporate),
        'financial_performance': np.random.randint(1, 11, n_corporate),
        'digital_openness': np.random.uniform(0, 1, n_corporate),
        'promotion_sensitivity': np.random.uniform(0, 1, n_corporate),
        'innovation_openness': np.random.uniform(0, 1, n_corporate)
    })
    df_customers = pd.concat([individual_df, sme_df, corporate_df], ignore_index=True)
    df_customers['sector'] = df_customers['sector'].fillna('None')
    df_customers['category'] = df_customers['category'].fillna('Corporate')
    return df_customers

def generate_transactions(df_customers, months=6, max_sample=20000):
    # For large data, only create transactions for first N customers (to protect memory in demo)
    if len(df_customers) > max_sample:
        print(f"WARNING: Transactions are generated for only the first {max_sample} customers (for demo)!")
        df_customers = df_customers.sample(n=max_sample, random_state=42)
    category_list = CATEGORY_LIST
    transaction_list = []
    for idx, row in df_customers.iterrows():
        for month in range(1, months+1):
            n_trans = np.random.randint(12, 36)
            if row['segment'] == 'Individual':
                p_cat = np.array([0.25 if k == row['category'] else 0.75/(len(category_list)-1) for k in category_list])
                p_cat = p_cat / p_cat.sum()
            else:
                p_cat = np.ones(len(category_list)) / len(category_list)
            for t in range(n_trans):
                category = np.random.choice(category_list, p=p_cat)
                amount = np.round(np.random.uniform(100, 20000), 2)
                transaction_list.append({
                    'customer_id': row['customer_id'],
                    'month': month,
                    'amount': amount,
                    'category': category,
                    'channel': np.random.choice(['Digital', 'Branch', 'ATM'], p=[0.7, 0.18, 0.12]),
                    'weekday': np.random.choice([0, 1], p=[0.22, 0.78])
                })
    return pd.DataFrame(transaction_list)
