import pandas as pd

def aggregate_transactions(df_customers, df_transactions):
    agg_df = df_transactions.groupby('customer_id').agg({
        'amount': ['mean', 'sum', 'count', 'max', 'std'],
        'category': lambda x: x.value_counts().idxmax(),
        'channel': lambda x: x.value_counts().idxmax(),
        'weekday': 'mean'
    })
    agg_df.columns = [
        'avg_amount', 'total_amount', 'tx_count', 'max_amount', 'std_amount',
        'top_category', 'top_channel', 'weekday_ratio'
    ]
    agg_df = agg_df.reset_index()
    agg_df['tx_category_count'] = df_transactions.groupby('customer_id')['category'].nunique().values

    if 'category' in df_customers.columns:
        agg_df = pd.merge(agg_df, df_customers[['customer_id', 'category']], on='customer_id', how='left')
        agg_df['main_spending'] = agg_df['category'].combine_first(agg_df['top_category'])
        agg_df.drop('category', axis=1, inplace=True)
    else:
        agg_df['main_spending'] = agg_df['top_category']

    agg_df['past_product_interest'] = ((agg_df['total_amount'] > 50000) & (agg_df['tx_category_count'] > 8)).astype(int)
    df_main = pd.merge(df_customers, agg_df, on='customer_id', how='left')
    return df_main

def product_effect_score(row, filters):
    score = 1
    if row['segment'] not in filters.get('segment', []):
        score *= 0.8
    if row['segment'] in ['SME', 'Corporate']:
        if row.get('sector', None) not in filters.get('sector', []):
            score *= 0.85
    if row['segment'] == 'Individual' and row.get('category', '') not in filters.get('category', []):
        score *= 0.85
    # Channel match: En az bir eşleşme aranır!
    filter_channels = filters.get('channel', ["Digital"])
    if isinstance(filter_channels, str):
        filter_channels = [filter_channels]
    if not any([ch in filter_channels for ch in row.get('channel', ["Digital"])]):
        score *= 0.7
    if "Cashback" in filters.get('promotion', []) and row.get('promotion_sensitivity', 0) > 0.5:
        score *= 1.12
    if "Digital Convenience" in filters.get('promotion', []) and row.get('digital_openness', 0) > 0.7:
        score *= 1.10
    if filters.get('innovation_level', 'Medium') == 'High' and row.get('innovation_openness', 0) > 0.7:
        score *= 1.08
    if filters.get('risk_level', 'Medium') == 'High':
        score *= 0.85
    if filters.get('term', 12) > 36:
        score *= 0.92
    if filters.get('launch_year', 2024) == 2024:
        score *= 1.06
    if row.get('tx_category_count', 0) > 8:
        score *= 1.04
    return score
