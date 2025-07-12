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
