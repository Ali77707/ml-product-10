import pandas as pd
df = pd.read_csv('yelp_reviews.csv')
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(df[['text','stars']].head(3).to_string())
stars_dist = df['stars'].value_counts().sort_index().to_dict()
print(f"Stars distribution: {stars_dist}")
unique_biz = df['business_id'].nunique() if 'business_id' in df.columns else 'N/A'
unique_users = df['user_id'].nunique() if 'user_id' in df.columns else 'N/A'
print(f"Unique businesses: {unique_biz}")
print(f"Unique users: {unique_users}")
