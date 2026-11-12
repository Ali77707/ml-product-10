import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from datasets import load_dataset

print("Загрузка Amazon US Reviews (категория: Wireless, 500 записей)...")
ds = load_dataset("amazon_us_reviews", "Wireless_v1_00", split="train", streaming=True)

rows = []
for i, row in enumerate(ds):
    if i >= 500:
        break
    rows.append({
        'text': row['review_body'],
        'rating': row['star_rating'],
        'product': row['product_title'],
        'date': row['review_date'],
        'user_id': row['customer_id']
    })

df = pd.DataFrame(rows)
df['sentiment'] = df['rating'].apply(lambda r: 1 if r >= 4 else (0 if r <= 2 else None))
df.to_csv('amazon_reviews_500.csv', index=False, encoding='utf-8-sig')
print(f"Сохранено {len(df)} записей")
print(f"Колонки: {list(df.columns)}")
print(f"Рейтинг: {df['rating'].value_counts().sort_index().to_dict()}")
print(f"Уникальных товаров: {df['product'].nunique()}")
print(f"Уникальных пользователей: {df['user_id'].nunique()}")
print("\nПример:")
print(df[['text', 'rating', 'product']].head(3).to_string())
