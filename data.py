import pandas as pd
import numpy as np
import random
import re
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

positive_phrases = [
    "Absolutely love this {product}! The quality exceeded my expectations.",
    "Great {product}, very satisfied with my purchase. Highly recommend!",
    "Excellent quality {product}. Would definitely buy again.",
    "Perfect {product}! Exactly what I was looking for. Fast delivery too.",
    "Very happy with this {product}. Great value for the price.",
    "Outstanding {product}! The material is premium and the fit is perfect.",
    "Love love love this {product}! Best purchase I've made this year.",
    "This {product} is amazing. Super comfortable and looks great.",
    "Extremely pleased with this {product}. Quality is top notch.",
    "Wonderful {product}! Exceeded all my expectations. Five stars!",
    "Great quality {product} at a reasonable price. Very impressed.",
    "This is the best {product} I have ever bought. Highly recommended!",
    "Super happy with my new {product}. Arrived quickly and in perfect condition.",
    "Fantastic {product}! Comfortable, stylish, and durable. What more could you want?",
    "I am in love with this {product}. The quality is outstanding for the price.",
    "Really impressed with this {product}. Better than I expected!",
    "Amazing {product}! Will definitely be ordering more from this brand.",
    "This {product} is perfect for everyday use. Very well made.",
    "Couldn't be happier with my {product}. Great purchase!",
    "Top quality {product}. Fast shipping, excellent customer service."
]

negative_phrases = [
    "Very disappointed with this {product}. The quality is poor and it didn't last.",
    "Not worth the money. This {product} broke within a week.",
    "Terrible {product}. Cheap materials and poor construction.",
    "Would not recommend this {product}. Very dissatisfied with my purchase.",
    "Poor quality {product}. Definitely not as described in the listing.",
    "This {product} is a waste of money. So disappointing.",
    "The {product} looks nothing like the picture. Very unhappy.",
    "Low quality materials used. This {product} started falling apart immediately.",
    "Extremely disappointed. The {product} is not worth half the price.",
    "Awful {product}! Do not waste your money on this.",
    "The {product} arrived damaged and the customer service was unhelpful.",
    "Very poor craftsmanship on this {product}. Would give zero stars if I could.",
    "Regret buying this {product}. It's uncomfortable and poorly made.",
    "This {product} is terrible quality. Sent it back for a refund.",
    "Not impressed at all. The {product} feels cheap and flimsy.",
    "Save your money. This {product} is not worth it at all.",
    "Really bad experience with this {product}. Doesn't fit as expected.",
    "The quality control on this {product} is terrible. Has defects.",
    "Disappointed. The {product} started showing wear after just one use.",
    "Horrible {product}! Worst online purchase I have ever made."
]

neutral_phrases = [
    "The {product} is okay for the price. Nothing special but does the job.",
    "Average {product}. It's decent but I've seen better quality elsewhere.",
    "It's alright. The {product} serves its purpose but not amazing.",
    "Decent {product} for everyday use. Could be better quality though.",
    "Not bad but not great either. The {product} is just average.",
    "The {product} is fine but I expected more based on the reviews.",
    "Mediocre {product}. It works but I probably wouldn't buy again.",
    "It's an okay {product}. Nothing to write home about.",
    "Pretty standard {product}. Gets the job done but that's about it.",
    "Middle of the road {product}. Not terrible, not amazing.",
    "The {product} is satisfactory for the price point.",
    "Acceptable quality but I've had better {product}s for less money.",
    "It's a mixed bag. The {product} looks good but the quality is average.",
    "Reasonable {product} for the price. Not great but not terrible.",
    "The {product} is fine. Does what it's supposed to do."
]

def generate_review_text(rating, product):
    if rating >= 4.0:
        phrases = positive_phrases
    elif rating <= 2.5:
        phrases = negative_phrases
    else:
        phrases = neutral_phrases
    return random.choice(phrases).format(product=product.lower())

def mask_phone(phone):
    if len(phone) >= 10:
        return phone[:5] + "***" + phone[-4:]
    return phone

def mask_name(name):
    parts = name.split()
    masked = []
    for part in parts:
        if len(part) <= 2:
            masked.append(part[0] + "*" * (len(part) - 1))
        else:
            masked.append(part[0] + "*" * (len(part) - 2) + part[-1])
    return " ".join(masked)

def load_and_prepare_dataset(csv_path='customer_details.csv'):
    df = pd.read_csv(csv_path)

    records = []
    for _, row in df.iterrows():
        rating = row['Review Rating']
        product = row['Item Purchased']
        user_id = int(row['Customer ID'])
        text = generate_review_text(rating, product)

        if rating >= 4.0:
            sentiment = 1
        elif rating <= 2.5:
            sentiment = 0
        else:
            sentiment = None

        purchase_amount = row.get('Purchase Amount (USD)', 0)
        category = row.get('Category', 'General')

        records.append({
            'id': len(records) + 1,
            'text': text,
            'rating': round(rating),
            'rating_raw': rating,
            'sentiment': sentiment,
            'date': datetime.now() - timedelta(days=random.randint(0, 90), hours=random.randint(0, 24)),
            'product': product,
            'category': category,
            'user_id': user_id,
            'user_name': f"Customer_{user_id}",
            'phone': f"+1{random.randint(200, 999)}{random.randint(1000000, 9999999)}"
        })

    result = pd.DataFrame(records)
    return result

def generate_dataset(n=300):
    products = [
        'Smartphone', 'Laptop', 'Headphones', 'Smart Watch', 'Tablet',
        'Backpack', 'Sneakers', 'Jacket', 'T-Shirt', 'Coffee Maker'
    ]
    categories = ['Electronics', 'Clothing', 'Home', 'Accessories']
    records = []

    for i in range(n):
        rating = random.choices([1, 2, 3, 4, 5], weights=[1, 1, 2, 3, 3], k=1)[0]
        product = random.choice(products)

        if rating >= 4:
            sentiment = 1
        elif rating <= 2:
            sentiment = 0
        else:
            sentiment = None

        user_id = random.randint(1, max(30, n // 3))
        records.append({
            'id': i + 1,
            'text': generate_review_text(rating, product),
            'rating': rating,
            'rating_raw': float(rating),
            'sentiment': sentiment,
            'date': datetime.now() - timedelta(days=random.randint(0, 90), hours=random.randint(0, 24)),
            'product': product,
            'category': random.choice(categories),
            'user_id': user_id,
            'user_name': f'Customer_{user_id}',
            'phone': f'+1{random.randint(200, 999)}{random.randint(1000000, 9999999)}'
        })

    return pd.DataFrame(records)

if __name__ == "__main__":
    df = load_and_prepare_dataset()
    print(f"Создано {len(df)} записей")
    print(f"Уникальных товаров: {df['product'].nunique()}")
    print(f"Уникальных категорий: {df['category'].nunique()}")
    print(f"Уникальных пользователей: {df['user_id'].nunique()}")
    print(f"\nРаспределение оценок (округлённых):")
    print(df['rating'].value_counts().sort_index().to_string())
    print(f"\nРаспределение тональности:")
    print(df['sentiment'].value_counts(dropna=False).to_string())
    print(f"\nПримеры:")
    for _, r in df.head(3).iterrows():
        print(f"  [{r['rating']} star] {r['product']}: {r['text'][:60]}...")
    df.to_csv('reviews_dataset.csv', index=False, encoding='utf-8')
    print("\nСохранено в reviews_dataset.csv")
