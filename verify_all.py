import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['SENTENCE_TRANSFORMERS_HOME'] = os.path.join(os.path.dirname(__file__), '.st_cache')

import sys
sys.path.insert(0, os.path.dirname(__file__))

from data import generate_dataset, mask_phone, mask_name
from models import SentimentAnalyzer
from recommender import Recommender

print("="*60)
print(" Проверка всех компонентов Grade 5")
print("="*60)

print("\n1. Генерация датасета...")
df = generate_dataset(300)
print(f"   Создано {len(df)} записей, {df['product'].nunique()} товаров, {df['user_id'].nunique()} пользователей")

print("\n2. Маскировка данных...")
print(f"   Телефон: +79991234567 -> {mask_phone('+79991234567')}")
print(f"   Имя: Александр Иванов -> {mask_name('Александр Иванов')}")

print("\n3. Обучение моделей...")
analyzer = SentimentAnalyzer(df)
analyzer.train_all_models()

print("\n4. Predicted sentiment test:")
for phrase in ["Отличный товар, всем рекомендую", "Ужасное качество, полный брак", "Нормальный товар за свою цену"]:
    label, conf = analyzer.predict_sentiment(phrase)
    print(f"   '{phrase[:40]}...' -> {label} ({conf}%)")

print("\n5. Топ товаров по позитиву:")
top = analyzer.get_top_products_by_positivity(5)
for _, row in top.iterrows():
    print(f"   {row['product'][:30]:30s} {row['positive_share']:.1f}%")

print("\n6. Частые слова в негативе:")
for word, count in analyzer.get_top_negative_words(5):
    print(f"   {word}: {count}")

print("\n7. Генерация отчёта...")
analyzer.generate_report('student_report.png')
print("   Отчёт сохранён: student_report.png")

print("\n8. Кластеризация...")
result_df, cluster_info = analyzer.cluster_reviews('kmeans', 4)
for cid, info in cluster_info.items():
    print(f"   Кластер {cid}: {info['size']} отзывов, позитив={info['positive_share']}%")

print("\n9. Рекомендательная система...")
rec = Recommender(df)
for uid in [1, 5, 10]:
    recs = rec.get_top_for_user(uid, 5)
    print(f"   Пользователь {uid}: {recs}")

print("\n10. ROC-кривые...")
roc = analyzer.get_roc_data()
for name, data in roc.items():
    print(f"   {name}: AUC={data['auc']}")

print("\n" + "="*60)
print(" ВСЕ КОМПОНЕНТЫ РАБОТАЮТ УСПЕШНО")
print("="*60)
