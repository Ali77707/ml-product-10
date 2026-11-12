import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['SENTENCE_TRANSFORMERS_HOME'] = os.path.join(os.path.dirname(__file__), '.st_cache')

import sys
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data import load_and_prepare_dataset, mask_phone, mask_name
from models import SentimentAnalyzer
from recommender import Recommender
from utils import log_action
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

@pytest.fixture(scope="module")
def dataset():
    df = load_and_prepare_dataset('customer_details.csv')
    df = df.head(500).copy()
    return df

@pytest.fixture(scope="module")
def analyzer(dataset):
    a = SentimentAnalyzer(dataset)
    a.train_all_models()
    return a

@pytest.fixture(scope="module")
def recommender(dataset):
    return Recommender(dataset)

def test_mask_phone():
    phone = "+79991234567"
    masked = mask_phone(phone)
    assert masked == "+7999***4567"
    assert len(masked) == len(phone)
    assert "***" in masked
    assert masked[0] == "+"

def test_mask_phone_short():
    assert mask_phone("123") == "123"
    assert mask_phone("") == ""

def test_mask_name():
    name = "Александр Иванов"
    masked = mask_name(name)
    assert "***" in masked
    assert len(masked.split()) == 2
    assert "***" in masked
    assert len(masked.split()) == 2

def test_mask_name_short():
    assert mask_name("Ан") == "А*"
    assert mask_name("Иван") == "И**н"

def test_predict_sentiment(analyzer):
    label, conf = analyzer.predict_sentiment("Отличный товар, всем рекомендую")
    assert label in ["ПОЗИТИВ", "НЕГАТИВ"]
    assert 0 <= conf <= 100

    label2, conf2 = analyzer.predict_sentiment("Ужасное качество, полный брак")
    assert label2 in ["ПОЗИТИВ", "НЕГАТИВ"]
    assert 0 <= conf2 <= 100

def test_predict_sentiment_empty(analyzer):
    label, conf = analyzer.predict_sentiment("")
    assert label == "Ошибка" or isinstance(label, str)

def test_load_data(dataset):
    assert len(dataset) >= 200
    required_cols = ['text', 'rating', 'product', 'date', 'user_id']
    for col in required_cols:
        assert col in dataset.columns, f"Column {col} missing"
    assert dataset['rating'].between(1, 5).all()
    assert dataset['text'].notna().all()

def test_sentiment_correlation(dataset):
    high_rating = dataset[dataset['rating'] >= 4]
    low_rating = dataset[dataset['rating'] <= 2]
    high_pos = high_rating[high_rating['sentiment'] == 1]
    low_pos = low_rating[low_rating['sentiment'] == 1]
    assert len(high_pos) / len(high_rating) > 0.5
    assert len(low_pos) / len(low_rating) < 0.5

def test_metrics_computation(analyzer):
    assert hasattr(analyzer, 'results_before')
    assert len(analyzer.results_before) >= 5
    for name, metrics in analyzer.results_before.items():
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['f1'] <= 1

def test_recommendations(recommender):
    recs = recommender.get_top_for_user(1, top_n=3)
    assert len(recs) <= 5
    assert len(recs) > 0
    assert all(isinstance(r, str) for r in recs)

def test_recommendations_new_user(recommender):
    recs = recommender.get_top_for_user(9999, top_n=5)
    assert len(recs) > 0

def test_clustering(analyzer):
    result_df, cluster_info = analyzer.cluster_reviews(method='kmeans', n_clusters=4)
    assert 'cluster' in result_df.columns
    assert 'x' in result_df.columns
    assert 'y' in result_df.columns
    assert len(cluster_info) > 0

def test_top_products(dataset, analyzer):
    top = analyzer.get_top_products_by_positivity(5)
    assert len(top) <= 5
    assert 'positive_share' in top.columns

def test_neg_words(dataset, analyzer):
    words = analyzer.get_top_negative_words(5)
    assert len(words) <= 5

def test_gridsearch(analyzer):
    for name in ['LogisticRegression', 'RandomForest', 'XGBoost', 'MLPClassifier', 'CatBoost']:
        assert name in analyzer.results_after
        assert 'best_params' in analyzer.results_after[name]

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
