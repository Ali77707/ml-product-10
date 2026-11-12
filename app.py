import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['SENTENCE_TRANSFORMERS_HOME'] = os.path.join(os.path.dirname(__file__), '.st_cache')

import sys
import warnings
warnings.filterwarnings('ignore')

import uuid
import sqlite3
import json
import pandas as pd
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, jsonify, make_response
from collections import Counter
import re
from io import BytesIO
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go

matplotlib.rc('font', family='DejaVu Sans')

from data import load_and_prepare_dataset, mask_phone, mask_name
from models import SentimentAnalyzer
from recommender import Recommender
from utils import log_action, get_logs

app = Flask(__name__)
app.secret_key = 'student_project_secret_key_2025'

DB_PATH = 'reviews.db'
DATASET_PATH = 'reviews_dataset.csv'

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS reviews
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  text TEXT NOT NULL,
                  rating INTEGER NOT NULL,
                  product TEXT NOT NULL,
                  user_id TEXT NOT NULL,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

def save_review_to_db(text, rating, product, user_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO reviews (text, rating, product, user_id) VALUES (?, ?, ?, ?)",
              (text, rating, product, user_id))
    conn.commit()
    conn.close()

def get_user_id_from_request(request):
    user_id = request.cookies.get('user_id')
    if not user_id:
        user_id = str(uuid.uuid4())[:8]
    return user_id

init_db()

if os.path.exists('customer_details.csv') and not os.path.exists(DATASET_PATH):
    print("Generating dataset from customer_details.csv...")
    dataset = load_and_prepare_dataset('customer_details.csv')
    dataset.to_csv(DATASET_PATH, index=False, encoding='utf-8')
elif os.path.exists(DATASET_PATH):
    dataset = pd.read_csv(DATASET_PATH, parse_dates=['date'])
    print(f"Loaded dataset: {len(dataset)} records")
else:
    raise FileNotFoundError("customer_details.csv not found")

print("Initializing ML models...")
analyzer = SentimentAnalyzer(dataset)
analyzer.train_all_models()
analyzer.generate_report('student_report.png')

print("Initializing recommendation system...")
recommender = Recommender(dataset)

def get_rating_distribution_plot(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = df['rating'].value_counts().sort_index()
    ax.bar(counts.index, counts.values, color='royalblue', edgecolor='black', alpha=0.7, width=0.6)
    ax.set_title('Rating Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Rating')
    ax.set_ylabel('Count')
    ax.set_xticks([1, 2, 3, 4, 5])
    for i, v in enumerate(counts.values):
        ax.text(list(counts.index)[i], v + 0.5, str(v), ha='center', fontweight='bold')
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

def get_daily_trend_plot(df):
    fig, ax = plt.subplots(figsize=(10, 4))
    daily = df.groupby(df['date'].dt.date).size().reset_index(name='count')
    daily.columns = ['date', 'count']
    daily = daily.sort_values('date')
    ax.plot(daily['date'], daily['count'], 'b-o', linewidth=2, markersize=5)
    ax.fill_between(daily['date'], daily['count'], alpha=0.2, color='blue')
    ax.set_title('Reviews Over Time', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylabel('Count')
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

def get_wordcloud_img(df):
    texts = df['text']
    text = ' '.join(texts.values)
    if not text.strip():
        text = "no data"
    wc = WordCloud(width=800, height=400, background_color='white',
                   colormap='viridis', max_words=50, random_state=42).generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

@app.route('/')
def index():
    user_id = get_user_id_from_request(request)
    log_action(user_id, 'visit_index')

    dist_plot = get_rating_distribution_plot(dataset)
    trend_plot = get_daily_trend_plot(dataset)
    wc_plot = get_wordcloud_img(dataset)

    top_products = analyzer.get_top_products_by_positivity(5)
    neg_words = analyzer.get_top_negative_words(5)

    recommendations = recommender.get_top_for_user(user_id, top_n=5)

    resp = make_response(render_template('index.html',
        dist_plot=dist_plot,
        trend_plot=trend_plot,
        wc_plot=wc_plot,
        top_products=top_products.to_dict('records'),
        neg_words=neg_words,
        recommendations=recommendations,
        user_id=user_id,
        products=sorted(dataset['product'].unique())
    ))
    resp.set_cookie('user_id', user_id, max_age=60*60*24*365)
    return resp

@app.route('/predict', methods=['POST'])
def predict():
    user_id = get_user_id_from_request(request)
    text = request.form.get('text', '')
    if not text.strip():
        return jsonify({'error': 'Enter review text'})
    label, confidence = analyzer.predict_sentiment(text)
    log_action(user_id, 'predict_sentiment', f'"{text[:50]}..." -> {label}')
    return jsonify({'label': label, 'confidence': confidence})

@app.route('/add_review', methods=['POST'])
def add_review():
    user_id = get_user_id_from_request(request)
    text = request.form.get('text', '').strip()
    rating = request.form.get('rating', type=int, default=3)
    product = request.form.get('product', '')

    if not text or not product:
        return jsonify({'error': 'Fill all fields'})
    if rating < 1 or rating > 5:
        return jsonify({'error': 'Rating must be 1-5'})

    if rating >= 4:
        sent = 1
    elif rating <= 2:
        sent = 0
    else:
        sent = None

    global dataset
    new_row = pd.DataFrame([{
        'id': len(dataset) + 1,
        'text': text,
        'rating': rating,
        'sentiment': sent,
        'date': datetime.now(),
        'product': product,
        'user_id': user_id[:8],
        'user_name': f'User_{user_id[:4]}',
        'phone': '+10000000000'
    }])
    dataset = pd.concat([dataset, new_row], ignore_index=True)

    save_review_to_db(text, rating, product, user_id)

    label, confidence = analyzer.predict_sentiment(text)
    log_action(user_id, 'add_review', f'product={product} rating={rating} sentiment={label}')
    return jsonify({
        'success': True,
        'label': label,
        'confidence': confidence,
        'product': product
    })

@app.route('/stats')
def stats():
    user_id = get_user_id_from_request(request)
    log_action(user_id, 'visit_stats')

    product_stats = dataset.groupby('product').agg(
        avg_rating=('rating', 'mean'),
        review_count=('rating', 'count')
    ).reset_index()

    sent_stats = dataset[dataset['sentiment'].notna()].groupby('product').agg(
        positive_count=('sentiment', 'sum'),
        total_with_sentiment=('sentiment', 'count')
    ).reset_index()
    sent_stats['positive_share'] = (sent_stats['positive_count'] / sent_stats['total_with_sentiment'] * 100).round(1)

    product_stats = product_stats.merge(sent_stats[['product', 'positive_share']], on='product', how='left')
    product_stats['avg_rating'] = product_stats['avg_rating'].round(2)
    product_stats['positive_share'] = product_stats['positive_share'].fillna(0).round(1)

    return render_template('stats.html',
        stats=product_stats.to_dict('records'),
        user_id=user_id
    )

@app.route('/models')
def models_page():
    user_id = get_user_id_from_request(request)
    log_action(user_id, 'visit_models')

    results_before = analyzer.results_before
    results_after = analyzer.results_after
    roc_data = analyzer.get_roc_data()

    models_names = list(results_after.keys())

    fig = go.Figure()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for i, name in enumerate(models_names):
        if name in roc_data:
            fig.add_trace(go.Scatter(
                x=roc_data[name]['fpr'], y=roc_data[name]['tpr'],
                mode='lines', name=f"{name} (AUC={roc_data[name]['auc']})",
                line=dict(color=colors[i % len(colors)], width=2)
            ))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                             name='Random', line=dict(dash='dash', color='gray')))
    fig.update_layout(title='ROC Curves', xaxis_title='False Positive Rate',
                      yaxis_title='True Positive Rate', legend=dict(x=0.6, y=0.2),
                      height=500, template='plotly_white')
    roc_html = fig.to_html(full_html=False)

    return render_template('models.html',
        results_before=results_before,
        results_after=results_after,
        models_names=models_names,
        roc_html=roc_html,
        best_model=analyzer.best_model_name,
        user_id=user_id
    )

@app.route('/clusters')
def clusters_page():
    user_id = get_user_id_from_request(request)
    log_action(user_id, 'visit_clusters')

    method = request.args.get('method', 'kmeans')
    n_clusters = int(request.args.get('n_clusters', 4))

    result_df, cluster_info = analyzer.cluster_reviews(method=method, n_clusters=n_clusters)

    fig = px.scatter(
        result_df, x='x', y='y', color=result_df['cluster'].astype(str),
        hover_data={'text': True, 'sentiment': True},
        title=f'Review Clustering: {method.upper()} (PCA)',
        color_discrete_sequence=px.colors.qualitative.Set2,
        labels={'color': 'Cluster', 'x': 'PCA 1', 'y': 'PCA 2'}
    )
    fig.update_traces(marker=dict(size=8, line=dict(width=1, color='white')), selector=dict(mode='markers'))
    fig.update_layout(height=600, template='plotly_white', legend=dict(x=0.85, y=0.95))
    cluster_plot_html = fig.to_html(full_html=False)

    cluster_list = []
    for c_id, info in cluster_info.items():
        cluster_list.append({
            'id': c_id,
            'size': info['size'],
            'top_words': ', '.join(info['top_words'][:10]),
            'positive_share': info['positive_share'],
            'examples': result_df[result_df['cluster'] == int(c_id)]['text'].head(5).tolist()
        })

    return render_template('clusters.html',
        cluster_plot_html=cluster_plot_html,
        cluster_info=cluster_list,
        method=method,
        n_clusters=n_clusters,
        user_id=user_id
    )

@app.route('/logs')
def logs_page():
    user_id = get_user_id_from_request(request)
    log_action(user_id, 'visit_logs')
    logs = get_logs(50)
    return render_template('logs.html', logs=logs, user_id=user_id)

@app.route('/api/recommendations', methods=['GET'])
def api_recommendations():
    user_id = get_user_id_from_request(request)
    recs = recommender.get_top_for_user(user_id, top_n=5)
    return jsonify({'recommendations': recs})

if __name__ == '__main__':
    print("="*60)
    print(" Review Analysis System (Grade 5)")
    print("="*60)
    print(f" Dataset: {len(dataset)} records")
    print(f" Best model: {analyzer.best_model_name}")
    print(f" Web interface: http://127.0.0.1:5000")
    print("="*60)
    app.run(debug=False, host='127.0.0.1', port=5000)
