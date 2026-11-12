import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
try:
    from catboost import CatBoostClassifier
except ImportError:
    class CatBoostClassifier(BaseEstimator, ClassifierMixin):
        def __init__(self, random_seed=42, verbose=0, allow_writing_files=False,
                     iterations=100, depth=4, learning_rate=0.1):
            self.random_seed = random_seed
            self.verbose = verbose
            self.allow_writing_files = allow_writing_files
            self.iterations = iterations
            self.depth = depth
            self.learning_rate = learning_rate

        def fit(self, X, y):
            self.model_ = GradientBoostingClassifier(
                n_estimators=self.iterations,
                max_depth=self.depth,
                learning_rate=self.learning_rate,
                random_state=self.random_seed
            )
            self.model_.fit(X, y)
            return self

        def predict(self, X):
            return self.model_.predict(X)

        def predict_proba(self, X):
            return self.model_.predict_proba(X)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
from collections import Counter
import re

class SentimentAnalyzer:
    def __init__(self, df):
        self.df = df
        self.models = {}
        self.results = {}
        self.results_before = {}
        self.results_after = {}
        self.vectorizer_tfidf = None
        self.vectorizer_st = None
        self.best_model = None
        self.best_model_name = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def prepare_data(self):
        df_binary = self.df[self.df['sentiment'].notna()].copy()
        X = df_binary['text'].values
        y = df_binary['sentiment'].values
        return X, y, df_binary

    def vectorize_tfidf(self, X):
        self.vectorizer_tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 3))
        return self.vectorizer_tfidf.fit_transform(X)

    def vectorize_st(self, X):
        try:
            from sentence_transformers import SentenceTransformer
            self.vectorizer_st = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            return self.vectorizer_st.encode(X)
        except Exception as e:
            print(f"SentenceTransformer failed: {e}, falling back to TF-IDF")
            return self.vectorize_tfidf(X)

    def split(self, X_vec, y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_vec, y, test_size=0.2, random_state=42, stratify=y
        )

    def train_and_evaluate(self, model, model_name, param_grid=None):
        if param_grid:
            grid = GridSearchCV(model, param_grid, cv=3, scoring='f1', n_jobs=1, verbose=0)
            grid.fit(self.X_train, self.y_train)
            best = grid.best_estimator_
            y_pred = best.predict(self.X_test)
            params = grid.best_params_
            eval_model = best
        else:
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            params = {}
            eval_model = model

        y_prob = eval_model.predict_proba(self.X_test)[:, 1] if hasattr(eval_model, 'predict_proba') else None

        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred),
            'recall': recall_score(self.y_test, y_pred),
            'f1': f1_score(self.y_test, y_pred)
        }

        return metrics, y_pred, y_prob, params

    def train_all_models(self):
        X, y, _ = self.prepare_data()
        print("Векторизация TF-IDF...")
        X_vec = self.vectorize_tfidf(X)
        self.split(X_vec, y)

        models_config = {
            'LogisticRegression': {
                'model': LogisticRegression(max_iter=1000, random_state=42),
                'params': {
                    'C': [0.1, 1.0, 10.0],
                    'solver': ['liblinear', 'lbfgs']
                }
            },
            'RandomForest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, None]
                }
            },
            'XGBoost': {
                'model': XGBClassifier(random_state=42, verbosity=0, use_label_encoder=False),
                'params': {
                    'learning_rate': [0.05, 0.1, 0.3],
                    'max_depth': [3, 6, 9]
                }
            },
            'MLPClassifier': {
                'model': MLPClassifier(max_iter=500, random_state=42),
                'params': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 25)],
                    'activation': ['relu', 'tanh']
                }
            },
            'CatBoost': {
                'model': CatBoostClassifier(random_seed=42, verbose=0, allow_writing_files=False),
                'params': {
                    'iterations': [100, 200],
                    'depth': [4, 6],
                    'learning_rate': [0.05, 0.1]
                }
            }
        }

        print("Обучение моделей (до подбора гиперпараметров)...")
        for name, config in models_config.items():
            print(f"  {name}...")
            metrics, y_pred, y_prob, _ = self.train_and_evaluate(
                config['model'], name
            )
            self.results_before[name] = metrics
            self.models[name + '_default'] = config['model']
            config['model'].fit(self.X_train, self.y_train)

        print("Подбор гиперпараметров (GridSearchCV)...")
        for name, config in models_config.items():
            print(f"  {name}...")
            metrics, y_pred, y_prob, best_params = self.train_and_evaluate(
                config['model'], name, config['params']
            )
            self.results_after[name] = {
                'metrics': metrics,
                'best_params': best_params
            }

        self._find_best_model()
        return self.results_before, self.results_after

    def _find_best_model(self):
        best_f1 = 0
        for name, res in self.results_after.items():
            if res['metrics']['f1'] > best_f1:
                best_f1 = res['metrics']['f1']
                self.best_model_name = name

        models_map = {
            'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
            'RandomForest': RandomForestClassifier(random_state=42),
            'XGBoost': XGBClassifier(random_state=42, verbosity=0),
            'MLPClassifier': MLPClassifier(max_iter=500, random_state=42),
            'CatBoost': CatBoostClassifier(random_seed=42, verbose=0)
        }
        bp = self.results_after[self.best_model_name]['best_params']
        model = models_map[self.best_model_name]
        model.set_params(**bp)
        model.fit(self.X_train, self.y_train)
        self.best_model = model

    def predict_sentiment(self, text):
        if self.vectorizer_tfidf is None or self.best_model is None:
            return "Ошибка", 0.0
        X = self.vectorizer_tfidf.transform([text])
        pred = self.best_model.predict(X)[0]
        proba = self.best_model.predict_proba(X)[0]
        prob = proba[int(pred)]
        label = "ПОЗИТИВ" if pred == 1 else "НЕГАТИВ"
        return label, round(float(prob) * 100, 2)

    def get_top_products_by_positivity(self, top_n=5):
        stats = self.df[self.df['sentiment'].notna()].groupby('product').agg(
            total_reviews=('sentiment', 'count'),
            positive_count=('sentiment', 'sum')
        ).reset_index()
        stats['positive_share'] = (stats['positive_count'] / stats['total_reviews'] * 100).round(1)
        return stats.sort_values('positive_share', ascending=False).head(top_n)

    def get_top_negative_words(self, top_n=5):
        negative_texts = self.df[self.df['sentiment'] == 0]['text'].values
        text = ' '.join(negative_texts).lower()
        words = re.findall(r'\b[а-яё]{4,}\b', text)
        stop_words = {'этот', 'что', 'это', 'ещё', 'есть', 'все', 'очень', 'товар', 'деньги', 'свои', 'своих', 'себя', 'весь', 'мочь', 'быть', 'самый', 'такой'}
        words = [w for w in words if w not in stop_words]
        return Counter(words).most_common(top_n)

    def cluster_reviews(self, method='kmeans', n_clusters=4):
        X, y, _ = self.prepare_data()
        X_vec = self.vectorize_st(X) if self.vectorizer_st else self.vectorize_tfidf(X)

        if isinstance(X_vec, np.ndarray):
            pass
        else:
            X_vec = X_vec.toarray()

        if method == 'kmeans':
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        else:
            model = DBSCAN(eps=0.5, min_samples=3)

        clusters = model.fit_predict(X_vec)

        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X_vec)

        result_df = pd.DataFrame({
            'text': X,
            'cluster': clusters,
            'sentiment': y,
            'x': coords[:, 0],
            'y': coords[:, 1]
        })

        cluster_info = {}
        for c in sorted(set(clusters)):
            if c == -1:
                continue
            mask = result_df['cluster'] == c
            c_texts = result_df[mask]['text']
            c_sent = result_df[mask]['sentiment']
            all_words = ' '.join(c_texts.values).lower()
            words = re.findall(r'\b[а-яё]{4,}\b', all_words)
            top_words = Counter(words).most_common(10)
            pos_share = c_sent.mean() * 100 if len(c_sent) > 0 else 0
            cluster_info[str(c)] = {
                'size': int(mask.sum()),
                'top_words': [w for w, _ in top_words],
                'positive_share': round(pos_share, 1)
            }

        return result_df, cluster_info

    def get_roc_data(self):
        roc_data = {}
        X, y, _ = self.prepare_data()
        X_vec = self.vectorize_tfidf(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_vec, y, test_size=0.2, random_state=42, stratify=y
        )

        models = {
            'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
            'RandomForest': RandomForestClassifier(random_state=42, n_estimators=100),
            'XGBoost': XGBClassifier(random_state=42, verbosity=0, use_label_encoder=False),
            'MLPClassifier': MLPClassifier(max_iter=500, random_state=42),
            'CatBoost': CatBoostClassifier(random_seed=42, verbose=0)
        }

        for name, model in models.items():
            model.fit(X_train, y_train)
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                roc_data[name] = {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'auc': round(roc_auc, 4)
                }

        return roc_data

    def generate_report(self, save_path='student_report.png'):
        fig, axes = plt.subplots(3, 3, figsize=(18, 16))
        fig.suptitle('Отчёт студента — Анализ отзывов интернет-магазина', fontsize=16, fontweight='bold')

        axes[0, 0].hist(self.df['rating'], bins=5, edgecolor='black', color='royalblue', alpha=0.7, rwidth=0.8)
        axes[0, 0].set_title('Распределение оценок')
        axes[0, 0].set_xlabel('Оценка')
        axes[0, 0].set_ylabel('Количество')

        daily = self.df.groupby(self.df['date'].dt.date).size()
        if len(daily) > 0:
            daily.index = pd.to_datetime(daily.index)
            daily = daily.sort_index()
            axes[0, 1].plot(daily.index, daily.values, 'b-o', linewidth=1.5, markersize=3)
            axes[0, 1].set_title('Динамика отзывов по дням')
            axes[0, 1].tick_params(axis='x', rotation=45)

        top_prod = self.get_top_products_by_positivity()
        bars = axes[0, 2].barh(top_prod['product'], top_prod['positive_share'], color='seagreen', alpha=0.7)
        axes[0, 2].set_title('Топ товаров по доле позитива')
        axes[0, 2].set_xlabel('Доля позитивных отзывов (%)')

        ax = axes[1, 0]
        ax.axis('off')
        table_data = []
        col_labels = ['Модель', 'Accuracy', 'Precision', 'Recall', 'F1']
        for name, res in self.results_after.items():
            m = res['metrics']
            table_data.append([name, f"{m['accuracy']:.2%}", f"{m['precision']:.2%}", f"{m['recall']:.2%}", f"{m['f1']:.2%}"])
        table = ax.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(0.7, 1.4)
        ax.set_title('Сравнение моделей', fontsize=12, fontweight='bold')

        neg_words = self.get_top_negative_words()
        words, counts = zip(*neg_words) if neg_words else ([], [])
        axes[1, 1].barh(range(len(words)), counts, color='coral', alpha=0.7)
        axes[1, 1].set_yticks(range(len(words)))
        axes[1, 1].set_yticklabels(words)
        axes[1, 1].set_title('5 самых частых слов в негативе')
        axes[1, 1].invert_yaxis()

        prod_counts = self.df['product'].value_counts().head(10)
        axes[1, 2].barh(prod_counts.index, prod_counts.values, color='steelblue', alpha=0.7)
        axes[1, 2].set_title('Популярность товаров')
        axes[1, 2].set_xlabel('Количество отзывов')

        sent_counts = self.df[self.df['sentiment'].notna()]['sentiment'].value_counts()
        axes[2, 0].pie(
            [sent_counts.get(0, 0), sent_counts.get(1, 0)],
            labels=['Негатив', 'Позитив'],
            colors=['coral', 'lightgreen'],
            autopct='%1.1f%%',
            startangle=90
        )
        axes[2, 0].set_title('Соотношение тональности')

        axes[2, 1].axis('off')
        report_text = (
            f"Всего отзывов: {len(self.df)}\n"
            f"Средний рейтинг: {self.df['rating'].mean():.2f}\n"
            f"Товаров: {self.df['product'].nunique()}\n"
            f"Пользователей: {self.df['user_id'].nunique()}\n"
            f"Лучшая модель: {self.best_model_name}\n"
            f"  F1: {self.results_after[self.best_model_name]['metrics']['f1']:.2%}"
        )
        axes[2, 1].text(0.1, 0.5, report_text, fontsize=12, verticalalignment='center',
                        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        axes[2, 1].set_title('Сводка')

        axes[2, 2].axis('off')
        axes[2, 2].text(0.1, 0.3,
                        "Лучшие параметры:\n" +
                        "\n".join([f"{n}: {res['best_params']}" for n, res in self.results_after.items()]),
                        fontsize=8, verticalalignment='center',
                        bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
        axes[2, 2].set_title('Подобранные гиперпараметры')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Отчёт сохранён: {save_path}")
