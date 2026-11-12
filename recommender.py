import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class Recommender:
    def __init__(self, df):
        self.df = df
        self.user_item_matrix = None
        self.similarity_matrix = None
        self.users = None
        self.products = None
        self._build()

    def _build(self):
        ratings = self.df.groupby(['user_id', 'product'])['rating'].mean().reset_index()
        self.users = ratings['user_id'].unique()
        self.products = ratings['product'].unique()
        self.user_item_matrix = ratings.pivot(
            index='user_id', columns='product', values='rating'
        ).fillna(0)
        self.similarity_matrix = cosine_similarity(self.user_item_matrix)
        self.similarity_df = pd.DataFrame(
            self.similarity_matrix,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index
        )

    def recommend_for_user(self, user_id, top_n=5):
        if user_id not in self.similarity_df.index:
            return self.get_popular_products(top_n)

        similar_users = self.similarity_df[user_id].sort_values(ascending=False)
        similar_users = similar_users[similar_users.index != user_id]

        user_ratings = self.user_item_matrix.loc[user_id]
        already_rated = user_ratings[user_ratings > 0].index

        scores = {}
        for sim_user, sim_score in similar_users.items():
            if sim_score <= 0:
                continue
            sim_user_ratings = self.user_item_matrix.loc[sim_user]
            for product in sim_user_ratings[sim_user_ratings > 0].index:
                if product in already_rated:
                    continue
                if product not in scores:
                    scores[product] = 0
                scores[product] += sim_score * sim_user_ratings[product]

        if not scores:
            return self.get_popular_products(top_n)

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [product for product, _ in sorted_scores[:top_n]]

    def predict_rating(self, user_id, product):
        if user_id not in self.similarity_df.index or product not in self.user_item_matrix.columns:
            return self.df['rating'].mean()

        similar_users = self.similarity_df[user_id].sort_values(ascending=False)
        similar_users = similar_users[similar_users.index != user_id]

        ratings_sum = 0
        sim_sum = 0
        for sim_user, sim_score in similar_users.items():
            if sim_score <= 0:
                continue
            rating = self.user_item_matrix.loc[sim_user, product]
            if rating > 0:
                ratings_sum += sim_score * rating
                sim_sum += abs(sim_score)

        if sim_sum == 0:
            return self.df['rating'].mean()

        return ratings_sum / sim_sum

    def get_popular_products(self, top_n=5):
        popular = self.df['product'].value_counts().head(top_n).index.tolist()
        return popular

    def get_top_for_user(self, user_id, top_n=5):
        return self.recommend_for_user(user_id, top_n)
