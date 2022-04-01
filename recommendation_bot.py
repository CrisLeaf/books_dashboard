import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
# from scipy.sparse import csr_matrix
from data.get_stop_words import spanish_stop_words


class RecommendationBot():
	
	def __init__(self):
		train_df = pd.read_csv("data/train.csv")
		test_df = pd.read_csv("data/test.csv")
		self.df = pd.concat([train_df, test_df], axis=0)
	
	def fit(self):
		tfidf_vect_review_col = TfidfVectorizer(stop_words=spanish_stop_words)
		tfidf_mat_review_col = tfidf_vect_review_col.fit_transform(self.df["review"])
		
		sim_mat_review_col = linear_kernel(tfidf_mat_review_col, tfidf_mat_review_col,
										   dense_output=False)
		# sim_mat_review_col = sim_mat_review_col / np.max(sim_mat_review_col) * 20
		# sim_mat_review_col = sim_mat_review_col.astype("uint8")
		
		self.similarity = sim_mat_review_col
		
		return self
	
	def recommend(self, review):
		extra_row = {
			"name": "extra",
			"review": review,
			"link": "null"
		}
		extra_df = pd.DataFrame(extra_row, index=[0])
		self.df = pd.concat([self.df, extra_df], ignore_index=True)
		
		self.fit()
		
		scores = list(enumerate(self.similarity.toarray()[self.df.shape[0] - 1]))
		
		scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:]
		max_score_index = max(scores, key=lambda x: x[1])
		
		return {
			"name": self.df["name"].iloc[max_score_index[0]],
			"link": self.df["link"].iloc[max_score_index[0]],
			"score": max_score_index[1],
		}

# return self.base_df["link"].iloc[max_score_index[0]], max_score_index[1]
