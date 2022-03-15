import psycopg2
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
# from scipy.sparse import csr_matrix
from .get_stop_words import spanish_stop_words


class RecommendationsBot():
	
	def __init__(self, psql_params):
		self.psql_params = psql_params
	
	def get_data_from_psql(self):
		try:
			conn = psycopg2.connect(**self.psql_params)
			curr = conn.cursor()
			
			curr.execute("SELECT name, review, link FROM books WHERE id <= 5000")
			tuples = curr.fetchall()
			
			self.base_df = pd.DataFrame(tuples, columns=["name", "review", "link"])
			self.indexes = pd.Series(self.base_df.index, index=self.base_df["link"])
			
			curr.close()
			conn.close()
		
		except Exception as e:
			print(f"No se pudo cargar los datos: {e}")
		
		return self
	
	def get_similarity(self, df):
		tfidf_vect_review_col = TfidfVectorizer(stop_words=spanish_stop_words)
		tfidf_mat_review_col = tfidf_vect_review_col.fit_transform(df["review"])
		
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
		df = pd.concat([self.base_df, extra_df], ignore_index=True)
		
		self.get_similarity(df)
		
		scores = list(enumerate(self.similarity.toarray()[df.shape[0] - 1]))
		scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:]
		max_score_index = max(scores, key=lambda x: x[1])
		
		return {
			"name": self.base_df["name"].iloc[max_score_index[0]],
			"link": self.base_df["link"].iloc[max_score_index[0]],
			"score": max_score_index[1],
		}
	
	# return self.base_df["link"].iloc[max_score_index[0]], max_score_index[1]
