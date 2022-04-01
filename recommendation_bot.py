import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from scipy.sparse import csr_matrix
from data.get_stop_words import spanish_stop_words


class RecommendationBot():
	
	def __init__(self):
		cols = ["name", "review_raw", "link", "bayesian_rating", "book_category", "review"]
		train_df = pd.read_csv("data/train.csv")[cols]
		test_df = pd.read_csv("data/test.csv")[cols]
		df = pd.concat([train_df, test_df], axis=0)
		df.dropna(axis=0, subset=["review_raw"], inplace=True)
		
		self.df = df
	
	def clean_string(self, string):
		output_string = ""
		
		for character in string.lower():
			if character == "á":
				output_string += "a"
			elif character == "é":
				output_string += "e"
			elif character == "í":
				output_string += "i"
			elif character == "ó":
				output_string += "o"
			elif character == "ú":
				output_string += "u"
			elif character == "ñ":
				output_string += "n"
			elif character in "abcdefghijklmnopqrstuvwxyz0123456789 ":
				output_string += character
		
		return output_string
	
	def fit(self, temp_df):
		review_vect = TfidfVectorizer(stop_words=spanish_stop_words)
		review_tfidf = review_vect.fit_transform(temp_df["review_raw"])
		review_matrix = linear_kernel(review_tfidf, review_tfidf, dense_output=False)
		
		rating_matrix = np.tile(temp_df["bayesian_rating"] / max(temp_df["bayesian_rating"]),
								(temp_df.shape[0], 1))
		rating_matrix = csr_matrix(rating_matrix)
		
		category_vect = CountVectorizer(stop_words=spanish_stop_words)
		category_count = category_vect.fit_transform(temp_df["book_category"])
		category_matrix = linear_kernel(category_count, category_count, dense_output=False)
		
		self.similarity = 0.7 * review_matrix + 0.1 * rating_matrix + 0.2 * category_matrix
		
		return self
	
	def recommend(self, review):
		random_indexes = np.random.randint(0, self.df.shape[0], 5_000)
		temp_df = self.df.iloc[random_indexes]
		temp_df.reset_index(drop=True, inplace=True)
		
		raw_input = self.clean_string(review)
		extra_row = {
			"name": "extra",
			"review_raw": raw_input,
			"link": "null",
			"bayesian_rating": 0,
			"book_category": raw_input
		}
		extra_df = pd.DataFrame(extra_row, index=[0])
		temp_df = pd.concat([temp_df, extra_df], ignore_index=True)
		
		self.fit(temp_df)
		
		scores = list(enumerate(self.similarity.toarray()[temp_df.shape[0] - 1]))
		
		scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:]
		max_score_index = max(scores, key=lambda x: x[1])
		
		return {
			"review": temp_df["review"].iloc[max_score_index[0]],
			"name": temp_df["name"].iloc[max_score_index[0]],
			"link": temp_df["link"].iloc[max_score_index[0]],
			"score": max_score_index[1],
			"review_raw": raw_input,
			"scores": scores
		}
