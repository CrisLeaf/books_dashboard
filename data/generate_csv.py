import numpy as np
import psycopg2
import pandas as pd
import spacy
from get_stop_words import spanish_stop_words
import re


def database_to_dataframe(psql_params):
	conn = psycopg2.connect(**psql_params)
	curr = conn.cursor()
	
	curr.execute(
		"SELECT name, author, editorial, price, review, description, five_stars, four_stars, "
		"three_stars, two_stars, one_star, link, website "
		"FROM books WHERE id <= 15000"
	)
	tuples = curr.fetchall()
	
	df = pd.DataFrame(tuples, columns=["name", "author", "editorial", "price", "review",
									   "description", "five_stars", "four_stars", "three_stars",
									   "two_stars", "one_star", "link", "website"])
	
	curr.close()
	conn.close()
	
	return df

# Extraction Functions
def extract_book_format(string):
	if "fisico" in string.split():
		book_format = "fisico"
	elif "formatoebook" in string.split():
		book_format = "ebook"
	else:
		book_format = "<Unknown>"
	
	return book_format

def extract_book_category(string):
	splited_string = string.split()
	
	try:
		book_category = splited_string[splited_string.index("categoria") + 1]
		
		if "," in book_category:
			book_category += " " + splited_string[splited_string.index("categoria") + 2]
			
			if "," in splited_string[splited_string.index("categoria") + 2]:
				book_category += " " + splited_string[splited_string.index("categoria") + 3]
	except:
		book_category = "<Unknown>"
	
	if book_category == "no":
		book_category = "no-ficcion"
	
	return book_category

def extract_book_year(string):
	splited_string = string.split()
	
	try:
		book_year = splited_string[splited_string.index("ano") + 1]
		if book_year == "201726":
			book_year_int = 2017
		elif book_year == "20021":
			book_year_int = 2021
		elif book_year == "1a(1962)":
			book_year_int = 1962
		elif book_year == "3":
			book_year_int = np.nan
		elif book_year == "20210":
			book_year_int = 2021
		else:
			book_year_int = int(book_year)
	except:
		book_year_int = np.nan
	
	return book_year_int

def extract_book_pages(string):
	splited_string = string.split()
	
	try:
		book_pages = splited_string[splited_string.index("paginas") + 1]
		book_pages_int = int(book_pages)
	except:
		book_pages_int = np.nan
	
	return book_pages_int

def raw_string(string):
	for character in string:
		if character not in "abcdefghijklmnopqrstuvwxyz0123456789 ":
			string = string.replace(character, "")
	
	return string

def clean_string(nlp, string):
	try:
		string_raw = raw_string(string)
		document = nlp(string_raw)
		output = " ".join([word.lemma_ for word in document
						   if word.lemma_ not in spanish_stop_words])
		
		if len(output) == 0:
			output = string
	
	except:
		output = string
	
	return output

def calculate_bayesian_rating(rating, mean_rating, raters, max_raters):
	return rating * (raters / max_raters) + (1 - (raters / max_raters)) * mean_rating

def get_punctuations(column):
	punctuations = []
	
	for row in column:
		for character in row:
			if character not in "abcdefghijklmnopqrstuvwxyz ":
				punctuations.append(character)
	
	return list(set(punctuations))

def get_genres_names():
	female_names_raw = pd.read_csv("aux_names/females.csv")["nombre"]
	male_names_raw = pd.read_csv("aux_names/males.csv")["nombre"]
	
	female_names_raw.dropna(inplace=True)
	male_names_raw.dropna(inplace=True)
	
	female_names = [name.lower().split()[0] for name in female_names_raw]
	male_names = [name.lower().split()[0] for name in male_names_raw]
	
	repeated_males = ["julia", "angeles", "dominique", "maria", "andrea", "karen",
					  "rosario", "iris", "loreto", "consuelo", "carol", "karin", "vivian", "denis",
					  "ariel", "pilar", "kim", "lilian", "surya", "leonor", "montserrat",
					  "guadalupe",
					  "ashley", "camille", "mercedes", "trinidad", "harriet", "paz", "jane",
					  "simone"]
	repeated_females = ["robin", "joan", "juan", "nino", "chris", "alex", "erin", "gabriele",
						"willy", "jesus", "hermogenes", "noah"]
	
	for name in repeated_females:
		female_names.remove(name)
	
	for name in repeated_males:
		male_names.remove(name)
	
	return female_names, male_names

def match_name(string, names):
	if string.split()[0] in names:
		match = 1
	else:
		match = 0
	
	return match

# Generation Functions
def generate_columns_from_description(df):
	df["description"] = df["description"].apply(lambda x: " ".join(x.replace(",", ", ").split()))
	df["book_format"] = df["description"].apply(lambda x: extract_book_format(x))
	df["book_category"] = df["description"].apply(lambda x: extract_book_category(x))
	df["book_year"] = df["description"].apply(lambda x: extract_book_year(x))
	df["book_pages"] = df["description"].apply(lambda x: extract_book_pages(x))

def generate_name_cleaned(df):
	nlp = spacy.load("es_core_news_sm")
	df["name_cleaned"] = df["name"].apply(lambda x: clean_string(nlp, x))

def generate_bayesian_rating(df):
	rating = []
	raters = []
	
	for index in range(df.shape[0]):
		star5 = int(re.search(r"\((.*?)\)", df["five_stars"].iloc[index]).group(1))
		star4 = int(re.search(r"\((.*?)\)", df["four_stars"].iloc[index]).group(1))
		star3 = int(re.search(r"\((.*?)\)", df["three_stars"].iloc[index]).group(1))
		star2 = int(re.search(r"\((.*?)\)", df["two_stars"].iloc[index]).group(1))
		star1 = int(re.search(r"\((.*?)\)", df["one_star"].iloc[index]).group(1))
		
		numerator = 5 * star5 + 4 * star4 + 3 * star3 + 2 * star2 + star1
		denominator = star5 + star4 + star3 + star2 + star1
		
		if denominator == 0:
			rating.append(0)
		else:
			rating.append(numerator / denominator)
		
		raters.append(denominator)
	
	mean_rating = sum(rating) / len(rating)
	max_raters = max(raters)
	
	bayesian_rating = [calculate_bayesian_rating(rating[i], mean_rating, raters[i], max_raters)
					   for i in range(df.shape[0])]
	
	df.insert(df.shape[1], "bayesian_rating", bayesian_rating)

def generate_columns_from_review(df):
	df["review_raw"] = df["review"].apply(lambda x: raw_string(x))
	
	df["review_chars_count"] = df["review"].apply(lambda x: len(x))
	df["review_words_count"] = df["review_raw"].apply(lambda x: len(x.split()))
	df["review_uniques_count"] = df["review_raw"].apply(lambda x: len(set(x.split())))
	df["review_stopwords_count"] = df["review_raw"].apply(
		lambda x: len([word for word in x.split() if word in spanish_stop_words])
	)
	df["review_non_stopwords_rate"] = (df["review_stopwords_count"] / df["review_words_count"])
	df["review_mean_word_length"] = df["review_raw"].apply(
		lambda x: np.mean([len(word) for word in x.split()])
	)
	df["review_std_word_length"] = df["review_raw"].apply(
		lambda x: np.std([len(word) for word in x.split()])
	)
	punctuations = get_punctuations(df["review"])
	df["review_punctuations_count"] = df["review"].apply(
		lambda x: len([char for char in x if char in punctuations])
	)
	
	df["review_sentence_count"] = df["review"].apply(
		lambda x: len([sent for sent in x.split(".") if len(sent) != 0])
	)
	df["review_mean_sentence_length"] = df["review"].apply(
		lambda x: np.mean([len(sent) for sent in x.split(".") if len(sent) != 0])
	)
	df["review_std_sentence_length"] = df["review"].apply(
		lambda x: np.std([len(sent) for sent in x.split(".") if len(sent) != 0])
	)
	df["review_max_sentence_length"] = df["review"].apply(
		lambda x: max([len(sent) for sent in x.split(".")])
	)

def generate_genres_columns(df):
	female_names, male_names = get_genres_names()
	
	df["female_author"] = df["author"].apply(lambda x: match_name(x, female_names))
	df["male_author"] = df["author"].apply(lambda x: match_name(x, male_names))

if __name__ == "__main__":
	from web_app.psql_secrets import psql_params
	import warnings
	
	
	warnings.filterwarnings("ignore")
	
	df = database_to_dataframe(psql_params)
	
	generate_columns_from_description(df)
	generate_name_cleaned(df)
	generate_bayesian_rating(df)
	generate_columns_from_review(df)
	generate_genres_columns(df)
	
	df = df.sample(frac=1, random_state=1234, ignore_index=True)
	
	print(df.info())
	
	train = df[:int(df.shape[0] * 0.9)]
	test = df[int(df.shape[0] * 0.9):]
	
	print("Shapes:")
	print(f"  Train shape: {train.shape}")
	print(f"  Test shape: {test.shape}")
	
	train.to_csv("train.csv", index=False)
	test.to_csv("test.csv", index=False)
