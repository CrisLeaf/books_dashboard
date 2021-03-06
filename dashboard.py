import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from collections import Counter
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from data.get_stop_words import spanish_stop_words
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
import re


df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

df["book_year"].fillna(method="ffill", inplace=True)
df["book_pages"].fillna(method="ffill", inplace=True)
df["review_non_stopwords_rate"].fillna(method="ffill", inplace=True)
df["review_mean_word_length"].fillna(method="ffill", inplace=True)

def match_word_in_string(word, string):
	return 1 if word in string.split() else 0

editorials = ["planeta", "ediciones"]
categories = ["ficcion", "no-ficcion", "juvenil"]

for editorial in editorials:
	df["editorial_" + editorial] = df["editorial"].apply(
		lambda x: match_word_in_string(editorial, x))

for category in categories:
	df["category_" + category] = df["book_category"].apply(
		lambda x: match_word_in_string(category, x))

for editorial in editorials:
	test_df["editorial_" + editorial] = test_df["editorial"].apply(
		lambda x: match_word_in_string(editorial, x))

for category in categories:
	test_df["category_" + category] = test_df["book_category"].apply(
		lambda x: match_word_in_string(category, x))

random_indexes = np.random.randint(0, df.shape[0], 107)
na_inputs = df["book_pages"].iloc[random_indexes].values

na_indexes = test_df[test_df["book_pages"].isna()].index

for i, index in enumerate(na_indexes):
	test_df.at[index, "book_pages"] = na_inputs[i]

relevant_parameters = ["price", "book_pages", "editorial_planeta", "editorial_ediciones",
					   "category_ficcion", "category_no-ficcion", "category_juvenil"]

html_header = """
	<head>
	<link rel="stylesheet"href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css"integrity="sha512-Fo3rlrZj/k7ujTnHg4CGR2D7kSs0v4LLanw2qksYuRlEzO+tcaEPQogQ0KaoGN26/zrn20ImR1DfuLWnOo7aBA==" crossorigin="anonymous" referrerpolicy="no-referrer" />
	</head>
	<a href="https://crisleaf.herokuapp.com/">
		<i class="fas fa-arrow-left"></i>
	</a>
	<h2 style="text-align:center;">An??lisis Estad??stico de Libros</h2>
	<style>
		i {
			font-size: 30px;
			color: #222;
		}
		i:hover {
			color: cornflowerblue;
			transition: color 0.3s ease;
		}
	</style>
"""

page = st.sidebar.selectbox("Seleccione:", ["An??lisis: Uni-Variado",
											"An??lisis: Bi-Variado",
											"An??lisis: Multi-Variado",
											"Predecir Calificaciones"])

if page == "An??lisis: Uni-Variado":
	st.markdown(html_header, unsafe_allow_html=True)
	st.write(f"Seleccione uno de los par??metros relevantes de los datos, para conocer su "
			 f"distribuci??n")
	st.write(f"(Los datos fueron recolectados de www.buscalibre.cl)")
	
	parameter_list = ["...", "Nombre", "G??nero", "Editorial", "Precio", "Formato", "Categor??a",
					  "A??o de publicaci??n", "N?? de p??ginas", "Ranking",
					  "Contra-portada: n?? de caracteres",
					  "Contra-portada: largo promedio de cada palabra",
					  "Contra-portada: largo promedio de cada oraci??n",
					  "Contra-portada: variaci??n del largo de oraciones",
					  "Contra-portada: proporci??n de palabras importantes"]
	
	parameter = st.selectbox("Seleccione un par??metro:", parameter_list)
	
	if parameter == "Nombre":
		all_book_names = ""
		
		for row in df["name_cleaned"]:
			all_book_names += row + " "
		
		commons = Counter(all_book_names.split())
		commons_list = sorted(commons.items(), key=lambda x: x[1], reverse=True)
		
		fig = px.bar(
			x=[item[1] for item in commons_list[0:10]],
			y=[item[0] for item in commons_list[0:10]],
			labels={"x": "Cantidad", "y": "Palabra"},
			title="Palabras m??s repetidas en el Nombre"
		).update_layout(yaxis={"autorange": "reversed"})
		st.plotly_chart(fig)
	
	elif parameter == "G??nero":
		females_data = pd.concat([
			df[(df["female_author"] == 1) &
			   (df["male_author"] == 0)]["bayesian_rating"].copy(),
			pd.Series(index=df[(df["female_author"] == 1) &
							   (df["male_author"] == 0)].index,
					  data="Mujeres", name="g??nero")
		], axis=1)
		females_data.reset_index(drop=True, inplace=True)
		
		males_data = pd.concat([
			df[(df["female_author"] == 0) &
			   (df["male_author"] == 1)]["bayesian_rating"].copy(),
			pd.Series(index=df[(df["female_author"] == 0) &
							   (df["male_author"] == 1)].index,
					  data="Hombres", name="g??nero")
		], axis=1)
		males_data.reset_index(drop=True, inplace=True)
		
		nas_data = pd.concat([
			df[(df["female_author"] == 0) &
			   (df["male_author"] == 0)]["bayesian_rating"].copy(),
			pd.Series(index=df[(df["female_author"] == 0) &
							   (df["male_author"] == 0)].index,
					  data="No Especificado", name="g??nero")
		], axis=1)
		nas_data.reset_index(drop=True, inplace=True)
		
		both_data = pd.concat([
			df[(df["female_author"] == 1) &
			   (df["male_author"] == 1)]["bayesian_rating"].copy(),
			pd.Series(index=df[(df["female_author"] == 1) &
							   (df["male_author"] == 1)].index,
					  data="No Especificado", name="g??nero")
		], axis=1)
		both_data.reset_index(drop=True, inplace=True)
		
		plot_data = pd.concat([females_data, males_data, nas_data, both_data])
		
		del females_data, males_data, nas_data, both_data
		
		fig = px.histogram(
			plot_data,
			x="g??nero",
			labels={"g??nero": "G??nero"},
			title="Histograma del G??nero del Autor"
		).update_layout(yaxis_title="Cantidad")
		st.plotly_chart(fig)
	
	
	elif parameter == "Editorial":
		all_editorials = ""
		
		for row in df["editorial"]:
			all_editorials += row + " "
		
		commons = Counter(all_editorials.split())
		commons_list = sorted(commons.items(), key=lambda x: x[1], reverse=True)
		
		fig = px.bar(
			x=[item[1] for item in commons_list[0:10]],
			y=[item[0] for item in commons_list[0:10]],
			labels={"x": "Cantidad", "y": "Palabra"},
			title="Palabras m??s repetidas en la Editorial"
		).update_layout(yaxis={"autorange": "reversed"})
		st.plotly_chart(fig)
	
	elif parameter == "Categor??a":
		all_categories = ""
		
		for row in df["book_category"]:
			all_categories += row + " "
		
		commons = Counter(all_categories.split())
		commons_list = sorted(commons.items(), key=lambda x: x[1], reverse=True)
		
		fig = px.bar(
			x=[item[1] for item in commons_list[0:10]],
			y=[item[0] for item in commons_list[0:10]],
			labels={"x": "Cantidad", "y": "Categor??a"},
			title="Categor??as m??s repetidas"
		).update_layout(yaxis={"autorange": "reversed"})
		st.plotly_chart(fig)
	
	elif parameter == "Precio":
		conf_int_low = np.percentile(df["price"], 2.5)
		conf_int_high = np.percentile(df["price"], 97.5)
		
		fig = px.histogram(
			df[(df["price"] >= conf_int_low) &
			   (df["price"] <= conf_int_high)],
			x="price",
			nbins=30,
			labels={"price": "Precio"},
			title="Histograma de Precios"
		).update_layout(yaxis_title="Cantidad")
		st.plotly_chart(fig)
	
	elif parameter == "A??o de publicaci??n":
		conf_int_low = np.percentile(df["book_year"], 2.5)
		conf_int_high = np.percentile(df["book_year"], 97.5)
		
		fig = px.histogram(
			df[(df["book_year"] >= conf_int_low) &
			   (df["book_year"] <= conf_int_high)],
			x="book_year",
			nbins=30,
			labels={"book_year": "A??o"},
			title="Histograma del A??o de Publicaci??n"
		).update_layout(yaxis_title="Cantidad")
		st.plotly_chart(fig)
	
	elif parameter == "N?? de p??ginas":
		conf_int_low = np.percentile(df["book_pages"], 2.5)
		conf_int_high = np.percentile(df["book_pages"], 97.5)
		
		fig = px.histogram(
			df[(df["book_pages"] >= conf_int_low) &
			   (df["book_pages"] <= conf_int_high)],
			x="book_pages",
			nbins=30,
			labels={"book_pages": "N?? de p??ginas"},
			title="Histograma del n?? de P??ginas"
		).update_layout(yaxis_title="Cantidad")
		st.plotly_chart(fig)
	
	elif parameter == "Ranking":
		conf_int_low = np.percentile(df["bayesian_rating"], 2.5)
		conf_int_high = np.percentile(df["bayesian_rating"], 97.5)
		
		fig = px.histogram(
			df[(df["bayesian_rating"] >= conf_int_low) &
			   (df["bayesian_rating"] <= conf_int_high)],
			x="bayesian_rating",
			nbins=30,
			labels={"bayesian_rating": "Ranking"},
			title="Histograma del Ranking"
		).update_layout(yaxis_title="Cantidad")
		st.plotly_chart(fig)
	
	elif parameter == "Formato":
		fig = px.histogram(
			df,
			x="book_format",
			nbins=30,
			labels={"book_format": "Formato"},
			title="Histograma del Formato"
		).update_layout(yaxis_title="Cantidad")
		st.plotly_chart(fig)
	
	elif parameter == "Contra-portada: n?? de caracteres":
		conf_int_low = np.percentile(df["review_chars_count"], 2.5)
		conf_int_high = np.percentile(df["review_chars_count"], 97.5)
		
		fig = px.histogram(
			df[(df["review_chars_count"] >= conf_int_low) &
			   (df["review_chars_count"] <= conf_int_high)],
			x="review_chars_count",
			nbins=30,
			labels={"review_chars_count": "N?? de caracteres"},
			title="Histograma del n?? de Caracteres"
		).update_layout(yaxis_title="Cantidad")
		st.plotly_chart(fig)
	
	elif parameter == "Contra-portada: largo promedio de cada palabra":
		conf_int_low = np.percentile(df["review_mean_word_length"], 2.5)
		conf_int_high = np.percentile(df["review_mean_word_length"], 97.5)
		
		fig = px.histogram(
			df[(df["review_mean_word_length"] >= conf_int_low) &
			   (df["review_mean_word_length"] <= conf_int_high)],
			x="review_mean_word_length",
			nbins=30,
			labels={"review_mean_word_length": "Largo promedio"},
			title="Histograma del Largo Promedio de Cada Palabra"
		).update_layout(yaxis_title="Cantidad")
		st.plotly_chart(fig)
	
	elif parameter == "Contra-portada: largo promedio de cada oraci??n":
		conf_int_low = np.percentile(df["review_mean_sentence_length"], 2.5)
		conf_int_high = np.percentile(df["review_mean_sentence_length"], 97.5)
		
		fig = px.histogram(
			df[(df["review_mean_sentence_length"] >= conf_int_low) &
			   (df["review_mean_sentence_length"] <= conf_int_high)],
			x="review_mean_sentence_length",
			nbins=30,
			labels={"review_mean_sentence_length": "Largo promedio"},
			title="Histograma del Largo Promedio de Cada Oraci??n"
		).update_layout(yaxis_title="Cantidad")
		st.plotly_chart(fig)
	
	elif parameter == "Contra-portada: variaci??n del largo de oraciones":
		conf_int_low = np.percentile(df["review_std_sentence_length"], 2.5)
		conf_int_high = np.percentile(df["review_std_sentence_length"], 97.5)
		
		fig = px.histogram(
			df[(df["review_std_sentence_length"] > conf_int_low) &
			   (df["review_std_sentence_length"] <= conf_int_high)],
			x="review_std_sentence_length",
			nbins=30,
			labels={"review_std_sentence_length": "Variaci??n"},
			title="Histograma de la Variaci??n del Largo de las Oraciones"
		).update_layout(yaxis_title="Cantidad")
		st.plotly_chart(fig)
	
	elif parameter == "Contra-portada: proporci??n de palabras importantes":
		conf_int_low = np.percentile(df["review_non_stopwords_rate"], 2.5)
		conf_int_high = np.percentile(df["review_non_stopwords_rate"], 97.5)
		
		fig = px.histogram(
			df[(df["review_non_stopwords_rate"] > conf_int_low) &
			   (df["review_non_stopwords_rate"] <= conf_int_high)],
			x="review_non_stopwords_rate",
			nbins=30,
			labels={"review_non_stopwords_rate": "Proporci??n"},
			title="Histograma de la Proporci??n de Palabras Importantes"
		).update_layout(yaxis_title="Cantidad")
		st.plotly_chart(fig)

elif page == "An??lisis: Bi-Variado":
	st.markdown(html_header, unsafe_allow_html=True)
	st.write("Seleccione uno de los par??metros para ver como se relaciona con las calificaciones "
			 "(**Ranking**) que dejaron los compradores para cada libro.")
	st.write(f"(Los datos fueron recolectados de www.buscalibre.cl)")
	
	parameter_list = ["...", "Nombre", "G??nero", "Editorial", "Precio", "Categor??a",
					  "A??o de publicaci??n", "N?? de p??ginas",
					  "Contra-portada: n?? de caracteres",
					  "Contra-portada: largo promedio de cada palabra",
					  "Contra-portada: largo promedio de cada oraci??n",
					  "Contra-portada: variaci??n del largo de oraciones",
					  "Contra-portada: proporci??n de palabras importantes"]
	
	parameter = st.selectbox("Seleccione un par??metro:", parameter_list)
	
	sorted_df = df.sort_values(by="bayesian_rating", ascending=False)
	
	if parameter == "Nombre":
		top1000_book_names = ""
		
		for row in sorted_df[0:1000]["name_cleaned"]:
			top1000_book_names += row + " "
		
		commons = Counter(top1000_book_names.split())
		commons_list = sorted(commons.items(), key=lambda x: x[1], reverse=True)
		
		fig = px.bar(
			x=[item[1] for item in commons_list[0:10]],
			y=[item[0] for item in commons_list[0:10]],
			labels={"x": "Cantidad", "y": "Palabra"},
			title="Palabras m??s repetidas en el Nombre de los top 1.000 Libros"
		).update_layout(yaxis={"autorange": "reversed"})
		st.plotly_chart(fig)
	
	elif parameter == "G??nero":
		females_data = pd.concat([
			df[(df["female_author"] == 1) &
			   (df["male_author"] == 0)]["bayesian_rating"].copy(),
			pd.Series(index=df[(df["female_author"] == 1) &
							   (df["male_author"] == 0)].index,
					  data="Mujeres", name="g??nero")
		], axis=1)
		females_data.reset_index(drop=True, inplace=True)
		
		males_data = pd.concat([
			df[(df["female_author"] == 0) &
			   (df["male_author"] == 1)]["bayesian_rating"].copy(),
			pd.Series(index=df[(df["female_author"] == 0) &
							   (df["male_author"] == 1)].index,
					  data="Hombres", name="g??nero")
		], axis=1)
		males_data.reset_index(drop=True, inplace=True)
		
		nas_data = pd.concat([
			df[(df["female_author"] == 0) &
			   (df["male_author"] == 0)]["bayesian_rating"].copy(),
			pd.Series(index=df[(df["female_author"] == 0) &
							   (df["male_author"] == 0)].index,
					  data="No Especificado", name="g??nero")
		], axis=1)
		nas_data.reset_index(drop=True, inplace=True)
		
		both_data = pd.concat([
			df[(df["female_author"] == 1) &
			   (df["male_author"] == 1)]["bayesian_rating"].copy(),
			pd.Series(index=df[(df["female_author"] == 1) &
							   (df["male_author"] == 1)].index,
					  data="No Especificado", name="g??nero")
		], axis=1)
		both_data.reset_index(drop=True, inplace=True)
		
		plot_data = pd.concat([females_data, males_data, nas_data, both_data])
		
		del females_data, males_data, nas_data, both_data
		
		fig = px.strip(
			plot_data,
			x="g??nero",
			y="bayesian_rating",
			labels={"g??nero": "G??nero", "bayesian_rating": "Ranking"},
			title="Ranking vs G??nero"
		)
		st.plotly_chart(fig)
	
	
	elif parameter == "Editorial":
		top1000_editorials = ""
		
		for row in sorted_df[0:1000]["editorial"]:
			top1000_editorials += row + " "
		
		commons = Counter(top1000_editorials.split())
		commons_list = sorted(commons.items(), key=lambda x: x[1], reverse=True)
		
		fig = px.bar(
			x=[item[1] for item in commons_list[0:10]],
			y=[item[0] for item in commons_list[0:10]],
			labels={"x": "Cantidad", "y": "Palabra"},
			title="Palabras m??s repetidas en la Editorial de los top 1.000 Libros"
		).update_layout(yaxis={"autorange": "reversed"})
		st.plotly_chart(fig)
	
	elif parameter == "Categor??a":
		top1000_categories = ""
		
		for row in sorted_df[0:1000]["book_category"]:
			top1000_categories += row + " "
		
		commons = Counter(top1000_categories.split())
		commons_list = sorted(commons.items(), key=lambda x: x[1], reverse=True)
		
		fig = px.bar(
			x=[item[1] for item in commons_list[0:10]],
			y=[item[0] for item in commons_list[0:10]],
			labels={"x": "Cantidad", "y": "Categor??a"},
			title="Categor??as m??s repetidas en los top 1.000 Libros"
		).update_layout(yaxis={"autorange": "reversed"})
		st.plotly_chart(fig)
	
	elif parameter == "Precio":
		conf_int_low = np.percentile(df["price"], 2.5)
		conf_int_high = np.percentile(df["price"], 97.5)
		
		fig = px.scatter(
			df[(df["price"] >= conf_int_low) &
			   (df["price"] <= conf_int_high)],
			x="price",
			y="bayesian_rating",
			labels={"price": "Precio"},
			title="Ranking vs Precio"
		).update_layout(yaxis_title="Ranking")
		st.plotly_chart(fig)
	
	elif parameter == "A??o de publicaci??n":
		conf_int_low = np.percentile(df["book_year"], 2.5)
		conf_int_high = np.percentile(df["book_year"], 97.5)
		
		fig = px.scatter(
			df[(df["book_year"] >= conf_int_low) &
			   (df["book_year"] <= conf_int_high)],
			x="book_year",
			y="bayesian_rating",
			labels={"book_year": "A??o"},
			title="Ranking vs A??o de Publicaci??n"
		).update_layout(yaxis_title="Ranking")
		st.plotly_chart(fig)
	
	elif parameter == "N?? de p??ginas":
		conf_int_low = np.percentile(df["book_pages"], 2.5)
		conf_int_high = np.percentile(df["book_pages"], 97.5)
		
		fig = px.scatter(
			df[(df["book_pages"] >= conf_int_low) &
			   (df["book_pages"] <= conf_int_high)],
			x="book_pages",
			y="bayesian_rating",
			labels={"book_pages": "N?? de p??ginas"},
			title="Ranking vs N?? de P??ginas"
		).update_layout(yaxis_title="Ranking")
		st.plotly_chart(fig)
	
	elif parameter == "Contra-portada: n?? de caracteres":
		conf_int_low = np.percentile(df["review_chars_count"], 2.5)
		conf_int_high = np.percentile(df["review_chars_count"], 97.5)
		
		fig = px.scatter(
			df[(df["review_chars_count"] >= conf_int_low) &
			   (df["review_chars_count"] <= conf_int_high)],
			x="review_chars_count",
			y="bayesian_rating",
			labels={"review_chars_count": "N?? de caracteres"},
			title="Ranking vs n?? de Caracteres"
		).update_layout(yaxis_title="Ranking")
		st.plotly_chart(fig)
	
	elif parameter == "Contra-portada: largo promedio de cada palabra":
		conf_int_low = np.percentile(df["review_mean_word_length"], 2.5)
		conf_int_high = np.percentile(df["review_mean_word_length"], 97.5)
		
		fig = px.scatter(
			df[(df["review_mean_word_length"] >= conf_int_low) &
			   (df["review_mean_word_length"] <= conf_int_high)],
			x="review_mean_word_length",
			y="bayesian_rating",
			labels={"review_mean_word_length": "Largo promedio"},
			title="Ranking vs Largo Promedio de Cada Palabra"
		).update_layout(yaxis_title="Ranking")
		st.plotly_chart(fig)
	
	elif parameter == "Contra-portada: largo promedio de cada oraci??n":
		conf_int_low = np.percentile(df["review_mean_sentence_length"], 2.5)
		conf_int_high = np.percentile(df["review_mean_sentence_length"], 97.5)
		
		fig = px.scatter(
			df[(df["review_mean_sentence_length"] >= conf_int_low) &
			   (df["review_mean_sentence_length"] <= conf_int_high)],
			x="review_mean_sentence_length",
			y="bayesian_rating",
			labels={"review_mean_sentence_length": "Largo promedio"},
			title="Ranking vs Largo Promedio de Cada Oraci??n"
		).update_layout(yaxis_title="Ranking")
		st.plotly_chart(fig)
	
	elif parameter == "Contra-portada: variaci??n del largo de oraciones":
		conf_int_low = np.percentile(df["review_std_sentence_length"], 2.5)
		conf_int_high = np.percentile(df["review_std_sentence_length"], 97.5)
		
		fig = px.scatter(
			df[(df["review_std_sentence_length"] > conf_int_low) &
			   (df["review_std_sentence_length"] <= conf_int_high)],
			x="review_std_sentence_length",
			y="bayesian_rating",
			labels={"review_std_sentence_length": "Variaci??n"},
			title="Ranking vs Variaci??n del Largo de las Oraciones"
		).update_layout(yaxis_title="Ranking")
		st.plotly_chart(fig)
	
	elif parameter == "Contra-portada: proporci??n de palabras importantes":
		conf_int_low = np.percentile(df["review_non_stopwords_rate"], 2.5)
		conf_int_high = np.percentile(df["review_non_stopwords_rate"], 97.5)
		
		fig = px.scatter(
			df[(df["review_non_stopwords_rate"] > conf_int_low) &
			   (df["review_non_stopwords_rate"] <= conf_int_high)],
			x="review_non_stopwords_rate",
			y="bayesian_rating",
			labels={"review_non_stopwords_rate": "Proporci??n"},
			title="Ranking vs Proporci??n de Palabras Importantes"
		).update_layout(yaxis_title="Ranking")
		st.plotly_chart(fig)

elif page == "An??lisis: Multi-Variado":
	st.markdown(html_header, unsafe_allow_html=True)
	st.write("Seleccione un modelo de Machine Learning, para ver la importancia que tiene cada "
			 "par??metro al momento de dejar una calificaci??n.")
	st.write(f"Para cada uno, se utilizaron {df.shape[0]} libros en el entrenamiento y"
			 f" {test_df.shape[0]} en la validaci??n del modelo.")
	
	model_list = ["...", "Regresi??n Lineal", "Regresi??n Polinomial", "LightGBM", "Random Forest"]
	
	model = st.selectbox("Seleccione un modelo:", model_list)
	
	if model == "Regresi??n Lineal":
		X = df[relevant_parameters]
		y = df["bayesian_rating"]
		
		lin_reg = LinearRegression()
		lin_reg.fit(X, y)
		
		y_pred = lin_reg.predict(X)
		
		fig = px.scatter(
			x=y,
			y=y_pred,
			labels={"x": "Valores Reales", "y": "Valores Obtenidos"},
			title="Predicci??n de Regresi??n Lineal en Datos de Entrenamiento"
		)
		fig.add_shape(
			type="line",
			x0=y.min(),
			x1=y.max(),
			y0=y.min(),
			y1=y.max(),
			line={"color": "red"}
		)
		fig.add_shape(
			type="line",
			x0=y.min(),
			x1=y.max(),
			y0=y.mean(),
			y1=y.mean(),
			line={"color": "green"}
		)
		st.plotly_chart(fig)
		
		X_test = test_df[relevant_parameters]
		
		y_pred = lin_reg.predict(X_test)
		
		mean_test_error = mean_squared_error(test_df["bayesian_rating"],
											 [y.mean() for i in range(test_df.shape[0])])
		test_error = mean_squared_error(test_df["bayesian_rating"], y_pred)
		
		st.markdown("""
			<font color="red">La l??nea roja representa el valor esperado</font>,
			<font color="green">la l??nea verde representa el valor promedio</font> y <font
			color="blue">los puntos azules el valor obtenido</font>.
		""", unsafe_allow_html=True)
		st.write(f"Error en los datos de prueba: Valor medio: "
				 f"{round(mean_test_error, 6)} - Predicci??n: {round(test_error, 6)}")
		st.write(f"Representa una mejora del {1 - test_error / mean_test_error:.0%}, "
				 f"con respecto al valor promedio **en los datos de prueba**.")
		
		fig = px.bar(
			x=[item for item in lin_reg.coef_],
			y=[item for item in lin_reg.feature_names_in_],
			labels={"x": "Importancia", "y": "Par??metro"},
			title="Importancia de los Par??metros",
			height=500
		)
		st.plotly_chart(fig)
	
	elif model == "Regresi??n Polinomial":
		X = df[relevant_parameters]
		y = df["bayesian_rating"]
		
		pol_reg = Pipeline([("pol", PolynomialFeatures(degree=2)),
							("reg", LinearRegression())])
		pol_reg.fit(X, y)
		
		y_pred = pol_reg.predict(X)
		
		fig = px.scatter(
			x=y,
			y=y_pred,
			labels={"x": "Valores Reales", "y": "Valores Obtenidos"},
			title="Predicci??n de Regresi??n Polinomial en Datos de Entrenamiento"
		)
		fig.add_shape(
			type="line",
			x0=y.min(),
			x1=y.max(),
			y0=y.min(),
			y1=y.max(),
			line={"color": "red"}
		)
		fig.add_shape(
			type="line",
			x0=y.min(),
			x1=y.max(),
			y0=y.mean(),
			y1=y.mean(),
			line={"color": "green"}
		)
		st.plotly_chart(fig)
		
		X_test = test_df[relevant_parameters]
		
		y_pred = pol_reg.predict(X_test)
		
		mean_test_error = mean_squared_error(test_df["bayesian_rating"],
											 [y.mean() for i in range(test_df.shape[0])])
		test_error = mean_squared_error(test_df["bayesian_rating"], y_pred)
		
		st.markdown("""
			<font color="red">La l??nea roja representa el valor esperado</font>,
			<font color="green">la l??nea verde representa el valor promedio</font> y <font
			color="blue">los puntos azules el valor obtenido</font>.
		""", unsafe_allow_html=True)
		st.write(f"Error en los datos de prueba: Valor medio: "
				 f"{round(mean_test_error, 6)} - Predicci??n: {round(test_error, 6)}")
		st.write(f"Representa una mejora del {1 - test_error / mean_test_error:.0%}, "
				 f"con respecto al valor promedio **en los datos de prueba**.")
		
		fig = px.bar(
			x=[item for item in pol_reg.named_steps["reg"].coef_.flatten()],
			y=[item for item in pol_reg.named_steps["pol"].get_feature_names_out()],
			labels={"x": "Importancia", "y": "Par??metro"},
			title="Importancia de los Par??metros",
			height=1200
		)
		st.plotly_chart(fig)
	
	elif model == "LightGBM":
		X = df[relevant_parameters]
		y = df["bayesian_rating"]
		
		lgb_reg = LGBMRegressor(random_state=1919, n_estimators=50)
		lgb_reg.fit(X, y)
		
		y_pred = lgb_reg.predict(X)
		
		fig = px.scatter(
			x=y,
			y=y_pred,
			labels={"x": "Valores Reales", "y": "Valores Obtenidos"},
			title="Predicci??n de LightGBM en Datos de Entrenamiento"
		)
		fig.add_shape(
			type="line",
			x0=y.min(),
			x1=y.max(),
			y0=y.min(),
			y1=y.max(),
			line={"color": "red"}
		)
		fig.add_shape(
			type="line",
			x0=y.min(),
			x1=y.max(),
			y0=y.mean(),
			y1=y.mean(),
			line={"color": "green"}
		)
		st.plotly_chart(fig)
		
		X_test = test_df[relevant_parameters]
		
		y_pred = lgb_reg.predict(X_test)
		
		mean_test_error = mean_squared_error(test_df["bayesian_rating"],
											 [y.mean() for i in range(test_df.shape[0])])
		test_error = mean_squared_error(test_df["bayesian_rating"], y_pred)
		
		st.markdown("""
			<font color="red">La l??nea roja representa el valor esperado</font>,
			<font color="green">la l??nea verde representa el valor promedio</font> y <font
			color="blue">los puntos azules el valor obtenido</font>.
		""", unsafe_allow_html=True)
		st.write(f"Error en los datos de prueba: Valor medio: "
				 f"{round(mean_test_error, 6)} - Predicci??n: {round(test_error, 6)}")
		st.write(f"Representa una mejora del {1 - test_error / mean_test_error:.0%}, "
				 f"con respecto al valor promedio **en los datos de prueba**.")
		
		fig = px.bar(
			x=[item for item in lgb_reg.feature_importances_],
			y=[item for item in lgb_reg.feature_name_],
			labels={"x": "Importancia", "y": "Par??metro"},
			title="Importancia de los Par??metros",
			height=500
		)
		st.plotly_chart(fig)
	
	
	elif model == "Random Forest":
		X = df[relevant_parameters]
		y = df["bayesian_rating"]
		
		ran_for = RandomForestRegressor(random_state=1919)
		ran_for.fit(X, y)
		
		y_pred = ran_for.predict(X)
		
		fig = px.scatter(
			x=y,
			y=y_pred,
			labels={"x": "Valores Reales", "y": "Valores Obtenidos"},
			title="Predicci??n de Random Forest en Datos de Entrenamiento"
		)
		fig.add_shape(
			type="line",
			x0=y.min(),
			x1=y.max(),
			y0=y.min(),
			y1=y.max(),
			line={"color": "red"}
		)
		fig.add_shape(
			type="line",
			x0=y.min(),
			x1=y.max(),
			y0=y.mean(),
			y1=y.mean(),
			line={"color": "green"}
		)
		st.plotly_chart(fig)
		
		X_test = test_df[relevant_parameters]
		
		y_pred = ran_for.predict(X_test)
		
		mean_test_error = mean_squared_error(test_df["bayesian_rating"],
											 [y.mean() for i in range(test_df.shape[0])])
		test_error = mean_squared_error(test_df["bayesian_rating"], y_pred)
		
		st.markdown("""
			<font color="red">La l??nea roja representa el valor esperado</font>,
			<font color="green">la l??nea verde representa el valor promedio</font> y <font
			color="blue">los puntos azules el valor obtenido</font>.
		""", unsafe_allow_html=True)
		st.write(f"Error en los datos de prueba: Valor medio: "
				 f"{round(mean_test_error, 6)} - Predicci??n: {round(test_error, 6)}")
		st.write(f"Representa una mejora del {1 - test_error / mean_test_error:.0%}, "
				 f"con respecto al valor promedio **en los datos de prueba**.")
		
		fig = px.bar(
			x=[item for item in ran_for.feature_importances_],
			y=[item for item in ran_for.feature_names_in_],
			labels={"x": "Importancia", "y": "Par??metro"},
			title="Importancia de los Par??metros",
			height=500
		)
		st.plotly_chart(fig)

elif page == "Predecir Calificaciones":
	html_header = """
			<head>
			<link rel="stylesheet"href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css"integrity="sha512-Fo3rlrZj/k7ujTnHg4CGR2D7kSs0v4LLanw2qksYuRlEzO+tcaEPQogQ0KaoGN26/zrn20ImR1DfuLWnOo7aBA==" crossorigin="anonymous" referrerpolicy="no-referrer" />
			</head>
			<a href="https://crisleaf.herokuapp.com/">
				<i class="fas fa-arrow-left"></i>
			</a>
			<h2 style="text-align:center;">Predecir Calificaciones</h2>
			<style>
				i {
					font-size: 30px;
					color: #222;
				}
				i:hover {
					color: cornflowerblue;
					transition: color 0.3s ease;
				}
			</style>
		"""
	st.markdown(html_header, unsafe_allow_html=True)
	
	user_name = st.text_input("Ingrese nombre del libro:")
	user_price = st.number_input("Ingrese precio (en pesos chilenos):", min_value=1000,
								 max_value=20_000, step=1000)
	user_pages = st.number_input("Ingrese n?? de p??ginas", min_value=1,
								 max_value=500, step=10)
	user_planeta = st.radio("??Pertenece a la Editorial Planeta?", ["S??", "No"])
	user_ficcion = st.radio("??Entra en la categor??a ficci??n?", ["S??", "No"])
	user_juvenil = st.radio("??Es para j??venes?", ["S??", "No"])
	
	if user_planeta == "S??":
		user_planeta = 1
	else:
		user_planeta = 0
	
	if user_ficcion == "S??":
		user_ficcion = 1
		user_noficcion = 0
	else:
		user_ficcion = 0
		user_noficcion = 1
	
	if user_juvenil == "S??":
		user_juvenil = 1
	else:
		user_juvenil = 0
	
	full_df = pd.concat([df, test_df], axis=0, ignore_index=True)
	
	X = full_df[relevant_parameters]
	y = full_df["bayesian_rating"]
	
	lin_reg = LinearRegression()
	lin_reg.fit(X, y)
	
	user_book = pd.DataFrame(data={
		"price": user_price,
		"book_pages": user_pages,
		"editorial_planeta": user_planeta,
		"editorial_ediciones": 0,
		"category_ficcion": user_ficcion,
		"category_no-ficcion": user_noficcion,
		"cateogry_juvenil": user_juvenil
	}, index=[0])
	
	predict_btn = st.button("Predecir calificaci??n")
	
	if predict_btn:
		y_pred = lin_reg.predict(user_book)[0]
		
		full_df["rating_diff"] = full_df["bayesian_rating"].apply(lambda x: abs(y_pred - x))
		similar = full_df.sort_values(by="rating_diff").iloc[np.random.randint(0, 5, 1)[0]][[
			"five_stars", "four_stars", "three_stars", "two_stars", "one_star", "link"
		]]
		
		extra_row = {
			"name": user_name,
			"five_stars": "0",
			"four_stars": "0",
			"three_stars": "0",
			"two_stars": "0",
			"one_star": "0",
			"link": "a"
		}
		extra_df = pd.DataFrame(extra_row, index=[0])
		extra_df = pd.concat([full_df[["name", "five_stars", "four_stars", "three_stars",
									   "two_stars", "one_star", "link"]],
							  extra_df],
							 ignore_index=True)
		
		name_vect = CountVectorizer(stop_words=spanish_stop_words)
		name_count = name_vect.fit_transform(extra_df["name"])
		name_matrix = linear_kernel(name_count, name_count, dense_output=False)
		
		scores = list(enumerate(name_matrix.toarray()[extra_df.shape[0] - 1]))[:-1]
		
		scores = sorted(scores, key=lambda x: x[1], reverse=True)
		max_score_index = scores[0]
		
		similar_name = extra_df.iloc[max_score_index[0]]["name"]
		
		weight1 = 0.2
		weight2 = 1.8
		
		pred_five1 = int(re.search(r"\((.*?)\)", similar["five_stars"]).group(1))
		pred_five2 = int(re.search(r"\((.*?)\)",
								   extra_df.iloc[max_score_index[0]]["five_stars"]).group(1))
		pred_five = int((weight1 * pred_five1 + weight2 * pred_five2) // 2)
		
		pred_four1 = int(re.search(r"\((.*?)\)", similar["four_stars"]).group(1))
		pred_four2 = int(re.search(r"\((.*?)\)",
								   extra_df.iloc[max_score_index[0]]["four_stars"]).group(1))
		pred_four = int((weight1 * pred_four1 + weight2 * pred_four2) // 2)
		
		pred_three1 = int(re.search(r"\((.*?)\)", similar["three_stars"]).group(1))
		pred_three2 = int(re.search(r"\((.*?)\)",
									extra_df.iloc[max_score_index[0]]["three_stars"]).group(1))
		pred_three = int((weight1 * pred_three1 + weight2 * pred_three2) // 2)
		
		pred_two1 = int(re.search(r"\((.*?)\)", similar["two_stars"]).group(1))
		pred_two2 = int(re.search(r"\((.*?)\)",
								  extra_df.iloc[max_score_index[0]]["two_stars"]).group(1))
		pred_two = int((weight1 * pred_two1 + weight2 * pred_two2) // 2)
		
		pred_one1 = int(re.search(r"\((.*?)\)", similar["one_star"]).group(1))
		pred_one2 = int(re.search(r"\((.*?)\)",
								  extra_df.iloc[max_score_index[0]]["one_star"]).group(1))
		pred_one = int((weight1 * pred_one1 + weight2 * pred_one2) // 2)
		
		pred_total = pred_five + pred_four + pred_three + pred_two + pred_one
		
		st.markdown(f":full_moon::full_moon::full_moon::full_moon::full_moon: "
					f"({pred_five / pred_total:.0%}) {pred_five}",
					unsafe_allow_html=True)
		st.markdown(f":full_moon::full_moon::full_moon::new_moon::new_moon: "
					f"({pred_four / pred_total:.0%}) {pred_four}",
					unsafe_allow_html=True)
		st.markdown(f":full_moon::full_moon::new_moon::new_moon::new_moon: "
					f"({pred_three / pred_total:.0%}) {pred_three}",
					unsafe_allow_html=True)
		st.markdown(f":full_moon::new_moon::new_moon::new_moon::new_moon: "
					f"({pred_two / pred_total:.0%}) {pred_two}",
					unsafe_allow_html=True)
		st.markdown(f":new_moon::new_moon::new_moon::new_moon::new_moon: "
					f"({pred_one / pred_total:.0%}) {pred_one}",
					unsafe_allow_html=True)
#
# elif page == "Recomi??ndame un Libro":
# 	html_header = """
# 		<head>
# 		<link rel="stylesheet"href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css"integrity="sha512-Fo3rlrZj/k7ujTnHg4CGR2D7kSs0v4LLanw2qksYuRlEzO+tcaEPQogQ0KaoGN26/zrn20ImR1DfuLWnOo7aBA==" crossorigin="anonymous" referrerpolicy="no-referrer" />
# 		</head>
# 		<a href="https://crisleaf.herokuapp.com/">
# 			<i class="fas fa-arrow-left"></i>
# 		</a>
# 		<h2 style="text-align:center;">Bot Recomendador de Libros</h2>
# 		<style>
# 			i {
# 				font-size: 30px;
# 				color: #222;
# 			}
# 			i:hover {
# 				color: cornflowerblue;
# 				transition: color 0.3s ease;
# 			}
# 		</style>
# 	"""
# 	st.markdown(html_header, unsafe_allow_html=True)
# 	user_review = st.text_input("Ingrese el nombre de su libro favorito, y el bot intentar?? "
# 								"recomendarle uno de similares caracter??sticas.")
#
# 	st.button("Recomendar")
#
# 	if user_review != "":
# 		rec_bot = RecommendationBot()
#
# 		recommendation = rec_bot.recommend(user_review)
#
# 		col1, col2 = st.columns(2)
#
# 		url = recommendation["link"]
# 		r = requests.get(url)
# 		html = r.text
# 		soup = BeautifulSoup(html, "lxml")
# 		img_link = soup.find_all("meta", {"property": "og:image"})
#
# 		col1.image(img_link[0]["content"])
#
# 		col2.write("Nombre:")
# 		col2.write(f"{recommendation['name'].capitalize()}.")
# 		col2.write("Descripci??n:")
# 		col2.write(f"{recommendation['review'].capitalize()}.")
#
# 		html_name_link = f"""
# 			<a href="{recommendation['link']}" target="_blank">
# 			    Visitar P??gina
# 			</a>
# 		"""
# 		col2.markdown(html_name_link, unsafe_allow_html=True)

#
html_source_code = """
	<p class="source-code">C??digo Fuente:
	<a href="https://github.com/CrisLeaf/books_dashboard" target="_blank">
	<i class="fab fa-github"></i></a></p>
	<style>
		.source-code {
			text-align: right;
			color: #666;
		}
		.fa-github {
			color: #666;
		}
	</style>
"""
st.markdown(html_source_code, unsafe_allow_html=True)
