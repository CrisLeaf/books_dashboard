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
from sklearn.model_selection import cross_val_score


df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

df["book_year"].fillna(method="ffill", inplace=True)
df["book_pages"].fillna(method="ffill", inplace=True)
df["review_non_stopwords_rate"].fillna(method="ffill", inplace=True)
df["review_mean_word_length"].fillna(method="ffill", inplace=True)

html_header = """
	<head>
	<link rel="stylesheet"href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css"integrity="sha512-Fo3rlrZj/k7ujTnHg4CGR2D7kSs0v4LLanw2qksYuRlEzO+tcaEPQogQ0KaoGN26/zrn20ImR1DfuLWnOo7aBA==" crossorigin="anonymous" referrerpolicy="no-referrer" />
	</head>
	<a href="https://crisleaf.herokuapp.com/">
		<i class="fas fa-arrow-left"></i>
	</a>
	<h2 style="text-align:center;">Análisis Estadístico de Libros</h2>
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
st.write("Los datos fueron recolectados de www.buscalibre.cl.")

page = st.sidebar.selectbox("Seleccione Tipo de Análisis", ["Uni-Variado",
															"Bi-Variado",
															"Multi-Variado"])

if page == "Uni-Variado":
	parameter_list = ["...", "Nombre", "Género", "Editorial", "Precio", "Formato", "Categoría",
					  "Año de publicación", "N° de páginas", "Ranking",
					  "Contra-portada: n° de caracteres",
					  "Contra-portada: largo promedio de cada palabra",
					  "Contra-portada: largo promedio de cada oración",
					  "Contra-portada: variación del largo de oraciones",
					  "Contra-portada: proporción de palabras importantes"]
	
	parameter = st.selectbox("Seleccione un parámetro:", parameter_list)
	
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
			title="Palabras más repetidas en el Nombre"
		).update_layout(yaxis={"autorange": "reversed"})
		st.plotly_chart(fig)
	
	elif parameter == "Género":
		females_data = pd.concat([
			df[(df["female_author"] == 1) &
			   (df["male_author"] == 0)]["bayesian_rating"].copy(),
			pd.Series(index=df[(df["female_author"] == 1) &
							   (df["male_author"] == 0)].index,
					  data="Mujeres", name="género")
		], axis=1)
		females_data.reset_index(drop=True, inplace=True)
		
		males_data = pd.concat([
			df[(df["female_author"] == 0) &
			   (df["male_author"] == 1)]["bayesian_rating"].copy(),
			pd.Series(index=df[(df["female_author"] == 0) &
							   (df["male_author"] == 1)].index,
					  data="Hombres", name="género")
		], axis=1)
		males_data.reset_index(drop=True, inplace=True)
		
		nas_data = pd.concat([
			df[(df["female_author"] == 0) &
			   (df["male_author"] == 0)]["bayesian_rating"].copy(),
			pd.Series(index=df[(df["female_author"] == 0) &
							   (df["male_author"] == 0)].index,
					  data="No Especificado", name="género")
		], axis=1)
		nas_data.reset_index(drop=True, inplace=True)
		
		both_data = pd.concat([
			df[(df["female_author"] == 1) &
			   (df["male_author"] == 1)]["bayesian_rating"].copy(),
			pd.Series(index=df[(df["female_author"] == 1) &
							   (df["male_author"] == 1)].index,
					  data="No Especificado", name="género")
		], axis=1)
		both_data.reset_index(drop=True, inplace=True)
		
		plot_data = pd.concat([females_data, males_data, nas_data, both_data])
		
		del females_data, males_data, nas_data, both_data
		
		fig = px.histogram(
			plot_data,
			x="género",
			labels={"género": "Género"},
			title="Histograma del Género del Autor"
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
			title="Palabras más repetidas en la Editorial"
		).update_layout(yaxis={"autorange": "reversed"})
		st.plotly_chart(fig)
	
	elif parameter == "Categoría":
		all_categories = ""
		
		for row in df["book_category"]:
			all_categories += row + " "
		
		commons = Counter(all_categories.split())
		commons_list = sorted(commons.items(), key=lambda x: x[1], reverse=True)
		
		fig = px.bar(
			x=[item[1] for item in commons_list[0:10]],
			y=[item[0] for item in commons_list[0:10]],
			labels={"x": "Cantidad", "y": "Categoría"},
			title="Categorías más repetidas"
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
	
	elif parameter == "Año de publicación":
		conf_int_low = np.percentile(df["book_year"], 2.5)
		conf_int_high = np.percentile(df["book_year"], 97.5)
		
		fig = px.histogram(
			df[(df["book_year"] >= conf_int_low) &
			   (df["book_year"] <= conf_int_high)],
			x="book_year",
			nbins=30,
			labels={"book_year": "Año"},
			title="Histograma del Año de Publicación"
		).update_layout(yaxis_title="Cantidad")
		st.plotly_chart(fig)
	
	elif parameter == "N° de páginas":
		conf_int_low = np.percentile(df["book_pages"], 2.5)
		conf_int_high = np.percentile(df["book_pages"], 97.5)
		
		fig = px.histogram(
			df[(df["book_pages"] >= conf_int_low) &
			   (df["book_pages"] <= conf_int_high)],
			x="book_pages",
			nbins=30,
			labels={"book_pages": "N° de páginas"},
			title="Histograma del n° de Páginas"
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
	
	elif parameter == "Contra-portada: n° de caracteres":
		conf_int_low = np.percentile(df["review_chars_count"], 2.5)
		conf_int_high = np.percentile(df["review_chars_count"], 97.5)
		
		fig = px.histogram(
			df[(df["review_chars_count"] >= conf_int_low) &
			   (df["review_chars_count"] <= conf_int_high)],
			x="review_chars_count",
			nbins=30,
			labels={"review_chars_count": "N° de caracteres"},
			title="Histograma del n° de Caracteres"
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
	
	elif parameter == "Contra-portada: largo promedio de cada oración":
		conf_int_low = np.percentile(df["review_mean_sentence_length"], 2.5)
		conf_int_high = np.percentile(df["review_mean_sentence_length"], 97.5)
		
		fig = px.histogram(
			df[(df["review_mean_sentence_length"] >= conf_int_low) &
			   (df["review_mean_sentence_length"] <= conf_int_high)],
			x="review_mean_sentence_length",
			nbins=30,
			labels={"review_mean_sentence_length": "Largo promedio"},
			title="Histograma del Largo Promedio de Cada Oración"
		).update_layout(yaxis_title="Cantidad")
		st.plotly_chart(fig)
	
	elif parameter == "Contra-portada: variación del largo de oraciones":
		conf_int_low = np.percentile(df["review_std_sentence_length"], 2.5)
		conf_int_high = np.percentile(df["review_std_sentence_length"], 97.5)
		
		fig = px.histogram(
			df[(df["review_std_sentence_length"] > conf_int_low) &
			   (df["review_std_sentence_length"] <= conf_int_high)],
			x="review_std_sentence_length",
			nbins=30,
			labels={"review_std_sentence_length": "Variación"},
			title="Histograma de la Variación del Largo de las Oraciones"
		).update_layout(yaxis_title="Cantidad")
		st.plotly_chart(fig)
	
	elif parameter == "Contra-portada: proporción de palabras importantes":
		conf_int_low = np.percentile(df["review_non_stopwords_rate"], 2.5)
		conf_int_high = np.percentile(df["review_non_stopwords_rate"], 97.5)
		
		fig = px.histogram(
			df[(df["review_non_stopwords_rate"] > conf_int_low) &
			   (df["review_non_stopwords_rate"] <= conf_int_high)],
			x="review_non_stopwords_rate",
			nbins=30,
			labels={"review_non_stopwords_rate": "Proporción"},
			title="Histograma de la Proporción de Palabras Importantes"
		).update_layout(yaxis_title="Cantidad")
		st.plotly_chart(fig)

elif page == "Bi-Variado":
	parameter_list = ["...", "Nombre", "Género", "Editorial", "Precio", "Categoría",
					  "Año de publicación", "N° de páginas",
					  "Contra-portada: n° de caracteres",
					  "Contra-portada: largo promedio de cada palabra",
					  "Contra-portada: largo promedio de cada oración",
					  "Contra-portada: variación del largo de oraciones",
					  "Contra-portada: proporción de palabras importantes"]
	
	parameter = st.selectbox("Seleccione un parámetro:", parameter_list)
	
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
			title="Palabras más repetidas en el Nombre de los top 1.000 Libros"
		).update_layout(yaxis={"autorange": "reversed"})
		st.plotly_chart(fig)
	
	elif parameter == "Género":
		females_data = pd.concat([
			df[(df["female_author"] == 1) &
			   (df["male_author"] == 0)]["bayesian_rating"].copy(),
			pd.Series(index=df[(df["female_author"] == 1) &
							   (df["male_author"] == 0)].index,
					  data="Mujeres", name="género")
		], axis=1)
		females_data.reset_index(drop=True, inplace=True)
		
		males_data = pd.concat([
			df[(df["female_author"] == 0) &
			   (df["male_author"] == 1)]["bayesian_rating"].copy(),
			pd.Series(index=df[(df["female_author"] == 0) &
							   (df["male_author"] == 1)].index,
					  data="Hombres", name="género")
		], axis=1)
		males_data.reset_index(drop=True, inplace=True)
		
		nas_data = pd.concat([
			df[(df["female_author"] == 0) &
			   (df["male_author"] == 0)]["bayesian_rating"].copy(),
			pd.Series(index=df[(df["female_author"] == 0) &
							   (df["male_author"] == 0)].index,
					  data="No Especificado", name="género")
		], axis=1)
		nas_data.reset_index(drop=True, inplace=True)
		
		both_data = pd.concat([
			df[(df["female_author"] == 1) &
			   (df["male_author"] == 1)]["bayesian_rating"].copy(),
			pd.Series(index=df[(df["female_author"] == 1) &
							   (df["male_author"] == 1)].index,
					  data="No Especificado", name="género")
		], axis=1)
		both_data.reset_index(drop=True, inplace=True)
		
		plot_data = pd.concat([females_data, males_data, nas_data, both_data])
		
		del females_data, males_data, nas_data, both_data
		
		fig = px.strip(
			plot_data,
			x="género",
			y="bayesian_rating",
			labels={"género": "Género", "bayesian_rating": "Ranking"},
			title="Ranking vs Género"
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
			title="Palabras más repetidas en la Editorial de los top 1.000 libros"
		).update_layout(yaxis={"autorange": "reversed"})
		st.plotly_chart(fig)
	
	elif parameter == "Categoría":
		top1000_categories = ""
		
		for row in sorted_df[0:1000]["book_category"]:
			top1000_categories += row + " "
		
		commons = Counter(top1000_categories.split())
		commons_list = sorted(commons.items(), key=lambda x: x[1], reverse=True)
		
		fig = px.bar(
			x=[item[1] for item in commons_list[0:10]],
			y=[item[0] for item in commons_list[0:10]],
			labels={"x": "Cantidad", "y": "Categoría"},
			title="Categorías más repetidas en los top 1.000 libros"
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
	
	elif parameter == "Año de publicación":
		conf_int_low = np.percentile(df["book_year"], 2.5)
		conf_int_high = np.percentile(df["book_year"], 97.5)
		
		fig = px.scatter(
			df[(df["book_year"] >= conf_int_low) &
			   (df["book_year"] <= conf_int_high)],
			x="book_year",
			y="bayesian_rating",
			labels={"book_year": "Año"},
			title="Ranking vs Año de Publicación"
		).update_layout(yaxis_title="Ranking")
		st.plotly_chart(fig)
	
	elif parameter == "N° de páginas":
		conf_int_low = np.percentile(df["book_pages"], 2.5)
		conf_int_high = np.percentile(df["book_pages"], 97.5)
		
		fig = px.scatter(
			df[(df["book_pages"] >= conf_int_low) &
			   (df["book_pages"] <= conf_int_high)],
			x="book_pages",
			y="bayesian_rating",
			labels={"book_pages": "N° de páginas"},
			title="Ranking vs N° de Páginas"
		).update_layout(yaxis_title="Ranking")
		st.plotly_chart(fig)
	
	elif parameter == "Contra-portada: n° de caracteres":
		conf_int_low = np.percentile(df["review_chars_count"], 2.5)
		conf_int_high = np.percentile(df["review_chars_count"], 97.5)
		
		fig = px.scatter(
			df[(df["review_chars_count"] >= conf_int_low) &
			   (df["review_chars_count"] <= conf_int_high)],
			x="review_chars_count",
			y="bayesian_rating",
			labels={"review_chars_count": "N° de caracteres"},
			title="Ranking vs n° de Caracteres"
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
	
	elif parameter == "Contra-portada: largo promedio de cada oración":
		conf_int_low = np.percentile(df["review_mean_sentence_length"], 2.5)
		conf_int_high = np.percentile(df["review_mean_sentence_length"], 97.5)
		
		fig = px.scatter(
			df[(df["review_mean_sentence_length"] >= conf_int_low) &
			   (df["review_mean_sentence_length"] <= conf_int_high)],
			x="review_mean_sentence_length",
			y="bayesian_rating",
			labels={"review_mean_sentence_length": "Largo promedio"},
			title="Ranking vs Largo Promedio de Cada Oración"
		).update_layout(yaxis_title="Ranking")
		st.plotly_chart(fig)
	
	elif parameter == "Contra-portada: variación del largo de oraciones":
		conf_int_low = np.percentile(df["review_std_sentence_length"], 2.5)
		conf_int_high = np.percentile(df["review_std_sentence_length"], 97.5)
		
		fig = px.scatter(
			df[(df["review_std_sentence_length"] > conf_int_low) &
			   (df["review_std_sentence_length"] <= conf_int_high)],
			x="review_std_sentence_length",
			y="bayesian_rating",
			labels={"review_std_sentence_length": "Variación"},
			title="Ranking vs Variación del Largo de las Oraciones"
		).update_layout(yaxis_title="Ranking")
		st.plotly_chart(fig)
	
	elif parameter == "Contra-portada: proporción de palabras importantes":
		conf_int_low = np.percentile(df["review_non_stopwords_rate"], 2.5)
		conf_int_high = np.percentile(df["review_non_stopwords_rate"], 97.5)
		
		fig = px.scatter(
			df[(df["review_non_stopwords_rate"] > conf_int_low) &
			   (df["review_non_stopwords_rate"] <= conf_int_high)],
			x="review_non_stopwords_rate",
			y="bayesian_rating",
			labels={"review_non_stopwords_rate": "Proporción"},
			title="Ranking vs Proporción de Palabras Importantes"
		).update_layout(yaxis_title="Ranking")
		st.plotly_chart(fig)

elif page == "Multi-Variado":
	model_list = ["...", "Regresión Lineal", "Regresión Polinomial", "LightGBM", "Random Forest"]
	
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
	
	model = st.selectbox("Seleccione un modelo:", model_list)
	
	relevant_parameters = ["price", "book_pages", "editorial_planeta", "editorial_ediciones",
						   "category_ficcion", "category_no-ficcion", "category_juvenil"]
	
	if model == "Regresión Lineal":
		X = df[relevant_parameters]
		y = df["bayesian_rating"]
		
		lin_reg = LinearRegression()
		lin_reg.fit(X, y)
		
		y_pred = lin_reg.predict(X)
		
		fig = px.scatter(
			x=y,
			y=y_pred,
			labels={"x": "Valores Reales", "y": "Valores Obtenidos"},
			title="Predicción de Regresión Lineal"
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
		
		st.write("Error en los datos de prueba:")
		st.write(f"Valor medio: {round(mean_test_error, 6)} - Predicción: {round(test_error, 6)}")
		st.write(f"Representa una mejora del {1 - test_error / mean_test_error:.0%}.")
	
	
	elif model == "Regresión Polinomial":
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
			title="Predicción de Regresión Polinomial"
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
		
		st.write("Error en los datos de prueba:")
		st.write(f"Valor medio: {round(mean_test_error, 6)} - Predicción: {round(test_error, 6)}")
		st.write(f"Representa una mejora del {1 - test_error / mean_test_error:.0%}.")
	
	elif model == "LightGBM":
		X = df[relevant_parameters]
		y = df["bayesian_rating"]
		
		lgb_reg = LGBMRegressor(random_state=1919)
		lgb_reg.fit(X, y)
		
		y_pred = lgb_reg.predict(X)
		
		fig = px.scatter(
			x=y,
			y=y_pred,
			labels={"x": "Valores Reales", "y": "Valores Obtenidos"},
			title="Predicción de LightGBM"
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
		
		st.write("Error en los datos de prueba:")
		st.write(f"Valor medio: {round(mean_test_error, 6)} - Predicción: {round(test_error, 6)}")
		st.write(f"Representa una mejora del {1 - test_error / mean_test_error:.0%}.")
	
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
			title="Predicción de Random Forest"
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
		
		st.write("Error en los datos de prueba:")
		st.write(f"Valor medio: {round(mean_test_error, 6)} - Predicción: {round(test_error, 6)}")
		st.write(f"Representa una mejora del {1 - test_error / mean_test_error:.0%}.")
