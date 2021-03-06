from stop_words import get_stop_words


def prepare_stop_words(language):
	stop_words = get_stop_words(language)
	stop_words = " ".join(stop_words)
	
	stop_words = stop_words.replace("á", "a")
	stop_words = stop_words.replace("é", "e")
	stop_words = stop_words.replace("í", "i")
	stop_words = stop_words.replace("ó", "o")
	stop_words = stop_words.replace("ú", "u")
	
	stop_words = stop_words.split(" ")
	
	return stop_words

spanish_stop_words = prepare_stop_words("es")
