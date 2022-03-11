from unidecode import unidecode


def clean_spaces(string):
	string = string.replace("\n\r", " ")
	string = string.replace("\r\n", " ")
	string = string.replace("\n", " ")
	string = string.replace("\t", " ")
	string = string.replace("\r", " ").strip()
	
	return string

def clean_uppers(string):
	return string.lower()

def clean_prices(value):
	value = value.replace("$", "")
	value = value.replace(",", "")
	value = value.replace(".", "").strip()
	
	return value

def clean_unicodes(string):
	return unidecode(string)
