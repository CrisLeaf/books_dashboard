from unidecode import unidecode
import re


def clean_spaces(string):
	string = string.replace("\n\r", " ")
	string = string.replace("\r\n", " ")
	string = string.replace("\n", " ")
	string = string.replace("\t", " ")
	string = string.replace("\r", " ")
	
	return " ".join(string.split())

def clean_uppers(string):
	return string.lower()

def clean_prices(value):
	value = value.replace("$", "")
	value = value.replace(",", "")
	value = value.replace(".", "")
	
	return value.strip()

def clean_unicodes(string):
	return unidecode(string)

def clean_links(link):
	return re.findall(r"<200(.*)>", str(link))[0].strip()
