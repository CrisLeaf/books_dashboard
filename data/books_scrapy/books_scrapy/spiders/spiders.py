import scrapy
from books_scrapy.items import BooksScrapyItem
from scrapy.loader import ItemLoader


class BuscalibreSpider(scrapy.Spider):
	"""
	Buscalibre Spider used to crawl through books data from buscalibre website.
	It iterates into each product of the website retrieving the HTML tags,
	to process them into an Item Loader.
	"""
	
	name = "buscalibre"
	page_number = 1
	start_urls = ["https://www.buscalibre.cl/libros-envio-express-chile_t.html"]
	
	def parse(self, response):
		for book_icon in response.xpath(".//div[@class='productos pais42']/div"):
			book_link = book_icon.xpath(".//a/@href").get()
			
			yield response.follow(url=book_link, callback=self.parse_book)
		
		print("\n\nASDFASDFASDFASD\n\n")
		
		if BuscalibreSpider.page_number <= 3:  # total pages: 134
			BuscalibreSpider.page_number += 1
			next_page = "https://www.buscalibre.cl/libros-envio-express-chile_t.html?page=" + \
						str(BuscalibreSpider.page_number)
			
			yield response.follow(url=next_page, callback=self.parse)
	
	def parse_book(self, response):
		loader = ItemLoader(item=BooksScrapyItem(), selector=response)
		loader.add_xpath("name", ".//h1")
		
		# loader.add_xpath("five_stars", ".//ul[@class='evaluacion']/li[1]/text()")
		# loader.add_xpath("four_stars", ".//ul[@class='evaluacion']/li[2]/text()")
		# loader.add_xpath("three_stars", ".//ul[@class='evaluacion']/li[3]/text()")
		# loader.add_xpath("two_stars", ".//ul[@class='evaluacion']/li[4]/text()")
		# loader.add_xpath("one_star", ".//ul[@class='evaluacion']/li[5]/text()")
		
		yield loader.load_item()
