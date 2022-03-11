import scrapy
from itemloaders.processors import TakeFirst, MapCompose
from w3lib.html import remove_tags
from .cleaning_functions import (
	clean_spaces, clean_uppers, clean_prices, clean_unicodes, clean_links
)


class BooksScrapyItem(scrapy.Item):
	name = scrapy.Field(input_processor=MapCompose(remove_tags, clean_spaces,
												   clean_uppers, clean_unicodes),
						output_processor=TakeFirst())
	author = scrapy.Field(input_processor=MapCompose(remove_tags, clean_spaces,
													 clean_uppers, clean_unicodes),
						  output_processor=TakeFirst())
	editorial = scrapy.Field(input_processor=MapCompose(remove_tags, clean_spaces,
														clean_uppers, clean_unicodes),
							 output_processor=TakeFirst())
	price = scrapy.Field(input_processor=MapCompose(remove_tags, clean_prices),
						 output_processor=TakeFirst())
	review = scrapy.Field(input_processor=MapCompose(remove_tags, clean_spaces,
													 clean_uppers, clean_unicodes),
						  output_processor=TakeFirst())
	description = scrapy.Field(input_processor=MapCompose(remove_tags, clean_spaces,
														  clean_uppers, clean_unicodes),
							   output_processor=TakeFirst())
	five_stars = scrapy.Field(input_processor=MapCompose(remove_tags, clean_spaces),
							  output_processor=TakeFirst())
	four_stars = scrapy.Field(input_processor=MapCompose(remove_tags, clean_spaces),
							  output_processor=TakeFirst())
	three_stars = scrapy.Field(input_processor=MapCompose(remove_tags, clean_spaces),
							   output_processor=TakeFirst())
	two_stars = scrapy.Field(input_processor=MapCompose(remove_tags, clean_spaces),
							 output_processor=TakeFirst())
	one_star = scrapy.Field(input_processor=MapCompose(remove_tags, clean_spaces),
							output_processor=TakeFirst())
	link = scrapy.Field(input_processor=MapCompose(clean_links),
						output_processor=TakeFirst())
	website = scrapy.Field(input_processor=MapCompose(), output_processor=TakeFirst())
