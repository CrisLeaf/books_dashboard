import psycopg2
from .psql_secrets import psql_params


class BooksScrapyPipeline(object):
	
	def open_spider(self, spider):
		try:
			self.conn = psycopg2.connect(**psql_params)
		except Exception as e:
			raise ConnectionError(e)
		
		self.curr = self.conn.cursor()
	
	def close_spider(self, spider):
		self.curr.close()
		self.conn.close()
	
	def process_item(self, item, spider):
		try:
			self.curr.execute("SELECT (id) FROM websites WHERE website = %s",
							  (item["website"],))
			record = self.curr.fetchall()
			
			if len(record) == 0:
				self.curr.execute("INSERT INTO websites (website) VALUES (%s)",
								  (item["website"],))
				self.curr.execute("SELECT (id) FROM websites WHERE website = %s",
								  (item["website"],))
				record = self.curr.fetchall()
				website_id = record[0][0]
			else:
				website_id = record[0][0]
			
			print("\n")
			print(website_id)
			print("\n")
			
			self.curr.execute(
				"""
				INSERT INTO books (name, author, editorial, price, review, description,
								   five_stars, four_stars, three_stars, two_stars, one_star,
								   link, website)
				VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
				""",
				(item["name"], item["author"], item["editorial"], item["price"], item["review"],
				 item["description"], item["five_stars"], item["four_stars"],
				 item["three_stars"], item["two_stars"], item["one_star"], item["link"],
				 website_id)
			)
			self.conn.commit()
		
		except:
			self.conn.rollback()
			raise
		
		return item
