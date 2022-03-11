import psycopg2
from psql_secrets import psql_params


def reset_tables():
	commands = (
		"""
		DROP TABLE IF EXISTS books
		""",
		"""
		DROP TABLE IF EXISTS websites
		""",
		"""
		CREATE TABLE websites (id SERIAL PRIMARY KEY,
							   name TEXT NOT NULL)
		""",
		"""
		CREATE TABLE books (id SERIAL PRIMARY KEY,
							name TEXT NOT NULL,
							author TEXT,
							editorial TEXT,
							price INT,
							review TEXT,
							description TEXT,
							five_stars TEXT,
							four_stars TEXT,
							three_stars TEXT,
							two_stars TEXT,
							one_star TEXT,
							link TEXT NOT NULL,
							website INT,
							CONSTRAINT fk_web FOREIGN KEY (website) REFERENCES websites (id))
		"""
	)
	
	conn = psycopg2.connect(**psql_params)
	curr = conn.cursor()
	
	for command in commands:
		curr.execute(command)
	
	print("Base de datos inicializada!")
	
	conn.commit()
	curr.close()
	conn.close()

if __name__ == "__main__":
	reset_tables()
