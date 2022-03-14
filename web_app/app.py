from bot_core import RecommendationsBot
from psql_secrets import psql_params


rec_bot = RecommendationsBot(psql_params)
rec_bot.get_data_from_psql()
print(rec_bot.recommend("me gustan las manzanas"))
