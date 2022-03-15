from flask import Blueprint, render_template, request
from .bot_core import RecommendationsBot
from .psql_secrets import psql_params


bp = Blueprint("application", __name__, url_prefix="/")

@bp.route("/", methods=["GET"])
def index():
	global rec_bot
	rec_bot = RecommendationsBot(psql_params)
	rec_bot.get_data_from_psql()
	
	return render_template("index.html")

@bp.route("/recom", methods=["POST"])
def recommend():
	rec_bot = RecommendationsBot(psql_params)
	rec_bot.get_data_from_psql()
	
	user_sentence = request.form["text"]
	recommendation = rec_bot.recommend(user_sentence)
	
	return render_template("index.html", recommendation=recommendation)
