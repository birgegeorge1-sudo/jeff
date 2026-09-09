import os
from dotenv import load_dotenv
load_dotenv()

MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY','')
MODEL = os.getenv('MODEL','mistral-large-latest')
MYSQL_URL = os.getenv('MYSQL_URL','')
HOST = os.getenv('HOST','127.0.0.1')
PORT = int(os.getenv('PORT','8000'))
MAX_HISTORY = int(os.getenv('MAX_HISTORY','40'))
MIN_QUESTION_INTERVAL = int(os.getenv('MIN_QUESTION_INTERVAL','3'))
