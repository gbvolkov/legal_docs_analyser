from dotenv import load_dotenv,dotenv_values
import os

from pathlib import Path

documents_path = Path.home() / ".env"

load_dotenv(os.path.join(documents_path, 'gv.env'))

TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_SHIPMENT_DATA_BOT_TOKEN')

GIGA_CHAT_USER_ID=os.environ.get('GIGA_CHAT_USER_ID')
GIGA_CHAT_SECRET = os.environ.get('GIGA_CHAT_SECRET')
GIGA_CHAT_AUTH = os.environ.get('GIGA_CHAT_AUTH')
GIGA_CHAT_SCOPE = "GIGACHAT_API_PERS"

LANGCHAIN_API_KEY = os.environ.get('LANGCHAIN_API_KEY')
LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

YA_API_KEY = os.environ.get('YA_API_KEY')
YA_FOLDER_ID = os.environ.get('YA_FOLDER_ID')
YA_AUTH_TOKEN = os.environ.get('YA_AUTH_TOKEN')

GEMINI_API_KEY=os.environ.get('GEMINI_API_KEY')

CHECK_RIGHTS=os.environ.get('CHECK_RIGHTS', default='False')
DATA_ROOT_PATH=os.environ.get('DATA_ROOT_PATH')
WHISPER_MODEL_PATH=os.environ.get('WHISPER_MODEL_PATH')

SHIPMENTS_SHEET_ID=os.environ.get('SHIPMENTS_SHEET_ID')
PROCUREMENTS_SHEET_ID=os.environ.get('PROCUREMENTS_SHEET_ID')
GOOGLE_SHEETS_CRED=os.environ.get('GOOGLE_SHEETS_CRED')
WHISPER_MODEL = os.environ.get('WHISPER_MODEL', default='small')
