import os

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


class BotConfigs(BaseModel):
    BOT_TOKEN : str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    MODEL_PATH : str = "./checkpoints/reward_model"

class Prompt(BaseModel):
    system : str = "You are a helpful assistant."