from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    api_title: str = "Dental Caries Detection API"
    api_version: str = "1.0.0"
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", 8000))
    debug: bool = os.getenv("DEBUG", "True").lower() == "true"
    
    model_path: str = os.getenv("MODEL_PATH", "models/dental_caries_model.pt")
    image_size: int = int(os.getenv("IMAGE_SIZE", 224))
    
    class Config:
        env_file = ".env"

settings = Settings()