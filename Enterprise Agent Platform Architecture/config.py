import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    app_name: str = "Cohere Enterprise Agent Platform"
    environment: str = os.getenv("ENVIRONMENT", "production")
    cohere_api_key: str
    qdrant_url: str = os.getenv("QDRANT_URL", "http://qdrant:6333")
    db_file: str = os.getenv("DB_FILE", "/data/audit_ledger.db")
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()