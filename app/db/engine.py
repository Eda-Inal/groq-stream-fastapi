from sqlalchemy.ext.asyncio import create_async_engine

from app.core.config import settings


DATABASE_URL = settings.database_url

engine = create_async_engine(
    DATABASE_URL,
    echo=True,  # dev only
    pool_pre_ping=True,
)
