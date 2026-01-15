from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from dotenv import load_dotenv
import os

username = os.getenv('DB_USERNAME', 'hymv')
password = os.getenv('DB_PASSWORD', 'qwer1234')
host = os.getenv('DB_HOST', '49.50.136.24')
port = os.getenv('DB_PORT', '3306')
database = os.getenv('DB_NAME', 'HYMV')

DB_URL = f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}"

engine = create_engine(DB_URL, echo=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()