# app/repo/db.py
import psycopg2
from pgvector.psycopg2 import register_vector
from app.core.settings import DATABASE_URL

def db():
    conn = psycopg2.connect(DATABASE_URL)
    register_vector(conn)
    return conn
