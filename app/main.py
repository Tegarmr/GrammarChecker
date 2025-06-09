from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, Column, Integer, Text, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# ENV config
DATABASE_URL = os.getenv("DATABASE_URL")

# DB setup
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Define DB model
class GrammarCheck(Base):
    __tablename__ = "grammar_checks"

    grammar_check_id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    check_grammar = Column(Text)
    hasil = Column(Text)
    user_id = Column(Integer, nullable=True)

# Create table if not exists
Base.metadata.create_all(bind=engine)

# FastAPI app
app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load grammar correction model
tokenizer = T5Tokenizer.from_pretrained("vennify/t5-base-grammar-correction")
model = T5ForConditionalGeneration.from_pretrained("vennify/t5-base-grammar-correction")

# Request schema
class TextInput(BaseModel):
    text: str
    user_id: int | None = None

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Route: Grammar Correction + DB Save
@app.post("/correct")
def correct_text(data: TextInput, db: Session = Depends(get_db)):
    input_text = "grammar: " + data.text
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs, max_length=512, num_beams=5, early_stopping=True)
    corrected = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Save to DB
    # entry = GrammarCheck(
    #     check_grammar=data.text,
    #     hasil=corrected,
    #     created_at=datetime.utcnow(),
    #     user_id=data.user_id
    # )
    # db.add(entry)
    # db.commit()
    # db.refresh(entry)

    return {
        "check": data.text,
        "hasil": corrected
    }
