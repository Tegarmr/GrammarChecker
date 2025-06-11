from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, Column, Integer, Text, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class GrammarCheck(Base):
    __tablename__ = "grammar_checks"
    grammar_check_id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    check_grammar = Column(Text)
    hasil = Column(Text)
    user_id = Column(Integer, nullable=True)

Base.metadata.create_all(bind=engine)

app = FastAPI()

origins = [
    "http://localhost:5173",          
    "https://fe-deploy-ppl.vercel.app",
    "https://ab-course-fe-go-live.vercel.app" 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tokenizer = None
model = None

def load_model():
    global tokenizer, model
    if tokenizer is None or model is None:
        tokenizer = T5Tokenizer.from_pretrained("app/cache_model")
        model = T5ForConditionalGeneration.from_pretrained("app/cache_model")

class TextInput(BaseModel):
    text: str
    user_id: int | None = None

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/correct")
def correct_text(data: TextInput, db: Session = Depends(get_db)):
    load_model()
    input_text = "grammar: " + data.text
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs, max_length=512, num_beams=5, early_stopping=True)
    corrected = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Uncomment untuk simpan ke DB
    # entry = GrammarCheck(check_grammar=data.text, hasil=corrected, user_id=data.user_id)
    # db.add(entry)
    # db.commit()
    # db.refresh(entry)

    return {"check": data.text, "hasil": corrected}
