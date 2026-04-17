from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from parser_utils import parse_uploaded_file
from rag_pipeline import (
    answer_question,
    get_all_document_text,
    get_embedding_model,
    index_document,
    retrieve_relevant_chunks,
)
from summary_utils import generate_summary, generate_flashcards, extract_topics
from quiz_utils import generate_quiz

app = FastAPI()

class AskRequest(BaseModel):
    question: str

class SummaryRequest(BaseModel):
    mode: str = "quick"

class QuizRequest(BaseModel):
    count: int = 5

@app.get("/")
def root():
    return {"status": "ok", "message": "KnowFlow backend running"}

@app.post("/ask")
def ask(req: AskRequest):
    model, _ = get_embedding_model()
    chunks = retrieve_relevant_chunks(req.question, model)
    if not chunks:
        return {"answer": "No relevant evidence found.", "mode": "no_retrieval"}
    answer, mode = answer_question(req.question, chunks)
    return {"answer": answer, "mode": mode}

@app.post("/summary")
def summary(req: SummaryRequest):
    full_text = get_all_document_text()
    result = generate_summary(full_text, req.mode)
    return {"summary": result}

@app.post("/quiz")
def quiz(req: QuizRequest):
    full_text = get_all_document_text()
    result = generate_quiz(full_text, req.count)
    return {"quiz": result}

@app.post("/flashcards")
def flashcards():
    full_text = get_all_document_text()
    result = generate_flashcards(full_text)
    return {"flashcards": result}

@app.get("/topics")
def topics():
    full_text = get_all_document_text()
    result = extract_topics(full_text)
    return {"topics": result}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    content = await file.read()
    temp = UploadFile(filename=file.filename, file=None)
    temp._file = None
    parsed = parse_uploaded_file(file)
    if parsed.error:
        return {"ok": False, "error": parsed.error}

    model, _ = get_embedding_model()
    result = index_document(parsed.filename, parsed.text, model)
    return result