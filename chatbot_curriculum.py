# 1 IMPORT LIBRARIES ------------------------------------------

import os
import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dataclasses import dataclass
from fastapi import FastAPI, HTTPException
from fastapi import Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

# 2 ENVIROMENT VARIABLES --------------------------------------

load_dotenv()
GEMINI_API_KEY = str(os.getenv('GEMINI_API_KEY'))
FILE_CV = str(os.getenv('FILE_CV'))


# 3 CHUKING  --------------------------------------------------

    # exp: Dividir em partes menores e gerenciáveis

def load_curriculo(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def split_text(text, max_length=500):

    """  Dividir currículo em trechos """

    sentences = text.split('. ')
    chunks, chunk = [], ""

    for sent in sentences:
        if len(chunk) + len(sent) < max_length:
            chunk += sent + ". " # Se for menor que tamanho maximo pega o anterior soma ao sent e coloca ponto
        else:
            chunks.append(chunk.strip())
            chunk = sent + ". "

    if chunk:
        chunks.append(chunk.strip())

    return chunks


# 4 EMBEDDINGS AND FAISS INDEX --------------------------------------

class retriver:
  def __init__(self,texts):
      self.texts = texts
      self.model = SentenceTransformer('all-MiniLM-L6-v2')
      self.embeddings = self.model.encode(self.texts, convert_to_numpy=True)
      self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
      self.index.add(self.embeddings)

  def retrieve(self,query,top_k=3):
      query_embedding = self.model.encode([query], convert_to_numpy=True)
      distances, indices = self.index.search(query_embedding, top_k)
      return [self.texts[i] for i in indices[0]]
  

# 5 GENERATION RESPONSE --------------------------------------

def gerar_resposta_llm(prompt, context):

    data = {
        
        'messages': [
            {'role': 'system', 'content': 'você é um assistente que responde com base apenas no currículo fornecido'},
            {'role':'user','content': f'contexto: {context} pergunta: {prompt}'}
        ],
        'temperature': 0.3, # A temperatura baixa (0.3) garante respostas mais determinísticas e aderentes ao contexto.
        # 'stream':False
    }
    
    # client = genai.Client(api_key=GEMINI_API_KEY)
    genai.configure(api_key=GEMINI_API_KEY)

    role_system = data['messages'][0]['role']
    role_user  = data['messages'][1]['role']
    content_system = data['messages'][0]['content']
    content_user = data['messages'][1]['content']
    contents = [role_system + ' ' + content_system, role_user+' '+ content_user]

    model = genai.GenerativeModel("gemini-2.0-flash")

    try:
        response = model.generate_content(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Erro na API da DeepSeek")

    return response.text

#6 PIPELINE DE RESPOSTA --------------------------------------

def responder_pergunta(curriculo_path, pergunta) -> str :

    curriculo = load_curriculo(curriculo_path) # 1) Carregar
    trechos = split_text(curriculo) #2) Seperar o curriculo em chunks
    retriever_parts = retriver(trechos) # 3) pegar esses chunks, criar um objeto retriever(em vetores embeddings) e add ao faiss
    context = retriever_parts.retrieve(pergunta) #4) buscar no faiss a pergunta e busca o top 3 no faiss
    contexto = '\n\n'.join(context)
    resposta = gerar_resposta_llm(pergunta, contexto) # pedir para LLM gerar a resposta

    return str(resposta)


#7 INSTANCIA APP ----------------------------------------------

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


# @dataclass
# class Question:
#    question:str

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "mensagens": []})

@app.post('/', response_class= HTMLResponse)
async def responder_api(request: Request, question:str = Form(...)):
    response = responder_pergunta(FILE_CV, question)
    message = [{"usuario": question, "bot": response}]
    return templates.TemplateResponse("index.html",{"request": request, "mensagens": message})


# uvicorn chatbot_curriculum.py:app --reload

