
import os
from langchain_community.vectorstores import FAISS  # Para la carga y búsqueda en la base de datos vectorial
from langchain_openai import OpenAIEmbeddings  # Para las embeddings de OpenAI
 # Para cargar la clave de la API de OpenAI

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

#---------------------------------------------------------------------
# LLAMADA A LA BASE DE DATOS VECTORIAL
    
# Cargar el modelo de embeddings
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    
#Cargar la base de datos vectorial desde el disco
db_faiss_loaded = FAISS.load_local("faiss_inicial", embeddings, allow_dangerous_deserialization=True)
    
#Consulta de búsqueda
query = "cómo manejo un cliente molesto"
docs = db_faiss_loaded.similarity_search(query)
print(docs[0].page_content)
