import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    
#---------------------------------------------------------------------
# Definir una clase de documento
class Documento:
          def __init__(self, page_content, metadata=None):
            self.page_content = page_content
            self.metadata = metadata
    
#---------------------------------------------------------------------
# DEFINICIÓN DEL PROCESO DE CHAIN
    
def process_chain(texto):
  # Configurar una cadena de procesamiento simple
  text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        
  # Dividir el texto en fragmentos
  fragmentos = text_splitter.split_text(texto)
        
  # Aquí puedes agregar más pasos al chain si es necesario
  # Por ejemplo, podrías analizar sentimientos, extraer entidades, etc.
        
  return fragmentos
    
          #---------------------------------------------------------------------
          
          # PDF
          
          # Obtener los datos de la base de datos en lotes
          # docs_clientes = obtener_datos_clientes()
    
          # # Pasar cada documento a través de la chain de procesamiento
    
          # docs_procesados = []
          # for doc in docs_clientes:
          #           fragmentos = process_chain(doc.page_content)
          # for fragmento in fragmentos:
          #           docs_procesados.append(Documento(page_content=fragmento, metadata=doc.metadata))
    
          # Ahora podemos añadir también los documentos PDF
    
pdfpath = 'MANUAL_COBRANZA.pdf'
loader = PyPDFLoader(pdfpath)
doc_text = loader.load()
    
# Dividir los textos del PDF
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
docs_pdf = text_splitter.split_documents(doc_text)
    
# Convertir los documentos PDF en objetos Documento
docs_pdf = [Documento(page_content=doc.page_content, metadata=doc.metadata) for doc in docs_pdf]
    
# Unir los documentos PDF con los datos de los clientes procesados
docs_total = docs_pdf #+ docs_procesados


# CREACIÓN DE LA BASE DE DATOS VECTORIAL
    
# Cargar el modelo de embeddings
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    
# Convertir documentos a vectores e indexar los vectores
db_faiss = FAISS.from_documents(docs_total, embeddings)
    
# Guardar la base de datos vectorial en el disco
db_faiss.save_local("faiss_inicial")
    
