import os
from langchain_community.chat_models import ChatOpenAI  # Cambiar OpenAI por ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import mysql.connector
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
MYSQL_USERNAME = os.getenv('MYSQL_USERNAME')
MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD')


# Configuración del modelo de lenguaje
llm = ChatOpenAI(model_name="gpt-4", openai_api_key=OPENAI_API_KEY)

tabla = 'deuda'  # Nombre de la tabla donde están los morosos
columna_id = "dni"
columna_numero = 'monto'  # Monto de la deuda


# Function to load the vector database
def load_politicas_cobranza():
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    DB_FAISS_PATH = 'vector_store'
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    return db

def connect_to_database():
    try:
        conn = mysql.connector.connect(
            host='localhost',
            user=MYSQL_USERNAME,
            password=MYSQL_PASSWORD,
            database='facturaelectronica'
        )
        return conn
    except mysql.connector.Error as err:
        print(f"Error al conectar con la base de datos: {err}")
        return None

# Conectar a la base de datos y obtener los montos de deuda por DNI
def obtener_deudas_por_dni(dni):
    try:
        print("Conectando a la base de datos...")   
        conn = connect_to_database()
        if conn is None:
            return []
        print("Conexión exitosa")
        cursor = conn.cursor()
        query = f"SELECT {columna_numero} FROM {tabla} WHERE {columna_id} = %s"
        print(query)
        cursor.execute(query, (dni,))
        resultados = cursor.fetchall()
        conn.close()
        return [monto[0] for monto in resultados]  # Devuelve una lista de montos
    except Exception as e:
        print(f"Error al conectar con la base de datos: {e}")
        return []    

# Definir la herramienta para obtener deudas por DNI
calc_tool = Tool(
    name="ObtenerDeudasPorDNI",
    func=obtener_deudas_por_dni,
    description="Usa esta herramienta para buscar deuda por DNI."
)

# Definir la herramienta para obtener deudas por DNI
rag_tool = Tool(
    name="ObtenerPoliticasCobranza",
    func=load_politicas_cobranza,
    description="Usa esta herramienta para obtener información de politicas de cobranza."
)

# Configurar la memoria
memory = ConversationBufferMemory(memory_key="conversation_history")

# Ingeniería de prompt
prompt = PromptTemplate(
    input_variables=["conversation_history", "input"],
    template="""
    Eres un asistente muy útil y amable. Usa las herramientas que tienes disponibles para responder la siguiente consulta de la mejor manera posible.

    Aquí está el historial de la conversación hasta ahora:
    {conversation_history}

    Pregunta:
    {input}

    Si necesitas obtener la deuda de una persona, usa la herramienta "ObtenerDeudasPorDNI". Si la persona quiere pagar su deuda usa  la información que brinda la herramienta "ObtenerPoliticasCobranza".

    Da una respuesta detallada y bien estructurada.
    """
)

# Inicializar el agente
tools = [calc_tool]
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    prompt=prompt
)

# Ejemplo de uso
response = agent.run("Hola, me llamo Jose, quiero saber mi Deuda con DNI 25723525 y como puedo refinanciar esta deuda según tus políticas y ver que opción de pago escoger.")
print(response)
