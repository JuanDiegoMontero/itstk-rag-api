import os
import logging
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Configuración del logger para trazabilidad del flujo de ejecución y depuración
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./data/chroma_db")

logger.info("Inicializando modelos y conexión a ChromaDB...")

# Instanciación del LLM local. Temperatura fijada en 0.0 para asegurar respuestas 
# altamente deterministas y mitigar el riesgo de alucinaciones corporativas.
llm = Ollama(
    base_url=OLLAMA_BASE_URL,
    model=OLLAMA_MODEL,
    temperature=0.0
)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)

# Configuración del retriever utilizando Maximal Marginal Relevance (MMR).
# Esto garantiza que los chunks inyectados al prompt no solo sean semánticamente 
# relevantes, sino también diversos entre sí, evitando redundancia en el contexto.
retriever = vectorstore.as_retriever(
    search_type="mmr", 
    search_kwargs={"k": 2, "fetch_k": 5}
)


# Decidí implementar etiquetas <contexto> y <pregunta> porque los modelos de la familia Llama/Mistral 
# parsean mucho mejor las fronteras cognitivas entre "mis instrucciones" y "la data del usuario" 
# cuando se usan demarcadores claros. Además, al estructurar las restricciones como una lista de 
# "REGLAS DE OPERACIÓN", el mecanismo de atención del LLM procesa las barreras anti-alucinación 
# de forma mucho más estricta que con un bloque de texto plano.
prompt_template = """Eres un asistente experto de soporte técnico para ITSTK S.A.S.
Tu único objetivo es responder a la pregunta del usuario utilizando ESTRICTAMENTE la información proporcionada.

REGLAS DE OPERACIÓN:
1. Si la respuesta no está explícitamente en la sección de contexto, debes responder EXACTAMENTE con la frase: "No encuentro esa información en los documentos disponibles".
2. Bajo ninguna circunstancia debes inventar, deducir o agregar información externa.
3. Tu respuesta debe ser directa, en español, clara y extremadamente concisa.

<contexto>
{context}
</contexto>

<pregunta>
{question}
</pregunta>

Respuesta:"""

PROMPT = PromptTemplate.from_template(prompt_template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Implementación de la cadena lógica usando LCEL (LangChain Expression Language)
# para una sintaxis declarativa y soporte nativo asíncrono.
async def procesar_pregunta_async(pregunta: str):
    logger.info(f"Procesando nueva pregunta: '{pregunta}'")
    
    docs_recuperados = await retriever.ainvoke(pregunta)
    contexto_str = format_docs(docs_recuperados)
    
    logger.info(f"Se recuperaron {len(docs_recuperados)} fragmentos de contexto usando MMR.")

    cadena = PROMPT | llm | StrOutputParser()
    
    respuesta = await cadena.ainvoke({"context": contexto_str, "question": pregunta})
    
    logger.info("Respuesta del LLM generada exitosamente.")
    
    return {
        "result": respuesta.strip(), # Agregamos strip() para limpiar espacios o saltos de línea innecesarios al inicio/fin
        "source_documents": docs_recuperados
    }