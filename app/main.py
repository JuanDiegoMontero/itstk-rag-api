import time
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from app.models import QueryRequest, QueryResponse
from app.rag import procesar_pregunta_async

logger = logging.getLogger(__name__)

# Configuración de FastAPI con metadatos para la generación automática de la especificación OpenAPI
app = FastAPI(
    title="API RAG ITSTK - Motor de Consulta Inteligente",
    description="API robusta para la consulta de acuerdos de nivel de servicio (SLA) mediante recuperación vectorial (MMR) y LLM local.",
    version="2.0.0"
)

# Implementación de un manejador de excepciones global para garantizar que el cliente 
# siempre reciba respuestas JSON estructuradas, incluso ante fallos no controlados del servidor.
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Error interno del servidor atrapado: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Se produjo un error interno al procesar su solicitud. Por favor, intente de nuevo."}
    )

@app.post("/consultar", response_model=QueryResponse, tags=["Motor RAG"])
async def consultar(request: QueryRequest):
    try:
        start_time = time.time()
        
        # Ejecución asíncrona de la cadena RAG para evitar el bloqueo del event loop principal,
        # maximizando la concurrencia de la API bajo carga.
        resultado = await procesar_pregunta_async(request.pregunta)
        
        tiempo_ms = int((time.time() - start_time) * 1000)
        fuentes = [doc.page_content for doc in resultado["source_documents"]]
        
        return QueryResponse(
            respuesta=resultado["result"],
            fuentes=fuentes,
            tiempo_ms=tiempo_ms
        )
    except Exception as e:
        # Se delega la excepción al manejador global configurado en la capa superior
        raise e