from pydantic import BaseModel, Field
from typing import List

# Definición estricta de esquemas de entrada/salida para validación automática de payloads
class QueryRequest(BaseModel):
    # Se impone una longitud mínima para evitar procesar consultas carentes de contexto semántico
    pregunta: str = Field(
        ..., 
        min_length=5, 
        description="Pregunta en lenguaje natural sobre los documentos indexados.", 
        json_schema_extra={"example": "¿Cuál es el tiempo de respuesta para incidencias críticas?"}
    )

class QueryResponse(BaseModel):
    respuesta: str = Field(description="Respuesta generada por el LLM basada en el contexto.")
    fuentes: List[str] = Field(description="Lista de fragmentos de texto recuperados de la base de datos vectorial.")
    tiempo_ms: int = Field(description="Tiempo total de procesamiento en milisegundos.")