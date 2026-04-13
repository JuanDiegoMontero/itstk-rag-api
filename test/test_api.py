from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_consultar_endpoint_success():
    # Caso 1: Validación del Happy Path donde la respuesta existe explícitamente en el corpus.
    payload = {
        "pregunta": "¿Cuál es el horario de soporte?"
    }
    response = client.post("/consultar", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    
    # Verificación de la estructura del contrato de salida
    assert "respuesta" in data
    assert "fuentes" in data
    assert "tiempo_ms" in data
    assert type(data["fuentes"]) == list
    assert type(data["tiempo_ms"]) == int

def test_consultar_endpoint_out_of_context():
    # Caso 2: Validación de mitigación de alucinaciones.
    # El LLM debe rechazar responder preguntas fuera del dominio del documento.
    payload = {
        "pregunta": "¿Cuál es la capital de Francia?"
    }
    response = client.post("/consultar", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "No encuentro esa información en los documentos disponibles" in data["respuesta"]

def test_consultar_endpoint_validation_error():
    # Caso 3: Validación del tipado estricto.
    # Se envía un payload malformado esperando el bloqueo automático de FastAPI/Pydantic.
    payload = {
        "query": "texto aleatorio"
    }
    response = client.post("/consultar", json=payload)
    
    # Se espera un HTTP 422 Unprocessable Entity
    assert response.status_code == 422