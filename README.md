# API REST - RAG para Consulta de SLA (ITSTK S.A.S.)

Esta es una API REST construida con **FastAPI** y **LangChain** que implementa un sistema RAG (Retrieval-Augmented Generation) 100% local. Utiliza **Ollama** (Mistral) como LLM y **ChromaDB** como base de datos vectorial.

---

## 📦 Dependencias
El proyecto está construido sobre Python 3.10+. Todas las dependencias están centralizadas en el archivo `requirements.txt`. Las tecnologías principales utilizadas son:

* **FastAPI & Uvicorn:** Framework web asíncrono y servidor ASGI.
* **LangChain (Core / Community):** Orquestación del flujo RAG utilizando arquitectura moderna LCEL.
* **ChromaDB:** Base de datos vectorial persistente en disco local.
* **Ollama (Mistral):** Inferencia del Modelo de Lenguaje Grande de forma 100% local.
* **HuggingFace Embeddings:** Generación de vectores usando el modelo ligero `all-MiniLM-L6-v2`.

---

## ⚙️ Variables de Entorno (.ENV)
Para cumplir estrictamente con los entregables solicitados en las instrucciones de esta prueba, el archivo `.env` con las variables de entorno ha sido incluido directamente en este repositorio. 

*(Nota: Soy consciente de que en un entorno real de producción o trabajo en equipo, los archivos .env jamás se suben al control de versiones por seguridad. Lo incluyo aquí explícitamente para facilitar su proceso de evaluación).*

Las variables configuradas para correr el proyecto localmente son:
```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral
CHROMA_DB_DIR=./data/chroma_db
DOCUMENT_PATH=./data/CONTRATO_SLA.txt
```


## Arquitectura y Decisiones de Diseño
* **LCEL (LangChain Expression Language):** Se implementó la cadena lógica usando LCEL para un flujo declarativo y nativamente asíncrono, descartando cadenas legacy (`RetrievalQA`).
* **Concurrencia Asíncrona:** Los endpoints consumen el LLM y el retriever mediante `await` para no bloquear el Event Loop de FastAPI.
* **Maximal Marginal Relevance (MMR):** El retriever no usa similitud básica, sino MMR (`fetch_k=5`, `k=2`) para garantizar que el contexto inyectado sea semánticamente relevante y diverso.
* **Prompt Engineering Avanzado:** Se utilizaron delimitadores XML (`<contexto>`, `<pregunta>`) y reglas de operación estrictas para anular el riesgo de alucinaciones.

## Prerrequisitos
1. Python 3.10+
2. [Ollama](https://ollama.com/) instalado.
3. Descargar el modelo Mistral en Ollama ejecutando: `ollama run mistral`

## Instalación y Configuración

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/JuanDiegoMontero/itstk-rag-api.git
   cd itstk_rag_api
   ```

2. Crear y activar el entorno virtual:
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```

3. Instalar las dependencias:
   ```bash
    pip install -r requirements.txt
   ```

## Ejecución del Proyecto

1. Ingesta de Datos (Crear Base Vectorial)

   ```bash
   python -m scripts.load_documents
   ```
2. Levantar la API

   ```bash
   uvicorn app.main:app --reload
   ```

La documentación interactiva (Swagger) estará disponible en: http://localhost:8000/docs

## Pruebas Unitarias

El proyecto cuenta con pruebas de integración y validación (Happy path, Out-of-context, Validation Error). Para ejecutarlas:
    ```
   pytest -v
    ```
