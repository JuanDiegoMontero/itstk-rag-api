# API REST - RAG para Consulta de SLA (ITSTK S.A.S.)

Esta es una API REST construida con **FastAPI** y **LangChain** que implementa un sistema RAG (Retrieval-Augmented Generation) 100% local. Utiliza **Ollama** (Mistral) como LLM y **ChromaDB** como base de datos vectorial.

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
   git clone <URL_DEL_REPOSITORIO>
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

4. Configurar las variables de entorno:
    Renombra el archivo .env.example a .env y verifica que los puertos coincidan con tu instalación local de Ollama.

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
