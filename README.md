# Benchmark en Vivo: Estilometría y Evaluación Metalingüística de LLMs

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)

Aplicación interactiva en **Python + Streamlit** diseñada para clases y experimentos de lingüística computacional. Permite comparar en tiempo real las salidas de **dos modelos de lenguaje (LLMs)** copiando y pegando sus textos lado a lado, sin depender de otro LLM como juez.

## ¿Qué evalúa? (Activables de forma independiente)

1. **🧪 Prueba 1 — Test Metalingüístico (Multiple Choice Q1–Q4):**
   Evaluación normativa (0 a 4 puntos) sobre concordancia de colectivos partitivos, impersonales con *se* y acusativo animado, queísmo/dequeísmo y gerundio de posterioridad.
2. **✍️ Prueba 2 — Auditoría Estilística en Prosa (Reframing + Dos Puntos):**
   - **Reframing asindético (*«No es X, es Y»*):** Detección mediante expresiones regulares de estructuras binarias adversativas yuxtapuestas por coma o punto y coma (*«No es X, es Y»*, *«No se trata de X, se trata de Y»*).
   - **Yuxtaposición explicativo-desarrolladora con dos puntos (`[Oración] : [Oración con verbo]`):** Detección morfosintáctica con **spaCy (`es_core_news_lg`)** que contabiliza únicamente las oraciones donde el segmento posterior a `:` supera los 3 tokens y contiene al menos un verbo conjugado o auxiliar (`VERB`/`AUX`), descartando listas y aposiciones nominales.
   - **Densidad normalizada cada 500 palabras:**
     $$\text{Densidad} = \left(\frac{\text{Ocurrencias}}{\text{Palabras totales}}\right) \times 500$$

## Cómo ejecutarlo online gratis (Streamlit Community Cloud)

1. Ingresa a **[share.streamlit.io](https://share.streamlit.io/)** con tu cuenta de GitHub.
2. Haz clic en **"Create app"** (o **"New app"**).
3. Selecciona:
   - **Repository:** `marianovilar/benchmark-ling-prueba`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Haz clic en **"Deploy!"**. Te generará un link público permanente para compartir con los alumnos en clase.

## Ejecución local

```bash
pip install -r requirements.txt
streamlit run app.py
```

En Windows también puedes abrir la aplicación haciendo doble clic sobre `iniciar_benchmark.bat`.
