# Benchmark en vivo: evaluación metalingüística y estilometría de LLMs

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)

Aplicación interactiva en **Python + Streamlit** diseñada con propósitos pedagógicos para clases y experimentos de lingüística computacional. Permite comparar en tiempo real las salidas de **dos modelos de lenguaje (LLMs)** copiando y pegando sus textos lado a lado, ejecutando una evaluación 100 % determinista en Python sin depender de otro LLM como juez.

## ¿Qué evalúa? (Activables de forma independiente)

1. **🧪 Prueba 1 — Test metalingüístico (opción múltiple Q1–Q4):**
   Evaluación normativa (0 a 4 puntos) sobre concordancia de colectivos partitivos con atributo, impersonales con *se* y acusativo animado, queísmo/dequeísmo y gerundio de posterioridad.
2. **✍️ Prueba 2 — Auditoría estilística en prosa (reframing + dos puntos):**
   - **Reframing asindético y adversativo (*«No es X, es Y»* / *«No era solo X, sino Y»*):** Detección mediante expresiones regulares de estructuras binarias adversativas yuxtapuestas por coma o punto y coma.
   - **Yuxtaposición explicativo-desarrolladora con dos puntos (`[Oración con verbo] : [Oración con verbo conjugado (forma finita)]`):** Detección morfosintáctica con **spaCy (`es_core_news_lg`)** que contabiliza únicamente las oraciones donde el segmento posterior a `:` supera los 3 tokens y contiene al menos un verbo conjugado en forma finita (`VERB`/`AUX` con `VerbForm=Fin`), descartando listas, aposiciones nominales y segmentos con solo infinitivos o participios.
   - **Densidad normalizada cada 500 palabras:**
     $$\text{Densidad} = \left(\frac{\text{Ocurrencias}}{\text{Palabras totales}}\right) \times 500$$
3. **🔎 Prueba 3 — Detección y corrección de errores en textos largos (~1000 palabras):**
   Evaluación uno a uno contra clave oficial (*ground truth*) sobre un banco de 3 textos extensos con errores **claros** (ortografía, homófonos, *haber* impersonal pluralizado) y **sutiles** (coma entre sujeto y verbo, dequeísmo, queísmo, quesuismo, gerundio de posterioridad o especificativo, paronimia), midiendo cobertura (*recall*), precisión aproximada e ítems sin correspondencia.

## Cómo ejecutarlo online gratis (Streamlit Community Cloud)

1. Ingresa a **[share.streamlit.io](https://share.streamlit.io/)** con tu cuenta de GitHub.
2. Haz clic en **"Create app"** (o **"New app"**).
3. Selecciona:
   - **Repository:** `marianovilar/benchmark-ling-prueba`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Haz clic en **"Deploy!"**.

## Ejecución local y tests

```bash
pip install -r requirements.txt
pytest
streamlit run app.py
```

En Windows también puedes abrir la aplicación haciendo doble clic sobre `iniciar_benchmark.bat`.
