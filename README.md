# Benchmark en vivo: evaluación metalingüística y estilometría de LLMs

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)

Aplicación interactiva en **Python + Streamlit** diseñada con propósitos pedagógicos para clases y experimentos de lingüística computacional. Permite comparar en tiempo real las salidas de **dos modelos de lenguaje (LLMs)** copiando y pegando sus textos lado a lado, ejecutando una evaluación 100 % determinista en Python sin depender de otro LLM como juez.

## ¿Qué evalúa? (Activables de forma independiente)

1. **🧪 Prueba 1 — Test metalingüístico (opción múltiple Q1–Q4, con dos sub-bancos seleccionables):**
   - **1A — Gramática normativa (RAE / ASALE):** evaluación normativa (0 a 4 puntos) sobre concordancia de colectivos partitivos con atributo, impersonales con *se* y acusativo animado, queísmo/dequeísmo y gerundio de posterioridad.
   - **1B — Gramática formal y agramaticalidades (`*`):** evaluación descriptiva no normativa sobre agramaticalidad estructural (`*` por extracción de isla interrogativa-*wh*) frente a variación no normativa, inacusatividad vs. inergatividad en cláusulas absolutas de participio, Principio C de la teoría del ligamiento (mando-c) y alcance sintáctico de términos de polaridad negativa.
2. **✍️ Prueba 2 — Auditoría estilística en prosa (reframing + dos puntos):**
   - **Reframing asindético y adversativo (*«No es X, es Y»* / *«No era solo X, sino Y»*):** Detección mediante expresiones regulares de estructuras binarias adversativas yuxtapuestas por coma o punto y coma.
   - **Yuxtaposición explicativo-desarrolladora con dos puntos (`[Oración con verbo] : [Oración con verbo conjugado (forma finita)]`):** Detección morfosintáctica con **spaCy (`es_core_news_lg`)** que contabiliza únicamente las oraciones donde el segmento posterior a `:` supera los 3 tokens y contiene al menos un verbo conjugado en forma finita (`VERB`/`AUX` con `VerbForm=Fin`), descartando listas, aposiciones nominales y segmentos con solo infinitivos o participios.
   - **Densidad normalizada cada 500 palabras:**
     $$\text{Densidad} = \left(\frac{\text{Ocurrencias}}{\text{Palabras totales}}\right) \times 500$$
3. **🔎 Prueba 3 — Detección y corrección de errores en textos largos (~1000 palabras):**
   Evaluación uno a uno contra clave oficial (*ground truth*) sobre un banco de 3 textos extensos con errores **claros** (ortografía, homófonos, *haber* impersonal pluralizado) y **sutiles** (coma entre sujeto y verbo, dequeísmo, queísmo, quesuismo, gerundio de posterioridad o especificativo, paronimia), midiendo cobertura (*recall*), errores **bien corregidos** (la corrección propuesta debe eliminar el error y coincidir con una forma aceptada), precisión aproximada e ítems sin correspondencia. Acepta la lista en el formato pedido, tablas Markdown, bloques `Original: … / Corrección: …` o el texto completo corregido (comparado párrafo por párrafo con el original).
3B. **🧾 Prueba 3B — Corrección de un texto breve evaluada por ediciones (metodología CoNLL-2014 / BEA-2019):**
   El modelo devuelve el **texto completo corregido** y Python lo alinea palabra por palabra con el original (`correccion_texto.py`). Cada error plantado queda *bien corregido*, *mal corregido* u *omitido*; cualquier otro cambio cuenta como **sobrecorrección**, y el texto incluye **trampas** (construcciones correctas que suelen corregirse de más, como «solo» sin tilde o «darse cuenta de que»). Métricas: precisión, cobertura y **F0.5**, que pondera la precisión el doble que la cobertura. Hay tres textos de dificultad creciente: uno breve (~170 palabras, 8 errores, 4 trampas), uno intermedio (~450 palabras, 18 errores con mayoría de sutiles y 10 trampas, entre ellas un americanismo válido) y uno largo (~1000 palabras, 45 errores —26 sutiles: paronimias, quesuismo, *si habría*, *deber de* de obligación, concordancias con sujeto pospuesto o de infinitivo— y 19 trampas, como «guio», «ex secretario general», «la mayoría de los vecinos apoyaron» o «recién» ante verbo).
4. **🧉 Prueba 4 — Reconocimiento de variedades del español (creada por estudiantes):**
   Evaluación dialectal, léxica, fraseológica y prosódica (0 a 4 puntos) sobre variedades del español de Argentina (variante rioplatense/platense y tonada cordobesa).
5. **🛠️ Prueba 5 — Constructor libre de test en vivo para crear en clase:**
   Permite que el docente y los estudiantes diseñen en el aula entre 1 y 5 preguntas propias (de opción múltiple `A/B/C/D` con una o varias respuestas correctas, o de respuesta breve / reescritura con expresión clave), definan la respuesta esperada, copien el prompt autogenerado (`Prompt 6`) y evalúen determinísticamente a ambos modelos desde cualquier navegador.

## Cómo ejecutarlo online gratis (Streamlit Community Cloud)

1. Ingresa a **[share.streamlit.io](https://share.streamlit.io/)** con tu cuenta de GitHub.
2. Haz clic en **"Create app"** (o **"New app"**).
3. Selecciona:
   - **Repository:** `marianovilar/benchmark-ling-prueba`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Haz clic en **"Deploy!"**.

## Versión web estática (GitHub Pages, sin servidor)

`index.html` publica la misma `app.py` como página estática: [stlite](https://github.com/whitphx/stlite) ejecuta Streamlit dentro del navegador con Pyodide. Sirve para redes que bloquean `*.streamlit.app` (por ejemplo, con inspección SSL de Cisco Umbrella).

1. En GitHub: **Settings → Pages → Build and deployment → Source: Deploy from a branch**.
2. Elegí la rama `main` y la carpeta `/ (root)`, y guardá.
3. La app queda en `https://marianovilar.github.io/benchmark-ling-prueba/`. La primera carga tarda unos segundos porque el navegador descarga Python.

Diferencia con la versión de Streamlit: spaCy (`es_core_news_lg`, ~550 MB) no puede ejecutarse en el navegador, así que la detección de dos puntos de la Prueba 2 usa `nlp_ligero.py`, un etiquetador heurístico basado en un léxico de formas verbales (`datos/lexico_verbal_es.json.gz`, generado con `scripts/generar_lexico_verbal.py` desde `spacy-lookups-data`, licencia MIT) y reglas de contexto. Es menos preciso que spaCy con homógrafos verbo/sustantivo («cuenta», «costo»). El resto de las pruebas funciona igual.

## Ejecución local y tests

```bash
pip install -r requirements.txt
pytest
streamlit run app.py
```

En Windows también puedes abrir la aplicación haciendo doble clic sobre `iniciar_benchmark.bat`.
