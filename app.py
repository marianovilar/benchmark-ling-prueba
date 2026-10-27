"""
app.py - Interfaz interactiva en Streamlit para comparar 2 modelos en vivo (ideal para clases y GitHub).

Permite activar o desactivar libremente cualquiera de las tres pruebas (todas basadas en clave
de respuestas correcta / ground truth determinista, sin usar un LLM como juez):
- Prueba 1: evaluación metalingüística objetiva (opción múltiple Q1-Q4 contra ground truth).
- Prueba 2: auditoría estilística en prosa (reframing asindético/adversativo + dos puntos oracionales con spaCy).
- Prueba 3: detección y corrección de errores de redacción y ortografía (sobre 3 textos de ~1000 palabras
  con clave de errores claros y sutiles, o un texto propio con su clave de errores).

Ejecución:
    streamlit run app.py
"""

from __future__ import annotations

import html
import importlib
import re

from typing import TYPE_CHECKING

import pandas as pd
import streamlit as st

import textos_errores
import nlp_ligero
import auditor

# Recargar módulos para evitar que Streamlit Cloud conserve versiones en caché tras un git push
importlib.reload(textos_errores)
importlib.reload(nlp_ligero)
importlib.reload(auditor)

if TYPE_CHECKING:
    from spacy.language import Language

from auditor import (
    GROUND_TRUTH_META,
    GROUND_TRUTH_META_FORMAL,
    GROUND_TRUTH_VARIEDADES,
    META_DEMO_MODELO_A,
    META_DEMO_MODELO_B,
    META_FORMAL_DEMO_MODELO_A,
    META_FORMAL_DEMO_MODELO_B,
    PROMPTS_SUGERIDOS,
    SPACY_MODEL_NAME,
    TEXTO_DEMO_MODELO_A,
    TEXTO_DEMO_MODELO_B,
    VARIEDADES_DEMO_MODELO_A,
    VARIEDADES_DEMO_MODELO_B,
    CustomQuestionSpec,
    ModelAuditResult,
    build_custom_test_prompt,
    compare_models_in_memory,
    format_detailed_audit_log,
    get_metalinguistic_prompt_text,
    get_variedades_prompt_text,
    load_spacy_model,
)
from textos_errores import (
    BANCO_TEXTOS_ERRORES,
    DEMO_RESPUESTA_ERRORES_MODELO_A,
    DEMO_RESPUESTA_ERRORES_MODELO_B,
    PlantedError,
    build_error_detection_prompt,
)

# ============================================================================
# CONFIGURACIÓN DE PÁGINA Y CACHÉ DE SPACY
# ============================================================================

st.set_page_config(
    page_title="Benchmark en vivo: evaluación metalingüística y estilometría de LLMs",
    page_icon="📐",
    layout="wide",
)

# Forzar salto de línea automático en bloques de prompts para evitar desplazamiento horizontal
st.markdown(
    """
    <style>
    div[data-testid="stCodeBlock"] pre,
    div[data-testid="stCodeBlock"] code,
    div[data-testid="stCode"] pre,
    div[data-testid="stCode"] code {
        white-space: pre-wrap !important;
        word-break: break-word !important;
        overflow-x: hidden !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Cargando modelo lingüístico de spaCy...")
def get_cached_nlp() -> tuple[Language, str]:
    """Carga y guarda en caché el modelo de spaCy solo cuando la prueba de estilometría está activa."""
    return load_spacy_model(SPACY_MODEL_NAME)


# ============================================================================
# FUNCIONES DE RESALTADO HTML PARA EVIDENCIA CUALITATIVA
# ============================================================================


def highlight_reframing_html(sentence: str, fragment: str) -> str:
    """Resalta en amarillo el patrón de reframing dentro de la oración."""
    escaped_sentence = html.escape(sentence)
    escaped_fragment = html.escape(fragment)
    if not escaped_fragment:
        return escaped_sentence
    pattern = re.compile(re.escape(escaped_fragment), re.IGNORECASE)
    return pattern.sub(
        lambda m: (
            '<mark style="background-color: #ffe082; color: #111; '
            'padding: 2px 5px; border-radius: 4px; font-weight: 600;">'
            f"{m.group(0)}</mark>"
        ),
        escaped_sentence,
        count=1,
    )


def highlight_colon_clause_html(seg_izq: str, seg_der: str, verbos_der: list[str]) -> str:
    """Resalta los dos puntos y los verbos conjugados (forma finita) detectados en la segunda proposición."""
    izq_html = html.escape(seg_izq)
    der_html = html.escape(seg_der)

    for verbo in sorted(set(verbos_der), key=len, reverse=True):
        verbo_esc = html.escape(verbo)
        pattern = re.compile(rf"\b{re.escape(verbo_esc)}\b", re.IGNORECASE)
        der_html = pattern.sub(
            lambda m: (
                '<span style="background-color: #b9f6ca; color: #004d40; '
                'padding: 1px 4px; border-radius: 3px; font-weight: 600;" '
                'title="Verbo conjugado (forma finita) detectado por spaCy: '
                '[Oración con verbo] : [Oración con verbo conjugado (forma finita)]">'
                f"{m.group(0)}</span>"
            ),
            der_html,
        )

    colon_badge = (
        '<span style="background-color: #ff8a80; color: #fff; '
        'padding: 1px 6px; margin: 0 4px; border-radius: 4px; font-weight: 700;">:</span>'
    )
    return f"{izq_html}{colon_badge}<span style='text-decoration: underline dotted #555;'>{der_html}</span>"


def parse_custom_ground_truth_errors(raw_gt_lines: str) -> tuple[PlantedError, ...]:
    """
    Convierte una lista manual de errores esperados escrita por el docente
    (formato: `Claro | error -> corrección` o simplemente `error -> corrección` por línea)
    en objetos `PlantedError` con `anclas` y `correcciones`.
    """
    errors: list[PlantedError] = []
    for idx, line in enumerate(raw_gt_lines.strip().splitlines(), start=1):
        clean = line.strip()
        if not clean:
            continue
        categoria = "Claro"
        if "|" in clean:
            cat_part, rest = [p.strip() for p in clean.split("|", 1)]
            if "sutil" in cat_part.lower():
                categoria = "Sutil"
            clean = rest

        if "->" in clean:
            err_frag, corr_frag = [p.strip(" \"'«»") for p in clean.split("->", 1)]
        else:
            err_frag = clean.strip(" \"'«»")
            corr_frag = "(Ver corrección)"

        if err_frag:
            correcciones = (corr_frag,) if (corr_frag and corr_frag != "(Ver corrección)") else ()
            errors.append(
                PlantedError(
                    orden=idx,
                    categoria=categoria,
                    fenomeno="Error definido por el usuario",
                    fragmento_erroneo=err_frag,
                    correccion_esperada=corr_frag,
                    anclas=(err_frag,),
                    correcciones=correcciones,
                )
            )
    return tuple(errors)


# ============================================================================
# ESTADO INICIAL DE LA SESIÓN (DATOS DE DEMOSTRACIÓN)
# ============================================================================


def init_session_state() -> None:
    if "nombre_a" not in st.session_state:
        st.session_state["nombre_a"] = ""
    if "nombre_b" not in st.session_state:
        st.session_state["nombre_b"] = ""
    if "meta_input_a" not in st.session_state:
        st.session_state["meta_input_a"] = ""
    if "meta_input_b" not in st.session_state:
        st.session_state["meta_input_b"] = ""
    if "texto_a" not in st.session_state:
        st.session_state["texto_a"] = ""
    if "texto_b" not in st.session_state:
        st.session_state["texto_b"] = ""
    if "error_input_a" not in st.session_state:
        st.session_state["error_input_a"] = ""
    if "error_input_b" not in st.session_state:
        st.session_state["error_input_b"] = ""
    if "variedades_input_a" not in st.session_state:
        st.session_state["variedades_input_a"] = ""
    if "variedades_input_b" not in st.session_state:
        st.session_state["variedades_input_b"] = ""
    if "custom_input_a" not in st.session_state:
        st.session_state["custom_input_a"] = ""
    if "custom_input_b" not in st.session_state:
        st.session_state["custom_input_b"] = ""
    if "run_meta" not in st.session_state:
        st.session_state["run_meta"] = True
    if "run_style" not in st.session_state:
        st.session_state["run_style"] = False
    if "run_errors" not in st.session_state:
        st.session_state["run_errors"] = False
    if "run_variedades" not in st.session_state:
        st.session_state["run_variedades"] = False
    if "run_custom" not in st.session_state:
        st.session_state["run_custom"] = False
    if "p4_num_q" not in st.session_state:
        st.session_state["p4_num_q"] = 1
    if "p4_q1_fen" not in st.session_state:
        st.session_state["p4_q1_fen"] = "Régimen preposicional (dequeísmo / queísmo)"
    if "p4_q1_enu" not in st.session_state:
        st.session_state["p4_q1_enu"] = "Seleccione la opción gramaticalmente correcta según la norma culta:"
    if "p4_q1_a" not in st.session_state:
        st.session_state["p4_q1_a"] = "Me alegro de que hayas aprobado el examen final."
    if "p4_q1_b" not in st.session_state:
        st.session_state["p4_q1_b"] = "Me alegro que hayas aprobado el examen final."
    if "p4_q1_c" not in st.session_state:
        st.session_state["p4_q1_c"] = "Pienso de que la hipótesis es correcta."
    if "p4_q1_d" not in st.session_state:
        st.session_state["p4_q1_d"] = "A y B son igualmente válidas en el registro formal."
    if "p4_q1_corr" not in st.session_state or isinstance(st.session_state.get("p4_q1_corr"), str):
        st.session_state["p4_q1_corr"] = ["A"]
    if "p4_q1_frase" not in st.session_state:
        st.session_state["p4_q1_frase"] = "Me alegro de que hayas aprobado"


def load_demo_texts() -> None:
    st.session_state["nombre_a"] = "Modelo A (ej. GPT-4o)"
    st.session_state["nombre_b"] = "Modelo B (ej. modelo pequeño / gratis)"
    if "1B" in str(st.session_state.get("p1_subbanco", "")):
        st.session_state["meta_input_a"] = META_FORMAL_DEMO_MODELO_A
        st.session_state["meta_input_b"] = META_FORMAL_DEMO_MODELO_B
    else:
        st.session_state["meta_input_a"] = META_DEMO_MODELO_A
        st.session_state["meta_input_b"] = META_DEMO_MODELO_B
    st.session_state["texto_a"] = TEXTO_DEMO_MODELO_A
    st.session_state["texto_b"] = TEXTO_DEMO_MODELO_B
    st.session_state["error_input_a"] = DEMO_RESPUESTA_ERRORES_MODELO_A
    st.session_state["error_input_b"] = DEMO_RESPUESTA_ERRORES_MODELO_B
    st.session_state["variedades_input_a"] = VARIEDADES_DEMO_MODELO_A
    st.session_state["variedades_input_b"] = VARIEDADES_DEMO_MODELO_B
    st.session_state["custom_input_a"] = '{"q1": "A"}'
    st.session_state["custom_input_b"] = '{"q1": "D"}'


def clear_texts() -> None:
    st.session_state["nombre_a"] = ""
    st.session_state["nombre_b"] = ""
    st.session_state["meta_input_a"] = ""
    st.session_state["meta_input_b"] = ""
    st.session_state["texto_a"] = ""
    st.session_state["texto_b"] = ""
    st.session_state["error_input_a"] = ""
    st.session_state["error_input_b"] = ""
    st.session_state["variedades_input_a"] = ""
    st.session_state["variedades_input_b"] = ""
    st.session_state["custom_input_a"] = ""
    st.session_state["custom_input_b"] = ""


# ============================================================================
# COMPONENTES DE LA INTERFAZ
# ============================================================================


def render_about_tab() -> None:
    """Renderiza la pestaña explicativa sobre el funcionamiento metodológico y créditos del sitio."""
    st.subheader("ℹ️ Cómo funciona este sitio")
    st.markdown(
        """
        Este sitio fue armado por **Mariano Vilar** con propósitos pedagógicos utilizando **Google Gemini**.
        Su objetivo es ofrecer un entorno experimental transparente para analizar y comparar el desempeño lingüístico,
        gramatical y estilístico de distintos modelos de lenguaje (LLMs) en español.

        ---

        ### 1. Evaluación determinista en Python (sin otro LLM como juez)
        A diferencia de otros benchmarks que emplean un modelo de lenguaje para calificar las respuestas de otro (*LLM-as-a-judge*), **todas las evaluaciones de esta plataforma se ejecutan exclusivamente mediante código Python determinista y auditable**:
        * **Prueba 1 (test metalingüístico de opción múltiple, con dos sub-bancos seleccionables):**
          1. **1A — Gramática normativa (RAE / ASALE):** compara de forma estricta las respuestas (`A`, `B`, `C` o `D`) contra una clave fija basada en la *Nueva gramática de la lengua española* (concordancia con partitivos, impersonales con *se*, queísmo/dequeísmo y gerundio de posterioridad).
          2. **1B — Gramática formal y agramaticalidades (`*`):** evalúa fenómenos de sintaxis descriptiva y formal no prescriptiva donde los LLMs suelen fallar: distinción entre variación no normativa vs. agramaticalidad estructural (`*` por extracción de isla interrogativa-*wh*), inacusatividad vs. inergatividad en cláusulas absolutas de participio, Principio C de la teoría del ligamiento (mando-c) y alcance sintáctico de términos de polaridad negativa.
        * **Prueba 2 (auditoría estilística en prosa):** contabiliza y normaliza cada 500 palabras la densidad de dos tics sintáctico-retóricos frecuentes en textos generados por IA:
          1. *Reframing asindético y adversativo* (*«No es X, es Y»*, *«No se trata de X, se trata de Y»*, *«No era solo X, sino Y»*), detectado mediante expresiones regulares en Python.
          2. *Yuxtaposición explicativo-desarrolladora con dos puntos* (`[Oración con verbo] : [Oración con verbo conjugado (forma finita)]`), detectada mediante segmentación oracional y etiquetado morfosintáctico (*POS-tagging*) con la biblioteca **`spaCy` (`es_core_news_lg`)**, descartando listas, aposiciones nominales y segmentos con solo formas no personales.
        * **Prueba 3 (detección y corrección de errores en textos largos):** verifica mediante reglas de coincidencia uno a uno en Python cuántos de los errores plantados en el texto de referencia (tanto errores **claros** de ortografía/concordancia como errores **sutiles** de puntuación, queísmo, dequeísmo, quesuismo, gerundios e impropiedades léxicas) fueron efectivamente identificados por cada modelo.
        * **Prueba 4 (reconocimiento de variedades del español — creada por estudiantes):** evalúa el conocimiento dialectal, léxico, fraseológico y prosódico sobre variedades del español de Argentina (variante rioplatense/platense y tonada cordobesa).
        * **Prueba 5 (constructor libre en vivo para crear en clase):** permite que el docente y los estudiantes diseñen en el aula un test lingüístico propio (de opción múltiple `A/B/C/D` con una o más respuestas válidas, o de respuesta breve / reescritura con expresión clave), definan la respuesta correcta esperada, copien el prompt autogenerado y califiquen determinísticamente a ambos modelos desde cualquier navegador.

        ---

        ### 2. Uso manual desde el navegador o la PC
        En su versión actual, **este sitio no se conecta automáticamente por API con los modelos de lenguaje**. Para realizar una prueba comparativa, el flujo de trabajo requiere que el usuario:
        1. Seleccione las pruebas que desea realizar y **copie los prompts correspondientes** desde las ventanas desplegables de esta página.
        2. **Acceda manualmente en su navegador web o en su PC** a los sitios o plataformas de los modelos que desee poner a prueba (por ejemplo, ChatGPT, Claude, Gemini, o plataformas gratuitas con modelos abiertos y pequeños como LMArena, Duck.ai, HuggingChat, Mistral, Qwen o DeepSeek).
        3. **Pegue las respuestas** generadas por cada uno de los dos modelos en las cajas de texto de esta interfaz y presione **«▶️ Ejecutar evaluación y comparar modelos»**.
        """
    )


def render_header_and_prompts() -> None:
    st.markdown(
        "Compara dos modelos de lenguaje en español pegando sus respuestas lado a lado. "
        "Puedes activar o desactivar libremente cualquiera de las **cinco pruebas** según el objetivo de la clase. "
        "**Este sitio requiere entrar por separado a las páginas web de los modelos que quieras probar, "
        "copiar los prompts de cada prueba y pegar las respuestas obtenidas en las ventanas de texto correspondientes.**"
    )

    col_guide, col_links = st.columns(2)

    with col_guide:
        with st.expander("📖 ¿Qué evalúa cada prueba?", expanded=False):
            st.markdown(
                """
                1. **🧪 Prueba 1 — Test metalingüístico (opción múltiple Q1–Q4):**
                   Permite elegir entre dos sub-bancos (0 a 4 puntos contra clave fija):
                   - **1A — Gramática normativa (RAE / ASALE):** concordancia con partitivos, impersonales con *se*, queísmo/dequeísmo y gerundio de posterioridad.
                   - **1B — Gramática formal y agramaticalidades (`*`):** sintaxis descriptiva no normativa (islas interrogativas-*wh*, inacusatividad vs. inergatividad en cláusulas absolutas, Principio C de ligamiento por mando-c y alcance de polaridad negativa).
                2. **✍️ Prueba 2 — Auditoría estilística en prosa (reframing + dos puntos):**
                   - **Tic 1 — Reframing asindético y adversativo (*«No es X, es Y»* / *«No era solo X, sino Y»*):** Fórmulas binarias donde se niega un término para reformularlo tras coma o punto y coma.
                   - **Tic 2 — Dos puntos oracionales (`[Oración con verbo] : [Oración con verbo conjugado (forma finita)]`):** Detecta con **spaCy** oraciones donde tras `:` hay más de 3 palabras y al menos un verbo conjugado en forma finita, descartando listas, aposiciones nominales e infinitivos/participios sueltos.
                3. **🔎 Prueba 3 — Detección y corrección de errores en textos largos (~1000 palabras):**
                   Compara la lista de errores que encuentra el modelo contra una **clave oficial de errores plantados** (divididos en **claros** y **sutiles**).
                4. **🧉 Prueba 4 — Reconocimiento de variedades del español (creada por alumnos):**
                   Evalúa el conocimiento léxico, fraseológico y prosódico sobre variedades del español de Argentina (variante rioplatense/platense y tonada cordobesa).
                5. **🛠️ Prueba 5 — Constructor libre de test en vivo para crear en clase:**
                   Permite diseñar junto con los estudiantes un test lingüístico sencillo (de opción múltiple `A/B/C/D` o de respuesta breve / reescritura), fijar la respuesta correcta, autogenerar el prompt listo para copiar y calificar a ambos modelos en tiempo real.
                """
            )

    with col_links:
        with st.expander("🌐 Modelos gratuitos y alternativos para probar en clase", expanded=False):
            st.markdown(
                """
                Además de los modelos comerciales conocidos (ChatGPT, Gemini, Claude), puedes usar estos sitios **gratuitos** para probar modelos abiertos, regionales, pequeños (7B–9B) o menos conocidos:

                * 🌎 **[CopuChat — LatamGPT](https://copuchat.latamgpt.org/)** — Chat público del proyecto abierto latinoamericano *Latam-GPT* (impulsado por CENIA), ideal para evaluar un modelo entrenado con foco en el español de América Latina.
                * 🧪 **[Ai2 Playground — AllenAI](https://playground.allenai.org/)** *(Recomendado)* — Plataforma abierta y sin registro del Allen Institute for AI; permite probar modelos académicos 100 % abiertos (`OLMo 2`, `Llama-Tülu 3`, `Molmo`) de distintos tamaños (7B, 13B, 32B, 70B), ideales para contrastar fallos gramaticales en español.
                * 🏟️ **[LMArena (Chatbot Arena)](https://lmarena.ai/)** *(Recomendado)* — Usa la pestaña *Side-by-Side* o *Direct Chat* para probar gratis y sin cuenta docenas de modelos grandes y pequeños (`Llama`, `Gemma`, `Qwen`, `Mistral`, `Command-R`, `Phi`, `Yi`, etc.).
                * 🦆 **[Duck.ai (DuckDuckGo AI Chat)](https://duck.ai/)** — Gratis y sin registro. Permite alternar entre modelos ligeros (`Mistral Small`, `Llama 3`, `Claude Haiku`, `GPT-4o mini`).
                * 🇫🇷 **[Le Chat — Mistral AI](https://chat.mistral.ai/)** — Modelo europeo gratuito en el navegador.
                * 🐉 **[Qwen Chat (Alibaba)](https://chat.qwen.ai/)** — Permite elegir distintas variantes de tamaño de la familia `Qwen` gratis.
                * 🐋 **[DeepSeek Chat](https://chat.deepseek.com/)** — Modelo gratuito alternativo útil para comparar tics retóricos en español.
                * 🤗 **[HuggingChat](https://huggingface.co/chat/)** — Acceso a modelos abiertos variados (`Qwen`, `Cohere Command R`, `Llama`, `SmolLM`), aunque requiere iniciar sesión y tiene un límite gratuito muy restringido.
                """
            )


def render_error_benchmark_selector() -> tuple[str, tuple[PlantedError, ...] | None]:
    """
    Renderiza el selector de los 3 textos de ~1000 palabras con errores (o carga de texto propio
    con su clave de errores) y muestra la ventana separada con el prompt estructurado listo para copiar.
    """
    st.markdown("#### 🔎 Configuración de la prueba 3: texto con errores (~1000 palabras) y clave de corrección")

    opciones_selector = {
        "texto_1": BANCO_TEXTOS_ERRORES["texto_1"].titulo,
        "texto_2": BANCO_TEXTOS_ERRORES["texto_2"].titulo,
        "texto_3": BANCO_TEXTOS_ERRORES["texto_3"].titulo,
        "custom": "📂 Usar otro texto propio + definir su lista de errores esperados",
    }

    seleccion = st.selectbox(
        "Elige uno de los 3 textos con clave de errores ya identificada (o carga uno propio):",
        options=list(opciones_selector.keys()),
        format_func=lambda k: opciones_selector[k],
    )

    if seleccion != "custom":
        bench_obj = BANCO_TEXTOS_ERRORES[seleccion]
        st.caption(bench_obj.descripcion)
        texto_activo = bench_obj.texto
        errores_plantados: tuple[PlantedError, ...] | None = bench_obj.errores

        with st.expander(
            f"📋 Prompt 4: búsqueda estructurada de errores (incluye el {bench_obj.id.replace('_', ' ')} completo para copiar)",
            expanded=False,
        ):
            st.caption("Haz clic en el icono de copiar (esquina superior derecha del recuadro) y pégalo en ambos modelos:")
            st.code(build_error_detection_prompt(texto_activo), language=None, wrap_lines=True)

        with st.expander(
            f"🔑 Ver clave oficial de los {len(bench_obj.errores)} errores ocultos en este texto (ground truth)",
            expanded=False,
        ):
            rows_gt = [
                {
                    "Orden": e.orden,
                    "Dificultad": "🟢 Claro" if e.categoria.lower() == "claro" else "🟣 Sutil",
                    "Fenómeno": e.fenomeno,
                    "Error en el texto": e.fragmento_erroneo,
                    "Corrección normativa": e.correccion_esperada,
                }
                for e in bench_obj.errores
            ]
            st.dataframe(pd.DataFrame(rows_gt), use_container_width=True, hide_index=True)
    else:
        up_col, gt_col = st.columns(2)
        with up_col:
            uploaded_file = st.file_uploader(
                "Sube un archivo .txt con tu texto con errores (opcional):",
                type=["txt"],
            )
            default_custom_text = ""
            if uploaded_file is not None:
                raw_bytes = uploaded_file.read()
                try:
                    default_custom_text = raw_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    default_custom_text = raw_bytes.decode("latin-1", errors="ignore")

            texto_activo = st.text_area(
                "Texto con errores a evaluar:",
                value=default_custom_text,
                height=180,
                placeholder="Pega aquí tu propio texto con errores...",
            )
        with gt_col:
            raw_custom_gt = st.text_area(
                "Clave de errores esperados en tu texto (1 por línea: `Claro | error -> corrección` o `Sutil | error -> corrección`):",
                value="Claro | atravez -> a través\nSutil | pensaban de que -> pensaban que",
                height=250,
                help="Para que el benchmark pueda calificar automáticamente sin usar otro LLM como juez, indica aquí los errores que plantaste en tu texto.",
            )
            errores_plantados = parse_custom_ground_truth_errors(raw_custom_gt)

        with st.expander("📋 Prompt 4: búsqueda estructurada de errores (con tu texto propio)", expanded=True):
            if texto_activo.strip():
                st.caption("Haz clic en el icono de copiar (esquina superior derecha del recuadro):")
                st.code(build_error_detection_prompt(texto_activo), language=None, wrap_lines=True)
            else:
                st.info("Sube o pega tu texto arriba para generar automáticamente el prompt estructurado.")

    return texto_activo, errores_plantados


def render_metalinguistic_comparison(
    res_a: ModelAuditResult,
    res_b: ModelAuditResult,
    meta_gt: dict[str, dict] = GROUND_TRUTH_META,
) -> None:
    """Muestra la tabla comparativa pregunta por pregunta del test metalingüístico (Q1-Q4, sub-banco 1A o 1B)."""
    sub_label = (
        "1B: gramática formal y agramaticalidades (*)"
        if meta_gt is GROUND_TRUTH_META_FORMAL
        else "1A: gramática normativa (RAE / ASALE)"
    )
    st.subheader(f"🧪 Desglose de la prueba 1 ({sub_label})")

    rows = []
    for det_a, det_b in zip(res_a.meta.detalles, res_b.meta.detalles):
        q_key = det_a.question_id.lower()
        enunciado = meta_gt.get(q_key, {}).get("enunciado", "")
        icon_a = "✅" if det_a.es_correcta else "❌"
        icon_b = "✅" if det_b.es_correcta else "❌"
        rows.append(
            {
                "Ítem": det_a.question_id,
                "Fenómeno gramatical": det_a.pregunta,
                "Enunciado": enunciado,
                "Clave oficial": det_a.respuesta_esperada,
                f"{res_a.modelo}": f"{icon_a} {det_a.respuesta_modelo}",
                f"{res_b.modelo}": f"{icon_b} {det_b.respuesta_modelo}",
            }
        )

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_error_detection_comparison(res_a: ModelAuditResult, res_b: ModelAuditResult) -> None:
    """Muestra los resultados comparativos de la Prueba 3 (Detección y corrección de errores)."""
    st.subheader("🔎 Desglose de la prueba 3: detección y corrección de errores")

    er_a = res_a.errores_result
    er_b = res_b.errores_result

    if er_a.aviso_formato:
        st.warning(
            f"⚠️ **{res_a.modelo}**: la respuesta no siguió el formato de lista numerada o con flechas (`->`). "
            "Se evaluó cada línea no vacía como un ítem individual."
        )
    if er_b.aviso_formato:
        st.warning(
            f"⚠️ **{res_b.modelo}**: la respuesta no siguió el formato de lista numerada o con flechas (`->`). "
            "Se evaluó cada línea no vacía como un ítem individual."
        )

    if er_a.tiene_ground_truth and er_b.tiene_ground_truth:
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.markdown(f"**{res_a.modelo}**")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric(
                "Cobertura (recall)",
                f"{er_a.detectados_total}/{er_a.total_plantados}",
                f"{er_a.cobertura_porcentaje:.1f}%",
            )
            k2.metric(
                "Precisión aproximada",
                f"{er_a.precision_aprox_porcentaje:.1f}%",
                f"{len(er_a.items_sin_correspondencia)} sin corresp.",
                delta_color="off",
            )
            k3.metric("Errores claros", f"{er_a.claros_detectados}/{er_a.claros_total}")
            k4.metric("Errores sutiles", f"{er_a.sutiles_detectados}/{er_a.sutiles_total}")
        with m_col2:
            st.markdown(f"**{res_b.modelo}**")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric(
                "Cobertura (recall)",
                f"{er_b.detectados_total}/{er_b.total_plantados}",
                f"{er_b.cobertura_porcentaje:.1f}%",
            )
            k2.metric(
                "Precisión aproximada",
                f"{er_b.precision_aprox_porcentaje:.1f}%",
                f"{len(er_b.items_sin_correspondencia)} sin corresp.",
                delta_color="off",
            )
            k3.metric("Errores claros", f"{er_b.claros_detectados}/{er_b.claros_total}")
            k4.metric("Errores sutiles", f"{er_b.sutiles_detectados}/{er_b.sutiles_total}")

        st.caption(
            "Nota sobre **Precisión aproximada** (`ítems con correspondencia / ítems listados`): "
            "algunos ítems sin correspondencia en la clave pueden ser errores o sugerencias de estilo reales no plantados."
        )

        rows_err = []
        for det_a, det_b in zip(er_a.detalles_plantados, er_b.detalles_plantados):
            pe = det_a.error_plantado
            badge_tipo = "🟢 Claro" if pe.categoria.lower() == "claro" else "🟣 Sutil"
            rows_err.append(
                {
                    "Orden": pe.orden,
                    "Dificultad": badge_tipo,
                    "Fenómeno": pe.fenomeno,
                    "Error en el texto": pe.fragmento_erroneo,
                    "Corrección esperada": pe.correccion_esperada,
                    f"{res_a.modelo}": "✅ Detectado" if det_a.detectado else "❌ Omitido",
                    f"Evidencia ({res_a.modelo})": det_a.evidencia_modelo,
                    f"{res_b.modelo}": "✅ Detectado" if det_b.detectado else "❌ Omitido",
                    f"Evidencia ({res_b.modelo})": det_b.evidencia_modelo,
                }
            )
        st.dataframe(pd.DataFrame(rows_err), use_container_width=True, hide_index=True)

        with st.expander("🔍 Ítems sin correspondencia en la clave", expanded=False):
            st.caption(
                "Ítems reportados por cada modelo que no coincidieron con ninguno de los errores plantados en la clave "
                "(pueden ser falsos positivos o sugerencias adicionales):"
            )
            u_left, u_right = st.columns(2)
            with u_left:
                st.markdown(f"**{res_a.modelo}** ({len(er_a.items_sin_correspondencia)} sin correspondencia)")
                if er_a.items_sin_correspondencia:
                    df_un_a = pd.DataFrame(
                        [
                            {
                                "#": it.numero,
                                "Error señalado": it.error_identificado,
                                "Corrección": it.correccion_propuesta,
                                "Explicación": it.explicacion,
                            }
                            for it in er_a.items_sin_correspondencia
                        ]
                    )
                    st.dataframe(df_un_a, use_container_width=True, hide_index=True)
                else:
                    st.success("0 ítems sin correspondencia.")
            with u_right:
                st.markdown(f"**{res_b.modelo}** ({len(er_b.items_sin_correspondencia)} sin correspondencia)")
                if er_b.items_sin_correspondencia:
                    df_un_b = pd.DataFrame(
                        [
                            {
                                "#": it.numero,
                                "Error señalado": it.error_identificado,
                                "Corrección": it.correccion_propuesta,
                                "Explicación": it.explicacion,
                            }
                            for it in er_b.items_sin_correspondencia
                        ]
                    )
                    st.dataframe(df_un_b, use_container_width=True, hide_index=True)
                else:
                    st.success("0 ítems sin correspondencia.")

    with st.expander("📋 Ver tabla de ítems extraídos de la respuesta de cada modelo", expanded=False):
        c_left, c_right = st.columns(2)
        with c_left:
            st.markdown(f"**{res_a.modelo}** ({len(er_a.items_extraidos)} ítems listados)")
            if er_a.items_extraidos:
                df_items_a = pd.DataFrame(
                    [
                        {
                            "#": it.numero,
                            "Error señalado": it.error_identificado,
                            "Corrección": it.correccion_propuesta,
                            "Explicación": it.explicacion,
                        }
                        for it in er_a.items_extraidos
                    ]
                )
                st.dataframe(df_items_a, use_container_width=True, hide_index=True)
            else:
                st.info("No se detectaron ítems en formato de lista para el modelo 1.")
        with c_right:
            st.markdown(f"**{res_b.modelo}** ({len(er_b.items_extraidos)} ítems listados)")
            if er_b.items_extraidos:
                df_items_b = pd.DataFrame(
                    [
                        {
                            "#": it.numero,
                            "Error señalado": it.error_identificado,
                            "Corrección": it.correccion_propuesta,
                            "Explicación": it.explicacion,
                        }
                        for it in er_b.items_extraidos
                    ]
                )
                st.dataframe(df_items_b, use_container_width=True, hide_index=True)
            else:
                st.info("No se detectaron ítems en formato de lista para el modelo 2.")


def render_model_evidence_card(res: ModelAuditResult, show_meta_metric: bool = True) -> None:
    """Muestra las métricas rápidas y las oraciones exactas detectadas para un modelo."""
    st.markdown(f"### 🔎 {res.modelo}")

    if show_meta_metric:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "Puntaje meta",
            f"{res.meta.puntaje_absoluto}/4",
            f"{res.meta.precision_porcentaje:.0f}%",
        )
        c2.metric("Palabras", f"{res.palabras_totales}", f"{res.oraciones_totales} orac.")
        c3.metric(
            "Reframing",
            f"{res.total_reframing}",
            f"{res.densidad_reframing_500w:.2f} / 500w",
            delta_color="inverse",
        )
        c4.metric(
            "Dos puntos orac.",
            f"{res.total_dos_puntos}",
            f"{res.densidad_dos_puntos_500w:.2f} / 500w",
            delta_color="inverse",
        )
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Palabras", f"{res.palabras_totales}", f"{res.oraciones_totales} orac.")
        c2.metric(
            "Reframing",
            f"{res.total_reframing}",
            f"{res.densidad_reframing_500w:.2f} / 500w",
            delta_color="inverse",
        )
        c3.metric(
            "Dos puntos orac.",
            f"{res.total_dos_puntos}",
            f"{res.densidad_dos_puntos_500w:.2f} / 500w",
            delta_color="inverse",
        )

    st.markdown(f"**🟨 Tic 1 — Reframing asindético y adversativo detectado ({res.total_reframing}):**")
    if not res.reframing_matches:
        st.success("Sin ocurrencias de reframing.")
    else:
        for idx, match in enumerate(res.reframing_matches, 1):
            html_sent = highlight_reframing_html(match.oracion_contexto, match.fragmento_coincidente)
            st.markdown(
                f"<div style='padding:10px 12px; background:#f8f9fa; border-left:4px solid #ffb300; "
                f"border-radius:4px; margin-bottom:8px; color:#1f2937; font-size:0.95rem;'>"
                f"<strong>{idx}.</strong> {html_sent}</div>",
                unsafe_allow_html=True,
            )

    st.markdown(
        f"**🟥 Tic 2 — Dos puntos oracionales (`[Oración con verbo] : [Oración con verbo conjugado (forma finita)]`) "
        f"detectados ({res.total_dos_puntos}):**"
    )
    if not res.dos_puntos_matches:
        st.success("Sin yuxtaposiciones oracionales con dos puntos.")
    else:
        for idx, match in enumerate(res.dos_puntos_matches, 1):
            html_colon = highlight_colon_clause_html(
                match.segmento_izquierdo,
                match.segmento_derecho,
                match.verbos_detectados_der,
            )
            verbos_badge = ", ".join(f"`{v}`" for v in match.verbos_detectados_der)
            st.markdown(
                f"<div style='padding:10px 12px; background:#f8f9fa; border-left:4px solid #e53935; "
                f"border-radius:4px; margin-bottom:4px; color:#1f2937; font-size:0.95rem;'>"
                f"<strong>{idx}.</strong> {html_colon}</div>",
                unsafe_allow_html=True,
            )
            st.caption(
                f"Verbos conjugados (forma finita) detectados tras `:` ({len(match.verbos_detectados_der)}): {verbos_badge}"
            )


def render_variedades_comparison(res_a: ModelAuditResult, res_b: ModelAuditResult) -> None:
    """Muestra la tabla comparativa pregunta por pregunta de la Prueba 4 (variedades del español)."""
    st.subheader("🧉 Desglose de la prueba 4: reconocimiento de variedades del español (rioplatense y cordobesa)")

    rows = []
    for det_a, det_b in zip(res_a.variedades_result.detalles, res_b.variedades_result.detalles):
        q_key = det_a.question_id.lower()
        enunciado = GROUND_TRUTH_VARIEDADES.get(q_key, {}).get("enunciado", "")
        icon_a = "✅" if det_a.es_correcta else "❌"
        icon_b = "✅" if det_b.es_correcta else "❌"
        rows.append(
            {
                "Ítem": det_a.question_id,
                "Fenómeno evaluado": det_a.pregunta,
                "Enunciado": enunciado,
                "Clave oficial": det_a.respuesta_esperada,
                f"{res_a.modelo}": f"{icon_a} {det_a.respuesta_modelo}",
                f"{res_b.modelo}": f"{icon_b} {det_b.respuesta_modelo}",
            }
        )

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_custom_test_builder() -> tuple[CustomQuestionSpec, ...]:
    """
    Renderiza el constructor interactivo de la Prueba 5 para diseñar en clase
    un test sencillo (de opción múltiple A/B/C/D o de respuesta breve / reescritura)
    y genera automáticamente el prompt listo para copiar.
    """
    st.markdown("#### 🛠️ Configuración de la prueba 5: constructor libre de test lingüístico en clase")
    st.caption(
        "Diseña aquí mismo junto con tus estudiantes entre 1 y 5 consignas lingüísticas, define la respuesta correcta "
        "y copia el prompt autogenerado sin necesidad de editar código ni usar un entorno de programación."
    )

    cfg_col1, cfg_col2 = st.columns([2, 1])
    with cfg_col1:
        modalidad_label = st.radio(
            "Formato de evaluación para esta prueba:",
            options=[
                "Opción múltiple (A / B / C / D)",
                "Respuesta breve o reescritura (busca expresión clave exacta)",
            ],
            horizontal=True,
            key="p4_modalidad",
        )
    with cfg_col2:
        num_q = int(
            st.number_input(
                "Cantidad de preguntas (1 a 5):",
                min_value=1,
                max_value=5,
                step=1,
                key="p4_num_q",
            )
        )

    modalidad_code = "multiple_choice" if "Opción múltiple" in modalidad_label else "frase_clave"
    specs: list[CustomQuestionSpec] = []

    for i in range(1, num_q + 1):
        with st.expander(f"📝 Configurar pregunta Q{i}", expanded=(i == 1)):
            c_fen, c_corr = st.columns([3, 1])
            with c_fen:
                fenomeno = st.text_input(
                    f"Fenómeno lingüístico evaluado (Q{i}):",
                    key=f"p4_q{i}_fen",
                    placeholder="Ej.: Concordancia, dequeísmo, uso del gerundio, condicional de rumor...",
                )
            enunciado = st.text_area(
                f"Enunciado o consigna de la pregunta (Q{i}):",
                key=f"p4_q{i}_enu",
                height=75,
                placeholder="Escribe aquí la pregunta o la oración que los modelos deben analizar o corregir...",
            )

            if modalidad_code == "multiple_choice":
                with c_corr:
                    corr_key = f"p4_q{i}_corr"
                    if corr_key not in st.session_state or isinstance(st.session_state.get(corr_key), str):
                        prev = st.session_state.get(corr_key, "A")
                        st.session_state[corr_key] = [prev] if prev in ("A", "B", "C", "D") else ["A"]
                    letras_correctas = st.multiselect(
                        f"Opción(es) correcta(s) (Q{i}):",
                        options=["A", "B", "C", "D"],
                        key=corr_key,
                        help="Puedes elegir una sola letra (ej. A) o más de una (ej. A y C) si la consigna pide marcar cuáles son válidas.",
                    )
                op_col1, op_col2 = st.columns(2)
                with op_col1:
                    op_a = st.text_input(f"Opción A (Q{i}):", key=f"p4_q{i}_a", placeholder="Texto de la opción A...")
                    op_c = st.text_input(f"Opción C (Q{i}):", key=f"p4_q{i}_c", placeholder="Texto de la opción C...")
                with op_col2:
                    op_b = st.text_input(f"Opción B (Q{i}):", key=f"p4_q{i}_b", placeholder="Texto de la opción B...")
                    op_d = st.text_input(f"Opción D (Q{i}):", key=f"p4_q{i}_d", placeholder="Texto de la opción D...")

                specs.append(
                    CustomQuestionSpec(
                        id=f"q{i}",
                        fenomeno=fenomeno.strip() or f"Pregunta {i}",
                        enunciado=enunciado.strip() or f"Seleccione la opción o las opciones válidas para la pregunta {i}:",
                        modo="multiple_choice",
                        opciones={
                            "A": op_a.strip() or "Opción A",
                            "B": op_b.strip() or "Opción B",
                            "C": op_c.strip() or "Opción C",
                            "D": op_d.strip() or "Opción D",
                        },
                        respuesta_correcta=", ".join(sorted(letras_correctas)),
                    )
                )
            else:
                frase_esperada = st.text_input(
                    f"Expresión correcta esperada (Q{i}) — si hay varias válidas, sepáralas con `;`:",
                    key=f"p4_q{i}_frase",
                    placeholder="Ej.: había muchos  (o varias alternativas separadas por punto y coma: había muchos ; existían muchos)",
                    help="Python verificará de forma determinista si la respuesta del modelo para este ítem contiene esta expresión normalizada.",
                )
                specs.append(
                    CustomQuestionSpec(
                        id=f"q{i}",
                        fenomeno=fenomeno.strip() or f"Pregunta {i}",
                        enunciado=enunciado.strip() or f"Responda la consigna {i}:",
                        modo="frase_clave",
                        opciones={},
                        respuesta_correcta=frase_esperada.strip() or "(definir respuesta esperada)",
                    )
                )

    tuple_specs = tuple(specs)
    with st.expander("📋 Prompt 6 (Prueba 5): prueba creada en clase (listo para copiar)", expanded=True):
        st.caption("Haz clic en el icono de copiar (esquina superior derecha del recuadro) y pégalo en ambos modelos:")
        st.code(build_custom_test_prompt(tuple_specs), language=None, wrap_lines=True)

    return tuple_specs


def render_custom_test_comparison(res_a: ModelAuditResult, res_b: ModelAuditResult) -> None:
    """Muestra la tabla comparativa pregunta por pregunta de la Prueba 5 creada en clase."""
    st.subheader("🛠️ Desglose de la prueba 5: test creado en clase")

    cr_a = res_a.custom_result
    cr_b = res_b.custom_result

    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.metric(
            f"Puntaje prueba 5 — {res_a.modelo}",
            f"{cr_a.puntaje_absoluto}/{cr_a.total_preguntas}",
            f"{cr_a.precision_porcentaje:.1f}%",
        )
    with m_col2:
        st.metric(
            f"Puntaje prueba 5 — {res_b.modelo}",
            f"{cr_b.puntaje_absoluto}/{cr_b.total_preguntas}",
            f"{cr_b.precision_porcentaje:.1f}%",
        )

    rows = []
    for det_a, det_b in zip(cr_a.detalles, cr_b.detalles):
        icon_a = "✅" if det_a.es_correcta else "❌"
        icon_b = "✅" if det_b.es_correcta else "❌"
        rows.append(
            {
                "Ítem": det_a.question_id,
                "Fenómeno evaluado": det_a.fenomeno,
                "Enunciado": det_a.enunciado,
                "Clave esperada": det_a.respuesta_esperada,
                f"{res_a.modelo}": f"{icon_a} {det_a.respuesta_modelo}",
                f"{res_b.modelo}": f"{icon_b} {det_b.respuesta_modelo}",
            }
        )

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_evaluator_tab() -> None:
    render_header_and_prompts()

    # ========================================================================
    # SELECTOR DE PRUEBAS ACTIVAS Y BOTONES DE DEMOSTRACIÓN
    # ========================================================================
    st.markdown("#### ⚙️ Selecciona qué pruebas deseas ejecutar:")
    ctrl1, ctrl2, ctrl3, ctrl4, ctrl5, ctrl6, ctrl7 = st.columns(
        [1.9, 1.7, 1.9, 2.0, 1.9, 0.9, 0.9]
    )
    with ctrl1:
        run_meta = st.toggle(
            "🧪 Prueba 1: test metalingüístico",
            key="run_meta",
        )
    with ctrl2:
        run_style = st.toggle(
            "✍️ Prueba 2: estilometría",
            key="run_style",
        )
    with ctrl3:
        run_errors = st.toggle(
            "🔎 Prueba 3: errores (~1000w)",
            key="run_errors",
        )
    with ctrl4:
        run_variedades = st.toggle(
            "🧉 Prueba 4: variedades (AR)",
            key="run_variedades",
        )
    with ctrl5:
        run_custom = st.toggle(
            "🛠️ Prueba 5: crear en clase",
            key="run_custom",
        )
    with ctrl6:
        st.button(
            "✨ Ejemplo",
            on_click=load_demo_texts,
            use_container_width=True,
        )
    with ctrl7:
        st.button(
            "🧹 Limpiar",
            on_click=clear_texts,
            use_container_width=True,
        )

    if not run_meta and not run_style and not run_errors and not run_variedades and not run_custom:
        st.warning("Activa al menos una de las cinco pruebas con los interruptores de arriba para continuar.")
        return

    # ========================================================================
    # PROMPTS DE LAS PRUEBAS ACTIVAS (DEBAJO DE LOS INTERRUPTORES)
    # ========================================================================
    meta_gt = GROUND_TRUTH_META
    if run_meta:
        subbanco_label = st.radio(
            "🧪 Sub-banco de la Prueba 1:",
            options=[
                "📘 1A: Gramática normativa (RAE / ASALE)",
                "🧬 1B: Gramática formal y agramaticalidades (*) — no normativa",
            ],
            horizontal=True,
            key="p1_subbanco",
            help=(
                "1A evalúa reglas normativas académicas (concordancia, 'se' impersonal, queísmo/dequeísmo, gerundio). "
                "1B evalúa sintaxis descriptiva y formal no prescriptiva (islas sintácticas, inacusatividad, ligamiento y polaridad negativa)."
            ),
        )
        if "1B" in subbanco_label:
            meta_gt = GROUND_TRUTH_META_FORMAL
            prompt_title = "📋 Prompt 1B (Prueba 1 — Gramática formal y agramaticalidades *): copiar prompt (Q1–Q4)"
        else:
            meta_gt = GROUND_TRUTH_META
            prompt_title = "📋 Prompt 1A (Prueba 1 — Gramática normativa RAE): copiar prompt (Q1–Q4)"

        with st.expander(prompt_title, expanded=False):
            st.caption("Haz clic en el icono de copiar (esquina superior derecha del recuadro) y pégalo en ambos modelos:")
            st.code(get_metalinguistic_prompt_text(meta_gt), language=None, wrap_lines=True)

    if run_style:
        col_p2, col_p3 = st.columns(2)
        with col_p2:
            with st.expander("📋 Prompt 2 (Prueba 2): ensayo sobre filología (opción 1)", expanded=False):
                st.caption("Haz clic en el icono de copiar (esquina superior derecha del recuadro):")
                st.code(PROMPTS_SUGERIDOS[0]["prompt"], language=None, wrap_lines=True)
        with col_p3:
            with st.expander("📋 Prompt 3 (Prueba 2): ensayo sobre IA y estilo (opción 2)", expanded=False):
                st.caption("Haz clic en el icono de copiar (esquina superior derecha del recuadro):")
                st.code(PROMPTS_SUGERIDOS[1]["prompt"], language=None, wrap_lines=True)

    planted_errors: tuple[PlantedError, ...] | None = None
    if run_errors:
        st.divider()
        _, planted_errors = render_error_benchmark_selector()
        st.divider()

    if run_variedades:
        with st.expander(
            "📋 Prompt 5 (Prueba 4 — Variedades del español: rioplatense y cordobesa): copiar prompt (Q1–Q4)",
            expanded=True,
        ):
            st.caption(
                "Prueba diseñada por los estudiantes. Haz clic en el icono de copiar (esquina superior derecha del recuadro) y pégalo en ambos modelos:"
            )
            st.code(get_variedades_prompt_text(), language=None, wrap_lines=True)

    custom_specs: tuple[CustomQuestionSpec, ...] | None = None
    if run_custom:
        if not run_errors:
            st.divider()
        custom_specs = render_custom_test_builder()
        st.divider()

    # ========================================================================
    # DOS PANELES PARALELOS (MODELO 1 vs MODELO 2) SEGÚN PRUEBAS ACTIVAS
    # ========================================================================
    col_a, col_b = st.columns(2)

    with col_a:
        nombre_a = st.text_input(
            "Nombre del modelo 1",
            key="nombre_a",
            placeholder="Ej.: GPT-4o, Gemini, Llama 3...",
        )
        if run_meta:
            meta_input_a = st.text_area(
                "🧪 Respuestas del test metalingüístico (Q1–Q4) — Modelo 1:",
                key="meta_input_a",
                height=90,
                help=(
                    'Pega el JSON del modelo (ej: {"q1": "B", "q2": "C", "q3": "C", "q4": "B"}), '
                    'o texto como "1: B, 2: C, 3: C, 4: B", o las 4 letras "B, C, C, B".'
                ),
                placeholder='Ej: {"q1": "B", "q2": "C", "q3": "C", "q4": "B"} o B, C, C, B',
            )
        else:
            meta_input_a = ""

        if run_style:
            texto_a = st.text_area(
                "✍️ Texto ensayístico para estilometría — Modelo 1:",
                key="texto_a",
                height=220,
                placeholder="Copia y pega aquí el ensayo generado por el modelo 1...",
            )
        else:
            texto_a = ""

        if run_errors:
            error_input_a = st.text_area(
                "🔎 Lista ordenada de errores y correcciones — Modelo 1:",
                key="error_input_a",
                height=220,
                placeholder='Pega aquí la lista numerada de errores que encontró el modelo 1 (ej: 1. "error" -> "corrección" | explicación)...',
            )
        else:
            error_input_a = ""

        if run_variedades:
            variedades_input_a = st.text_area(
                "🧉 Respuestas de variedades del español (Prueba 4, Q1–Q4) — Modelo 1:",
                key="variedades_input_a",
                height=90,
                help=(
                    'Pega el JSON del modelo (ej: {"q1": "C", "q2": "B", "q3": "D", "q4": "B"}), '
                    'o texto como "1: C, 2: B, 3: D, 4: B", o las 4 letras "C, B, D, B".'
                ),
                placeholder='Ej: {"q1": "C", "q2": "B", "q3": "D", "q4": "B"} o C, B, D, B',
            )
        else:
            variedades_input_a = ""

        if run_custom:
            custom_input_a = st.text_area(
                "🛠️ Respuestas de la prueba 5 (creada en clase) — Modelo 1:",
                key="custom_input_a",
                height=110,
                placeholder='Pega aquí la respuesta del modelo 1 al Prompt 6 (ej: {"q1": "A"} o las respuestas numeradas)...',
            )
        else:
            custom_input_a = ""

    with col_b:
        nombre_b = st.text_input(
            "Nombre del modelo 2",
            key="nombre_b",
            placeholder="Ej.: Claude, Mistral, Qwen, OLMo...",
        )
        if run_meta:
            meta_input_b = st.text_area(
                "🧪 Respuestas del test metalingüístico (Q1–Q4) — Modelo 2:",
                key="meta_input_b",
                height=90,
                help=(
                    'Pega el JSON del modelo (ej: {"q1": "A", "q2": "C", "q3": "B", "q4": "B"}), '
                    'o texto como "1: A, 2: C, 3: B, 4: B", o las 4 letras "A, C, B, B".'
                ),
                placeholder='Ej: {"q1": "A", "q2": "C", "q3": "B", "q4": "B"} o A, C, B, B',
            )
        else:
            meta_input_b = ""

        if run_style:
            texto_b = st.text_area(
                "✍️ Texto ensayístico para estilometría — Modelo 2:",
                key="texto_b",
                height=220,
                placeholder="Copia y pega aquí el ensayo generado por el modelo 2...",
            )
        else:
            texto_b = ""

        if run_errors:
            error_input_b = st.text_area(
                "🔎 Lista ordenada de errores y correcciones — Modelo 2:",
                key="error_input_b",
                height=220,
                placeholder='Pega aquí la lista numerada de errores que encontró el modelo 2 (ej: 1. "error" -> "corrección" | explicación)...',
            )
        else:
            error_input_b = ""

        if run_variedades:
            variedades_input_b = st.text_area(
                "🧉 Respuestas de variedades del español (Prueba 4, Q1–Q4) — Modelo 2:",
                key="variedades_input_b",
                height=90,
                help=(
                    'Pega el JSON del modelo (ej: {"q1": "A", "q2": "B", "q3": "D", "q4": "A"}), '
                    'o texto como "1: A, 2: B, 3: D, 4: A", o las 4 letras "A, B, D, A".'
                ),
                placeholder='Ej: {"q1": "A", "q2": "B", "q3": "D", "q4": "A"} o A, B, D, A',
            )
        else:
            variedades_input_b = ""

        if run_custom:
            custom_input_b = st.text_area(
                "🛠️ Respuestas de la prueba 5 (creada en clase) — Modelo 2:",
                key="custom_input_b",
                height=110,
                placeholder='Pega aquí la respuesta del modelo 2 al Prompt 6 (ej: {"q1": "D"} o las respuestas numeradas)...',
            )
        else:
            custom_input_b = ""

    # ========================================================================
    # BOTÓN DE EJECUCIÓN Y PANEL DE RESULTADOS
    # ========================================================================
    st.button(
        "▶️ Ejecutar evaluación y comparar modelos",
        type="primary",
        use_container_width=True,
    )

    has_any_input = any(
        val.strip()
        for val in (
            meta_input_a,
            meta_input_b,
            texto_a,
            texto_b,
            error_input_a,
            error_input_b,
            variedades_input_a,
            variedades_input_b,
            custom_input_a,
            custom_input_b,
        )
    )
    if not has_any_input:
        st.info(
            "💡 Pega las respuestas de los modelos en las cajas de arriba y presiona "
            "**«▶️ Ejecutar evaluación y comparar modelos»**, o haz clic en **«✨ Ejemplo»** para cargar una demostración."
        )
        return

    nlp: Language | None = None
    model_loaded = SPACY_MODEL_NAME
    if run_style:
        try:
            nlp, model_loaded = get_cached_nlp()
        except RuntimeError as exc:
            st.error(str(exc))
            st.stop()
        if model_loaded == nlp_ligero.NOMBRE_MODELO_LIGERO:
            st.info(
                "ℹ️ **Versión web:** spaCy no puede ejecutarse en el navegador, así que la detección de "
                "dos puntos de la Prueba 2 usa un etiquetador heurístico (léxico de formas verbales + reglas "
                "de contexto). Puede diferir levemente de la versión con spaCy (`es_core_news_lg`)."
            )

    df_summary, results = compare_models_in_memory(
        [
            (
                nombre_a or "Modelo A",
                texto_a,
                meta_input_a,
                error_input_a,
                custom_input_a,
                variedades_input_a,
            ),
            (
                nombre_b or "Modelo B",
                texto_b,
                meta_input_b,
                error_input_b,
                custom_input_b,
                variedades_input_b,
            ),
        ],
        nlp=nlp,
        planted_errors=planted_errors,
        custom_specs=custom_specs,
        run_meta=run_meta,
        run_style=run_style,
        run_errors=run_errors,
        run_variedades=run_variedades,
        run_custom=run_custom,
        meta_ground_truth=meta_gt,
    )

    # 1. Tabla resumen general adaptada a las pruebas activas
    st.divider()
    st.subheader("📊 Tabla comparativa de resultados")
    st.dataframe(
        df_summary,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Puntaje meta (0-4)": st.column_config.NumberColumn("Puntaje meta (0-4)", format="%d / 4"),
            "Precisión meta (%)": st.column_config.ProgressColumn(
                "Precisión meta (%)",
                format="%.1f%%",
                min_value=0,
                max_value=100,
            ),
            "Cobertura (%)": st.column_config.ProgressColumn(
                "Cobertura (%)",
                format="%.1f%%",
                min_value=0,
                max_value=100,
            ),
            "Precisión aprox. (%)": st.column_config.ProgressColumn(
                "Precisión aprox. (%)",
                format="%.1f%%",
                min_value=0,
                max_value=100,
            ),
            "Puntaje variedades (0-4)": st.column_config.NumberColumn(
                "Puntaje variedades (0-4)",
                format="%d / 4",
            ),
            "Precisión variedades (%)": st.column_config.ProgressColumn(
                "Precisión variedades (%)",
                format="%.1f%%",
                min_value=0,
                max_value=100,
            ),
            "Precisión prueba 5 (%)": st.column_config.ProgressColumn(
                "Precisión prueba 5 (%)",
                format="%.1f%%",
                min_value=0,
                max_value=100,
            ),
            "Densidad reframing / 500w": st.column_config.NumberColumn(format="%.2f"),
            "Densidad dos puntos / 500w": st.column_config.NumberColumn(format="%.2f"),
        },
    )

    # 2. Si la Prueba 1 (opción múltiple) está activa, mostrar desglose pregunta por pregunta
    if run_meta:
        st.divider()
        render_metalinguistic_comparison(results[0], results[1], meta_gt=meta_gt)

    # 3. Si la Prueba 3 (detección de errores) está activa, mostrar desglose claros vs sutiles
    if run_errors:
        st.divider()
        render_error_detection_comparison(results[0], results[1])

    # 4. Si la Prueba 4 (variedades del español) está activa, mostrar desglose pregunta por pregunta
    if run_variedades:
        st.divider()
        render_variedades_comparison(results[0], results[1])

    # 5. Si la Prueba 5 (creada en clase) está activa, mostrar desglose pregunta por pregunta
    if run_custom:
        st.divider()
        render_custom_test_comparison(results[0], results[1])

    # 6. Si la Prueba 2 (estilometría) está activa, mostrar gráfico de barras y visor cualitativo
    if run_style:
        st.divider()
        st.subheader("📈 Comparativa de densidad de tics sintácticos (cada 500 palabras)")
        chart_df = df_summary.set_index("Modelo")[
            ["Densidad reframing / 500w", "Densidad dos puntos / 500w"]
        ]
        st.bar_chart(chart_df, height=320)

        st.divider()
        st.subheader("🔍 Evidencia cualitativa lado a lado (oraciones detectadas)")
        st.caption(
            "Nota: En amarillo se resalta el patrón de *reframing*; en rojo los dos puntos oracionales "
            "(`[Oración con verbo] : [Oración con verbo conjugado (forma finita)]`) "
            "y en verde los verbos conjugados (forma finita) que validan la segunda proposición."
        )

        ev_col_a, ev_col_b = st.columns(2)
        with ev_col_a:
            render_model_evidence_card(results[0], show_meta_metric=run_meta)
        with ev_col_b:
            render_model_evidence_card(results[1], show_meta_metric=run_meta)

    # 7. Botones de descarga en memoria
    st.divider()
    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            "⬇️ Descargar tabla comparativa (CSV)",
            data=df_summary.to_csv(index=False).encode("utf-8-sig"),
            file_name="metricas_resumen.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with dl2:
        log_str = format_detailed_audit_log(
            results,
            spacy_model_used=model_loaded,
            run_meta=run_meta,
            run_style=run_style,
            run_errors=run_errors,
            run_variedades=run_variedades,
            run_custom=run_custom,
        )
        st.download_button(
            "📄 Descargar informe detallado (TXT)",
            data=log_str.encode("utf-8"),
            file_name="auditoria_detallada.txt",
            mime="text/plain",
            use_container_width=True,
        )


def main() -> None:
    init_session_state()
    st.title("📐 Benchmark en vivo: evaluación metalingüística y estilometría de LLMs")

    tab_evaluador, tab_como_funciona = st.tabs(
        ["🚀 Evaluador interactivo", "ℹ️ Cómo funciona este sitio"]
    )
    with tab_evaluador:
        render_evaluator_tab()
    with tab_como_funciona:
        render_about_tab()


if __name__ == "__main__":
    main()
