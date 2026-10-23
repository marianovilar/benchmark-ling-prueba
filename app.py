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

import pandas as pd
import streamlit as st
from spacy.language import Language

import textos_errores
import auditor

# Recargar módulos para evitar que Streamlit Cloud conserve versiones en caché tras un git push
importlib.reload(textos_errores)
importlib.reload(auditor)

from auditor import (
    GROUND_TRUTH_META,
    META_DEMO_MODELO_A,
    META_DEMO_MODELO_B,
    PROMPTS_SUGERIDOS,
    SPACY_MODEL_NAME,
    TEXTO_DEMO_MODELO_A,
    TEXTO_DEMO_MODELO_B,
    ModelAuditResult,
    compare_models_in_memory,
    format_detailed_audit_log,
    get_metalinguistic_prompt_text,
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
    """Resalta los dos puntos y los verbos conjugados detectados en la segunda proposición."""
    izq_html = html.escape(seg_izq)
    der_html = html.escape(seg_der)

    for verbo in sorted(set(verbos_der), key=len, reverse=True):
        verbo_esc = html.escape(verbo)
        pattern = re.compile(rf"\b{re.escape(verbo_esc)}\b", re.IGNORECASE)
        der_html = pattern.sub(
            lambda m: (
                '<span style="background-color: #b9f6ca; color: #004d40; '
                'padding: 1px 4px; border-radius: 3px; font-weight: 600;" '
                f'title="Verbo detectado por spaCy (VERB/AUX)">{m.group(0)}</span>'
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
    en objetos `PlantedError` para poder evaluar cualquier texto propio contra una clave fija.
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
            keys = tuple(k for k in (err_frag.lower(), corr_frag.lower()) if k and k != "(ver corrección)")
            errors.append(
                PlantedError(
                    orden=idx,
                    categoria=categoria,
                    fenomeno="Error definido por el usuario",
                    fragmento_erroneo=err_frag,
                    correccion_esperada=corr_frag,
                    claves_busqueda=keys,
                )
            )
    return tuple(errors)


# ============================================================================
# ESTADO INICIAL DE LA SESIÓN (DATOS DE DEMOSTRACIÓN)
# ============================================================================


def init_session_state() -> None:
    if "nombre_a" not in st.session_state:
        st.session_state["nombre_a"] = "Modelo A (ej. GPT-4o)"
    if "nombre_b" not in st.session_state:
        st.session_state["nombre_b"] = "Modelo B (ej. modelo pequeño / gratis)"
    if "meta_input_a" not in st.session_state:
        st.session_state["meta_input_a"] = META_DEMO_MODELO_A
    if "meta_input_b" not in st.session_state:
        st.session_state["meta_input_b"] = META_DEMO_MODELO_B
    if "texto_a" not in st.session_state:
        st.session_state["texto_a"] = TEXTO_DEMO_MODELO_A
    if "texto_b" not in st.session_state:
        st.session_state["texto_b"] = TEXTO_DEMO_MODELO_B
    if "error_input_a" not in st.session_state:
        st.session_state["error_input_a"] = DEMO_RESPUESTA_ERRORES_MODELO_A
    if "error_input_b" not in st.session_state:
        st.session_state["error_input_b"] = DEMO_RESPUESTA_ERRORES_MODELO_B
    if "run_meta" not in st.session_state:
        st.session_state["run_meta"] = True
    if "run_style" not in st.session_state:
        st.session_state["run_style"] = True
    if "run_errors" not in st.session_state:
        st.session_state["run_errors"] = True


def load_demo_texts() -> None:
    st.session_state["meta_input_a"] = META_DEMO_MODELO_A
    st.session_state["meta_input_b"] = META_DEMO_MODELO_B
    st.session_state["texto_a"] = TEXTO_DEMO_MODELO_A
    st.session_state["texto_b"] = TEXTO_DEMO_MODELO_B
    st.session_state["error_input_a"] = DEMO_RESPUESTA_ERRORES_MODELO_A
    st.session_state["error_input_b"] = DEMO_RESPUESTA_ERRORES_MODELO_B


def clear_texts() -> None:
    st.session_state["meta_input_a"] = ""
    st.session_state["meta_input_b"] = ""
    st.session_state["texto_a"] = ""
    st.session_state["texto_b"] = ""
    st.session_state["error_input_a"] = ""
    st.session_state["error_input_b"] = ""


# ============================================================================
# COMPONENTES DE LA INTERFAZ
# ============================================================================


def render_header_and_prompts() -> None:
    st.title("📐 Benchmark en vivo: evaluación metalingüística y estilometría de LLMs")
    st.markdown(
        "Compara dos modelos de lenguaje en español pegando sus respuestas lado a lado. "
        "Puedes activar o desactivar libremente cualquiera de las **tres pruebas** según el objetivo de la clase."
    )

    col_guide, col_links = st.columns(2)

    with col_guide:
        with st.expander("📖 ¿Qué evalúa cada prueba?", expanded=False):
            st.markdown(
                """
                1. **🧪 Prueba 1 — Test metalingüístico (opción múltiple Q1–Q4):**
                   Evalúa precisión gramatical normativa (0 a 4 puntos) contra una clave fija (*ground truth*).
                2. **✍️ Prueba 2 — Auditoría estilística en prosa (reframing + dos puntos):**
                   - **Tic 1 — Reframing asindético y adversativo (*«No es X, es Y»* / *«No era solo X, sino Y»*):** Fórmulas binarias donde se niega un término para reformularlo tras coma o punto y coma.
                   - **Tic 2 — Dos puntos oracionales (`[Oración] : [Oración con verbo]`):** Detecta con **spaCy** oraciones donde tras `:` hay más de 3 palabras y al menos un verbo conjugado (`VERB`/`AUX`).
                3. **🔎 Prueba 3 — Detección y corrección de errores en textos largos (~1000 palabras):**
                   Compara la lista de errores que encuentra el modelo contra una **clave oficial de errores plantados** (divididos en **claros** y **sutiles**).
                """
            )

    with col_links:
        with st.expander("🌐 Modelos gratuitos y alternativos para probar en clase", expanded=False):
            st.markdown(
                """
                Además de los modelos comerciales conocidos (ChatGPT, Gemini, Claude), puedes usar estos sitios **gratuitos** para probar modelos abiertos, pequeños (7B–9B) o menos conocidos:

                * 🏟️ **[LMArena (Chatbot Arena)](https://lmarena.ai/)** *(Recomendado)* — Usa la pestaña *Side-by-Side* o *Direct Chat* para probar gratis y sin cuenta docenas de modelos grandes y pequeños (`Llama`, `Gemma`, `Qwen`, `Mistral`, `Command-R`, `Phi`, `Yi`, etc.).
                * 🦆 **[Duck.ai (DuckDuckGo AI Chat)](https://duck.ai/)** — Gratis y sin registro. Permite alternar entre modelos ligeros (`Mistral Small`, `Llama 3`, `Claude Haiku`, `GPT-4o mini`).
                * 🧪 **[AllenAI Playground (OLMo)](https://playground.allenai.org/)** — Modelos académicos *open-source* pequeños (`OLMo 7B / 13B / 32B`), ideales para contrastar fallos gramaticales frente a modelos gigantes.
                * 🤗 **[HuggingChat](https://huggingface.co/chat/)** — Acceso gratuito a modelos abiertos variados (`Qwen`, `Cohere Command R`, `Llama`, `SmolLM`).
                * 🇫🇷 **[Le Chat — Mistral AI](https://chat.mistral.ai/)** — Modelo europeo gratuito en el navegador.
                * 🐉 **[Qwen Chat (Alibaba)](https://chat.qwen.ai/)** — Permite elegir distintas variantes de tamaño de la familia `Qwen` gratis.
                * 🐋 **[DeepSeek Chat](https://chat.deepseek.com/)** — Modelo gratuito alternativo útil para comparar tics retóricos en español.
                """
            )

    # Ventanas separadas para los prompts de las pruebas 1 y 2
    with st.expander("📋 Prompt 1: test metalingüístico (Q1–Q4)", expanded=False):
        st.caption("Haz clic en el icono de copiar (esquina superior derecha del recuadro):")
        st.code(get_metalinguistic_prompt_text(), language=None, wrap_lines=True)

    col_p2, col_p3 = st.columns(2)
    with col_p2:
        with st.expander("📋 Prompt 2: ensayo sobre filología (opción 1)", expanded=False):
            st.caption("Haz clic en el icono de copiar (esquina superior derecha del recuadro):")
            st.code(PROMPTS_SUGERIDOS[0]["prompt"], language=None, wrap_lines=True)
    with col_p3:
        with st.expander("📋 Prompt 3: ensayo sobre IA y estilo (opción 2)", expanded=False):
            st.caption("Haz clic en el icono de copiar (esquina superior derecha del recuadro):")
            st.code(PROMPTS_SUGERIDOS[1]["prompt"], language=None, wrap_lines=True)


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


def render_metalinguistic_comparison(res_a: ModelAuditResult, res_b: ModelAuditResult) -> None:
    """Muestra la tabla comparativa pregunta por pregunta del test metalingüístico (Q1-Q4)."""
    st.subheader("🧪 Desglose de la prueba 1: test metalingüístico (opción múltiple Q1–Q4)")

    rows = []
    for det_a, det_b in zip(res_a.meta.detalles, res_b.meta.detalles):
        q_key = det_a.question_id.lower()
        enunciado = GROUND_TRUTH_META.get(q_key, {}).get("enunciado", "")
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

    if er_a.tiene_ground_truth and er_b.tiene_ground_truth:
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.markdown(f"**{res_a.modelo}**")
            k1, k2, k3 = st.columns(3)
            k1.metric("Total detectados", f"{er_a.detectados_total}/{er_a.total_plantados}", f"{er_a.precision_porcentaje:.1f}%")
            k2.metric("Errores claros", f"{er_a.claros_detectados}/{er_a.claros_total}")
            k3.metric("Errores sutiles", f"{er_a.sutiles_detectados}/{er_a.sutiles_total}")
        with m_col2:
            st.markdown(f"**{res_b.modelo}**")
            k1, k2, k3 = st.columns(3)
            k1.metric("Total detectados", f"{er_b.detectados_total}/{er_b.total_plantados}", f"{er_b.precision_porcentaje:.1f}%")
            k2.metric("Errores claros", f"{er_b.claros_detectados}/{er_b.claros_total}")
            k3.metric("Errores sutiles", f"{er_b.sutiles_detectados}/{er_b.sutiles_total}")

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
                    f"{res_b.modelo}": "✅ Detectado" if det_b.detectado else "❌ Omitido",
                }
            )
        st.dataframe(pd.DataFrame(rows_err), use_container_width=True, hide_index=True)

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

    st.markdown(f"**🟥 Tic 2 — Dos puntos oracionales detectados ({res.total_dos_puntos}):**")
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
            st.caption(f"Verbos detectados tras `:` ({len(match.verbos_detectados_der)}): {verbos_badge}")


def main() -> None:
    init_session_state()
    render_header_and_prompts()

    # ========================================================================
    # SELECTOR DE PRUEBAS ACTIVAS Y BOTONES DE DEMOSTRACIÓN
    # ========================================================================
    st.markdown("#### ⚙️ Selecciona qué pruebas deseas ejecutar:")
    ctrl1, ctrl2, ctrl3, ctrl4, ctrl5 = st.columns([2.5, 2.5, 2.5, 1.25, 1.25])
    with ctrl1:
        run_meta = st.toggle(
            "🧪 Prueba 1: test metalingüístico (Q1–Q4)",
            key="run_meta",
        )
    with ctrl2:
        run_style = st.toggle(
            "✍️ Prueba 2: auditoría estilística",
            key="run_style",
        )
    with ctrl3:
        run_errors = st.toggle(
            "🔎 Prueba 3: detección de errores (~1000w)",
            key="run_errors",
        )
    with ctrl4:
        st.button(
            "✨ Ejemplo",
            on_click=load_demo_texts,
            use_container_width=True,
        )
    with ctrl5:
        st.button(
            "🧹 Limpiar",
            on_click=clear_texts,
            use_container_width=True,
        )

    if not run_meta and not run_style and not run_errors:
        st.warning("Activa al menos una de las tres pruebas con los interruptores de arriba para continuar.")
        return

    planted_errors: tuple[PlantedError, ...] | None = None
    if run_errors:
        st.divider()
        _, planted_errors = render_error_benchmark_selector()
        st.divider()

    # ========================================================================
    # DOS PANELES PARALELOS (MODELO 1 vs MODELO 2) SEGÚN PRUEBAS ACTIVAS
    # ========================================================================
    col_a, col_b = st.columns(2)

    with col_a:
        nombre_a = st.text_input("Nombre del modelo 1", key="nombre_a")
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

    with col_b:
        nombre_b = st.text_input("Nombre del modelo 2", key="nombre_b")
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

    # ========================================================================
    # BOTÓN DE EJECUCIÓN Y PANEL DE RESULTADOS
    # ========================================================================
    st.button(
        "▶️ Ejecutar evaluación y comparar modelos",
        type="primary",
        use_container_width=True,
    )

    nlp: Language | None = None
    model_loaded = SPACY_MODEL_NAME
    if run_style:
        try:
            nlp, model_loaded = get_cached_nlp()
        except RuntimeError as exc:
            st.error(str(exc))
            st.stop()

    df_summary, results = compare_models_in_memory(
        [
            (nombre_a or "Modelo A", texto_a, meta_input_a, error_input_a),
            (nombre_b or "Modelo B", texto_b, meta_input_b, error_input_b),
        ],
        nlp=nlp,
        planted_errors=planted_errors,
        run_meta=run_meta,
        run_style=run_style,
        run_errors=run_errors,
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
            "Detección errores (%)": st.column_config.ProgressColumn(
                "Detección errores (%)",
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
        render_metalinguistic_comparison(results[0], results[1])

    # 3. Si la Prueba 3 (detección de errores) está activa, mostrar desglose claros vs sutiles
    if run_errors:
        st.divider()
        render_error_detection_comparison(results[0], results[1])

    # 4. Si la Prueba 2 (estilometría) está activa, mostrar gráfico de barras y visor cualitativo
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
            "y en verde los verbos conjugados (`VERB`/`AUX`) que validan la segunda proposición."
        )

        ev_col_a, ev_col_b = st.columns(2)
        with ev_col_a:
            render_model_evidence_card(results[0], show_meta_metric=run_meta)
        with ev_col_b:
            render_model_evidence_card(results[1], show_meta_metric=run_meta)

    # 5. Botones de descarga en memoria
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
        )
        st.download_button(
            "📄 Descargar informe detallado (TXT)",
            data=log_str.encode("utf-8"),
            file_name="auditoria_detallada.txt",
            mime="text/plain",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
