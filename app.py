"""
app.py - Interfaz interactiva en Streamlit para comparar 2 modelos en vivo (ideal para clases y GitHub).

Permite activar o desactivar libremente cualquiera de las dos pruebas:
- Prueba 1: Evaluación Metalingüística Objetiva (Multiple Choice Q1-Q4 contra Ground Truth).
- Prueba 2: Auditoría Estilística en Prosa (Reframing Asindético + Dos Puntos Oracionales con spaCy).

Ejecución:
    streamlit run app.py
"""

from __future__ import annotations

import html
import re

import pandas as pd
import streamlit as st
from spacy.language import Language

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

# ============================================================================
# CONFIGURACIÓN DE PÁGINA Y CACHÉ DE SPACY
# ============================================================================

st.set_page_config(
    page_title="Benchmark Estilometría y Gramática LLM",
    page_icon="📐",
    layout="wide",
)


@st.cache_resource(show_spinner="Cargando modelo lingüístico de spaCy...")
def get_cached_nlp() -> tuple[Language, str]:
    """Carga y guarda en caché el modelo de spaCy solo cuando la Prueba 2 (Estilometría) está activa."""
    return load_spacy_model(SPACY_MODEL_NAME)


# ============================================================================
# FUNCIONES DE RESALTADO HTML PARA EVIDENCIA CUALITATIVA
# ============================================================================


def highlight_reframing_html(sentence: str, fragment: str) -> str:
    """Resalta en amarillo el patrón de reframing asindético dentro de la oración."""
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


# ============================================================================
# ESTADO INICIAL DE LA SESIÓN (DATOS DE DEMOSTRACIÓN)
# ============================================================================


def init_session_state() -> None:
    if "nombre_a" not in st.session_state:
        st.session_state["nombre_a"] = "Modelo A (ej. GPT-4o)"
    if "nombre_b" not in st.session_state:
        st.session_state["nombre_b"] = "Modelo B (ej. Claude 3.5)"
    if "meta_input_a" not in st.session_state:
        st.session_state["meta_input_a"] = META_DEMO_MODELO_A
    if "meta_input_b" not in st.session_state:
        st.session_state["meta_input_b"] = META_DEMO_MODELO_B
    if "texto_a" not in st.session_state:
        st.session_state["texto_a"] = TEXTO_DEMO_MODELO_A
    if "texto_b" not in st.session_state:
        st.session_state["texto_b"] = TEXTO_DEMO_MODELO_B
    if "run_meta" not in st.session_state:
        st.session_state["run_meta"] = True
    if "run_style" not in st.session_state:
        st.session_state["run_style"] = True


def load_demo_texts() -> None:
    st.session_state["meta_input_a"] = META_DEMO_MODELO_A
    st.session_state["meta_input_b"] = META_DEMO_MODELO_B
    st.session_state["texto_a"] = TEXTO_DEMO_MODELO_A
    st.session_state["texto_b"] = TEXTO_DEMO_MODELO_B


def clear_texts() -> None:
    st.session_state["meta_input_a"] = ""
    st.session_state["meta_input_b"] = ""
    st.session_state["texto_a"] = ""
    st.session_state["texto_b"] = ""


# ============================================================================
# COMPONENTES DE LA INTERFAZ
# ============================================================================


def render_header_and_prompts() -> None:
    st.title("📐 Benchmark en Vivo: Evaluación Metalingüística y Estilometría de LLMs")
    st.markdown(
        "Compara dos modelos de lenguaje en español pegando sus respuestas lado a lado. "
        "Puedes ejecutar **ambas pruebas al mismo tiempo o activar solo una** según lo que quieras mostrar en clase."
    )

    col_guide, col_prompts = st.columns(2)

    with col_guide:
        with st.expander("📖 ¿Qué evalúa cada prueba?", expanded=False):
            st.markdown(
                """
                1. **🧪 Prueba 1 — Test Metalingüístico (Multiple Choice Q1–Q4):**
                   Evalúa precisión gramatical normativa (0 a 4 puntos) en:
                   - `Q1`: Concordancia con colectivos partitivos (*clave: B*)
                   - `Q2`: Impersonal con *se* y acusativo animado (*clave: C*)
                   - `Q3`: Régimen preposicional y queísmo/dequeísmo (*clave: C*)
                   - `Q4`: Gerundio de posterioridad temporal (*clave: B*)
                2. **✍️ Prueba 2 — Auditoría Estilística en Prosa (Reframing + Dos Puntos):**
                   - **Tic 1 — Reframing asindético (*«No es X, es Y»*):** Fórmulas adversativas yuxtapuestas por coma o punto y coma.
                   - **Tic 2 — Dos puntos oracionales (`[Oración] : [Oración con verbo]`):** Detecta con **spaCy** oraciones donde tras `:` hay más de 3 palabras y al menos un verbo conjugado (`VERB`/`AUX`), descartando listas y aposiciones nominales.
                """
            )

    with col_prompts:
        with st.expander("📋 Prompts listos para copiar y probar en clase (Test + Ensayo)", expanded=False):
            st.markdown("**1. Prompt para el Test Metalingüístico (Q1–Q4):**")
            st.code(get_metalinguistic_prompt_text(), language="markdown")
            st.markdown("**2. Prompts para el Texto Ensayístico (Estilometría):**")
            for item in PROMPTS_SUGERIDOS:
                st.caption(item["titulo"])
                st.code(item["prompt"], language="markdown")


def render_metalinguistic_comparison(res_a: ModelAuditResult, res_b: ModelAuditResult) -> None:
    """Muestra la tabla comparativa pregunta por pregunta del Test Metalingüístico (Q1-Q4)."""
    st.subheader("🧪 Desglose de la Prueba 1: Test Metalingüístico (Multiple Choice Q1–Q4)")

    rows = []
    for det_a, det_b in zip(res_a.meta.detalles, res_b.meta.detalles):
        q_key = det_a.question_id.lower()
        enunciado = GROUND_TRUTH_META.get(q_key, {}).get("enunciado", "")
        icon_a = "✅" if det_a.es_correcta else "❌"
        icon_b = "✅" if det_b.es_correcta else "❌"
        rows.append(
            {
                "Ítem": det_a.question_id,
                "Fenómeno Gramatical": det_a.pregunta,
                "Enunciado": enunciado,
                "Clave Oficial": det_a.respuesta_esperada,
                f"{res_a.modelo}": f"{icon_a} {det_a.respuesta_modelo}",
                f"{res_b.modelo}": f"{icon_b} {det_b.respuesta_modelo}",
            }
        )

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_model_evidence_card(res: ModelAuditResult, show_meta_metric: bool = True) -> None:
    """Muestra las métricas rápidas y las oraciones exactas detectadas para un modelo."""
    st.markdown(f"### 🔎 {res.modelo}")

    if show_meta_metric:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "Puntaje Meta",
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
            "Dos Puntos Orac.",
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
            "Dos Puntos Orac.",
            f"{res.total_dos_puntos}",
            f"{res.densidad_dos_puntos_500w:.2f} / 500w",
            delta_color="inverse",
        )

    st.markdown(f"**🟨 Tic 1 — Reframing Asindético detectado ({res.total_reframing}):**")
    if not res.reframing_matches:
        st.success("Sin ocurrencias de reframing asindético.")
    else:
        for idx, match in enumerate(res.reframing_matches, 1):
            html_sent = highlight_reframing_html(match.oracion_contexto, match.fragmento_coincidente)
            st.markdown(
                f"<div style='padding:10px 12px; background:#f8f9fa; border-left:4px solid #ffb300; "
                f"border-radius:4px; margin-bottom:8px; color:#1f2937; font-size:0.95rem;'>"
                f"<strong>{idx}.</strong> {html_sent}</div>",
                unsafe_allow_html=True,
            )

    st.markdown(f"**🟥 Tic 2 — Dos Puntos Oracionales detectados ({res.total_dos_puntos}):**")
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
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([3, 3, 2, 2])
    with ctrl1:
        run_meta = st.toggle(
            "🧪 Prueba 1: Test Metalingüístico (Multiple Choice Q1–Q4)",
            key="run_meta",
        )
    with ctrl2:
        run_style = st.toggle(
            "✍️ Prueba 2: Auditoría Estilística (Reframing + Dos Puntos)",
            key="run_style",
        )
    with ctrl3:
        st.button(
            "✨ Cargar ejemplo",
            on_click=load_demo_texts,
            use_container_width=True,
        )
    with ctrl4:
        st.button(
            "🧹 Limpiar cajas",
            on_click=clear_texts,
            use_container_width=True,
        )

    if not run_meta and not run_style:
        st.warning("Activa al menos una de las dos pruebas con los interruptores de arriba para continuar.")
        return

    # ========================================================================
    # DOS PANELES PARALELOS (MODELO 1 vs MODELO 2) SEGÚN PRUEBAS ACTIVAS
    # ========================================================================
    col_a, col_b = st.columns(2)

    with col_a:
        nombre_a = st.text_input("Nombre del Modelo 1", key="nombre_a")
        if run_meta:
            meta_input_a = st.text_area(
                "🧪 Respuestas Multiple Choice Metalingüístico (Q1–Q4) — Modelo 1:",
                key="meta_input_a",
                height=100,
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
                "✍️ Texto Ensayístico para Estilometría — Modelo 1:",
                key="texto_a",
                height=260,
                placeholder="Copia y pega aquí el ensayo generado por el Modelo 1...",
            )
        else:
            texto_a = ""

    with col_b:
        nombre_b = st.text_input("Nombre del Modelo 2", key="nombre_b")
        if run_meta:
            meta_input_b = st.text_area(
                "🧪 Respuestas Multiple Choice Metalingüístico (Q1–Q4) — Modelo 2:",
                key="meta_input_b",
                height=100,
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
                "✍️ Texto Ensayístico para Estilometría — Modelo 2:",
                key="texto_b",
                height=260,
                placeholder="Copia y pega aquí el ensayo generado por el Modelo 2...",
            )
        else:
            texto_b = ""

    # ========================================================================
    # BOTÓN DE EJECUCIÓN Y PANEL DE RESULTADOS
    # ========================================================================
    st.button(
        "▶️ Ejecutar Evaluación y Comparar Modelos",
        type="primary",
        use_container_width=True,
    )

    if run_style and not texto_a.strip() and not texto_b.strip() and not run_meta:
        st.info("Pega el texto ensayístico de al menos un modelo para calcular la estilometría.")
        return

    if run_meta and not meta_input_a.strip() and not meta_input_b.strip() and not run_style:
        st.info("Pega las respuestas del Multiple Choice (Q1–Q4) de al menos un modelo para evaluar.")
        return

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
            (nombre_a or "Modelo A", texto_a, meta_input_a),
            (nombre_b or "Modelo B", texto_b, meta_input_b),
        ],
        nlp=nlp,
        run_meta=run_meta,
        run_style=run_style,
    )

    # 1. Tabla Resumen General adaptada a las pruebas activas
    st.divider()
    st.subheader("📊 Tabla Comparativa de Resultados")
    st.dataframe(
        df_summary,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Puntaje Meta (0-4)": st.column_config.NumberColumn("Puntaje Meta (0-4)", format="%d / 4"),
            "Precisión Meta (%)": st.column_config.ProgressColumn(
                "Precisión Meta (%)",
                format="%.1f%%",
                min_value=0,
                max_value=100,
            ),
            "Densidad Reframing / 500w": st.column_config.NumberColumn(format="%.2f"),
            "Densidad Dos Puntos / 500w": st.column_config.NumberColumn(format="%.2f"),
        },
    )

    # 2. Si la Prueba 1 (Multiple Choice) está activa, mostrar desglose pregunta por pregunta
    if run_meta:
        st.divider()
        render_metalinguistic_comparison(results[0], results[1])

    # 3. Si la Prueba 2 (Estilometría) está activa, mostrar gráfico de barras y visor cualitativo
    if run_style:
        st.divider()
        st.subheader("📈 Comparativa de Densidad de Tics Sintácticos (cada 500 palabras)")
        chart_df = df_summary.set_index("Modelo")[
            ["Densidad Reframing / 500w", "Densidad Dos Puntos / 500w"]
        ]
        st.bar_chart(chart_df, height=320)

        st.divider()
        st.subheader("🔍 Evidencia Cualitativa Lado a Lado (Oraciones Detectadas)")
        st.caption(
            "Nota: En amarillo se resalta el patrón de *Reframing Asindético*; en rojo los dos puntos oracionales "
            "y en verde los verbos conjugados (`VERB`/`AUX`) que validan la segunda proposición."
        )

        ev_col_a, ev_col_b = st.columns(2)
        with ev_col_a:
            render_model_evidence_card(results[0], show_meta_metric=run_meta)
        with ev_col_b:
            render_model_evidence_card(results[1], show_meta_metric=run_meta)

    # 4. Botones de descarga en memoria
    st.divider()
    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            "⬇️ Descargar Tabla Comparativa (CSV)",
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
        )
        st.download_button(
            "📄 Descargar Log de Auditoría Detallada (TXT)",
            data=log_str.encode("utf-8"),
            file_name="auditoria_detallada.txt",
            mime="text/plain",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
