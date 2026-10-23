"""
auditor.py - Motor en memoria de evaluación metalingüística y estilometría para LLMs.

Permite activar o desactivar de forma independiente:
- Prueba 1 (Módulo A): Evaluación Metalingüística Objetiva (Multiple Choice Q1-Q4 contra Ground Truth).
- Prueba 2 (Módulos B, C y D): Auditoría Estilística Fina (Reframing Asindético y Dos Puntos Oracionales
  con spaCy, normalizados cada 500 palabras).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
import re
import subprocess
import sys
from typing import Any, Final, Pattern

import pandas as pd
import spacy
from spacy.language import Language
from spacy.tokens import Doc, Span

# ============================================================================
# CONSTANTES Y PATRONES PRECOMPILADOS
# ============================================================================

SPACY_MODEL_NAME: Final[str] = "es_core_news_lg"
SPACY_FALLBACK_MODELS: Final[tuple[str, ...]] = ("es_core_news_lg", "es_core_news_md", "es_core_news_sm")

# Expresión regular para el Módulo B: Reframing Asindético y Adversativo («No es X, es Y» / «No [era/es] [solo] X, sino Y»)
PATRON_REFRAMING: Final[Pattern[str]] = re.compile(
    r"\b(?:ya\s+)?no\s+(?:"
    r"(?:solo\s+|meramente\s+|únicamente\s+|simplemente\s+)?(?:es|era|son|eran|fue|fueron|se\s+trata(?:ba)?\s+de|consiste\s+en|consistía\s+en)\s+[^,;.:\n]+[,;]\s+(?:es|era|son|eran|fue|fueron|se\s+trata(?:ba)?\s+de|consiste\s+en|consistía\s+en)(?:\s+[^,;.:\n]+)?"
    r"|"
    r"[^,;.:\n]{2,140}[,;]\s+sino(?:\s+que|\s+de|\s+también|\s+más\s+bien)?\s+[^,;.:\n]+"
    r")",
    re.IGNORECASE,
)

GROUND_TRUTH_META: Final[dict[str, dict[str, Any]]] = {
    "q1": {
        "pregunta": "Concordancia con colectivos partitivos",
        "enunciado": (
            "¿Cuál de las siguientes oraciones presenta una concordancia ad sensum o "
            "gramaticalmente normada en construcciones partitivas con atributo?"
        ),
        "opciones": {
            "A": "La mayoría de los asistentes quedaron sumamente cansados tras la jornada.",
            "B": "Un grupo de investigadores publicaron ayer los resultados, aunque la mayoría permaneció cautelosa.",
            "C": "La mitad de los libros fue trasladados al depósito central.",
            "D": "Resto de los participantes decidieron abandonar la sala.",
        },
        "respuesta_correcta": "B",
    },
    "q2": {
        "pregunta": "Valores de 'se' y acusativo preposicional",
        "enunciado": (
            "Identifique la oración que respeta la norma panhispánica sobre construcciones "
            "impersonales reflejas con complemento directo de persona determinado:"
        ),
        "opciones": {
            "A": "Se contrataron a los nuevos especialistas para el laboratorio.",
            "B": "Se los entrevistó a los candidatos durante toda la mañana.",
            "C": "Se convocó a los delegados sindicales para una reunión urgente.",
            "D": "Se buscan a profesionales con experiencia comprobable.",
        },
        "respuesta_correcta": "C",
    },
    "q3": {
        "pregunta": "Régimen verbal y queísmo/dequeísmo",
        "enunciado": (
            "Seleccione la opción libre de queísmo y dequeísmo según el régimen "
            "preposicional del verbo o sustantivo nuclear:"
        ),
        "opciones": {
            "A": "Nos aseguraron de que el informe estaría listo antes del viernes.",
            "B": "Me alegro que hayas podido resolver el problema a tiempo.",
            "C": "Tengo la absoluta certeza de que la hipótesis inicial es correcta.",
            "D": "Resulta evidente de que las condiciones meteorológicas empeorarán."
        },
        "respuesta_correcta": "C",
    },
    "q4": {
        "pregunta": "Gerundio de posterioridad temporal",
        "enunciado": (
            "¿Cuál de las siguientes oraciones emplea el gerundio de manera normativamente "
            "válida (simultaneidad o anterioridad inmediata), evitando el gerundio de posterioridad?"
        ),
        "opciones": {
            "A": "El autor publicó su primera novela en 1920, ganando el Premio Nacional diez años más tarde.",
            "B": "Saliendo del recinto universitario, se encontró de frente con su antiguo director de tesis.",
            "C": "El vehículo impactó contra el muro de contención, siendo remolcado tres horas después.",
            "D": "Se dictó una nueva normativa fiscal, provocando el rechazo inmediato del sector meses después.",
        },
        "respuesta_correcta": "B",
    },
}

PROMPTS_SUGERIDOS: Final[list[dict[str, str]]] = [
    {
        "titulo": "Opción 1: Comentario filológico y tradición textual",
        "prompt": (
            "Escribe un ensayo expositivo-argumentativo de aproximadamente 350 palabras sobre la "
            "evolución de la glosa y el comentario gramatical desde la Antigüedad tardía (Servio, Macrobio) "
            "hasta la filología moderna. Emplea un registro académico culto y fluido."
        ),
    },
    {
        "titulo": "Opción 2: IA y estandarización retórica en la escritura",
        "prompt": (
            "Redacta una reflexión crítica de aproximadamente 350 palabras acerca del impacto de los "
            "modelos de lenguaje en la prosa ensayística contemporánea, analizando la tensión entre "
            "estandarización sintáctica y estilo individual."
        ),
    },
]

META_DEMO_MODELO_A: Final[str] = '{"q1": "B", "q2": "C", "q3": "C", "q4": "B"}'
META_DEMO_MODELO_B: Final[str] = '{"q1": "A", "q2": "C", "q3": "B", "q4": "B"}'

TEXTO_DEMO_MODELO_A: Final[str] = (
    "La historia del comentario filológico atraviesa siglos de transformaciones silenciosas. "
    "En la Antigüedad tardía destacaron dos grandes modelos: Servio y Macrobio. Mientras el primero "
    "fijaba la atención en la textura gramatical del verso virgiliano, el segundo desplegaba una "
    "erudición enciclopédica de carácter filosófico.\n\n"
    "Sin embargo, el propósito de aquellas escolias iba mucho más allá de la enseñanza escolar. "
    "No es un simple ejercicio de paráfrasis, es una verdadera arquitectura de preservación cultural. "
    "La glosa marginal operaba bajo un principio hermenéutico claro: el texto clásico contenía una "
    "verdad latente que debía ser actualizada para cada nueva generación de lectores.\n\n"
    "Con el arribo del humanismo renacentista, la relación con las fuentes experimentó un giro decisivo. "
    "No se trata de repetir la autoridad de los antiguos, se trata de reconstruir críticamente la "
    "materialidad misma del códice. Los filólogos comprendieron entonces una lección fundamental: "
    "toda transmisión textual conlleva inevitablemente una interpretación histórica."
)

TEXTO_DEMO_MODELO_B: Final[str] = (
    "Desde los márgenes de los manuscritos medievales hasta las ediciones críticas digitales, la práctica "
    "del comentario ha acompañado de manera constante la lectura de los textos canónicos. Los primeros "
    "escoliastas como Donato o Servio redactaban notas breves destinadas a esclarecer dificultades léxicas, "
    "métricas o mitológicas que podían obstaculizar la comprensión de los estudiantes en las escuelas de gramática.\n\n"
    "A lo largo de los siglos, esa labor paciente permitió conservar testimonios indirectos de obras hoy "
    "perdidas y sentó las bases metodológicas de la ecdótica moderna. No es una tarea meramente mecánica, "
    "sino que es un diálogo crítico con la tradición, donde la comparación entre variantes manuscritas exige "
    "tanto rigor lingüístico como sensibilidad histórica frente a cada etapa de transmisión."
)


def get_metalinguistic_prompt_text() -> str:
    """Genera el texto completo del prompt para el test metalingüístico listo para copiar."""
    lines = [
        'Responde únicamente en formato JSON {"q1": "...", "q2": "...", "q3": "...", "q4": "..."} '
        "seleccionando la opción gramaticalmente normativa según la NGLE de la RAE/ASALE:",
        "",
    ]
    for q_id, q_data in GROUND_TRUTH_META.items():
        lines.append(f"{q_id.upper()} ({q_data['pregunta']}): {q_data['enunciado']}")
        for opt_k, opt_v in q_data["opciones"].items():
            lines.append(f"  {opt_k}) {opt_v}")
        lines.append("")
    return "\n".join(lines).strip()


# ============================================================================
# ESTRUCTURAS DE DATOS INMUTABLES / TIPADAS
# ============================================================================


@dataclass(frozen=True)
class QuestionDetail:
    """Resultado individual por pregunta del test metalingüístico."""

    question_id: str
    pregunta: str
    respuesta_esperada: str
    respuesta_modelo: str
    es_correcta: bool


@dataclass(frozen=True)
class MetaEvaluationResult:
    """Resultado consolidado del Módulo A (Evaluación Metalingüística)."""

    puntaje_absoluto: int
    total_preguntas: int
    precision_porcentaje: float
    detalles: list[QuestionDetail] = field(default_factory=list)


@dataclass(frozen=True)
class ReframingMatch:
    """Ocurrencia detectada por el Módulo B (Reframing Asindético)."""

    fragmento_coincidente: str
    oracion_contexto: str


@dataclass(frozen=True)
class ColonClauseMatch:
    """Ocurrencia detectada por el Módulo C (Dos Puntos Oracionales)."""

    oracion_completa: str
    segmento_izquierdo: str
    segmento_derecho: str
    verbos_detectados_der: list[str]


@dataclass(frozen=True)
class ModelAuditResult:
    """Resultado integral de auditoría en memoria para un modelo."""

    modelo: str
    meta: MetaEvaluationResult
    palabras_totales: int
    oraciones_totales: int
    reframing_matches: list[ReframingMatch]
    densidad_reframing_500w: float
    dos_puntos_matches: list[ColonClauseMatch]
    densidad_dos_puntos_500w: float

    @property
    def total_reframing(self) -> int:
        return len(self.reframing_matches)

    @property
    def total_dos_puntos(self) -> int:
        return len(self.dos_puntos_matches)

    def to_summary_dict(self, include_meta: bool = True, include_style: bool = True) -> dict[str, Any]:
        """Convierte el resultado al esquema dinámico según las pruebas activas."""
        row: dict[str, Any] = {"Modelo": self.modelo}
        if include_meta:
            row["Puntaje Meta (0-4)"] = self.meta.puntaje_absoluto
            row["Precisión Meta (%)"] = round(self.meta.precision_porcentaje, 2)
        if include_style:
            row["Palabras Totales"] = self.palabras_totales
            row["Reframing (N)"] = self.total_reframing
            row["Densidad Reframing / 500w"] = round(self.densidad_reframing_500w, 2)
            row["Dos Puntos Oracionales (N)"] = self.total_dos_puntos
            row["Densidad Dos Puntos / 500w"] = round(self.densidad_dos_puntos_500w, 2)
        return row


# ============================================================================
# CARGA DEL MODELO SPACY
# ============================================================================


def load_spacy_model(preferred_model: str = SPACY_MODEL_NAME) -> tuple[Language, str]:
    """
    Carga el modelo de spaCy (`es_core_news_lg`).
    Si no está instalado localmente, intenta descargarlo automáticamente.
    """
    candidates = [preferred_model] + [m for m in SPACY_FALLBACK_MODELS if m != preferred_model]
    for model_name in candidates:
        try:
            return spacy.load(model_name), model_name
        except OSError:
            continue

    try:
        subprocess.run(
            [sys.executable, "-m", "spacy", "download", preferred_model],
            check=True,
        )
        return spacy.load(preferred_model), preferred_model
    except Exception as exc:
        raise RuntimeError(
            f"No se pudo cargar ni descargar automáticamente el modelo '{preferred_model}'. "
            f"Ejecuta en tu terminal: python -m spacy download {preferred_model}"
        ) from exc


# ============================================================================
# MÓDULO A: CORRECCIÓN METALINGÜÍSTICA EN MEMORIA (JSON O TEXTO PEGADO)
# ============================================================================


def parse_meta_input(raw_input: str | dict[str, str] | None) -> dict[str, str]:
    """
    Extrae las respuestas de q1, q2, q3, q4 a partir de:
    - Un diccionario Python `{"q1": "B", ...}`
    - Un bloque JSON pegado (incluso dentro de bloques markdown ```json ... ```)
    - Texto libre pegado como `q1: B, q2: C, q3: C, q4: B` o `1. B / 2. C / 3. C / 4. B`
    - O una secuencia simple de 4 letras `B, C, C, B`.
    """
    if not raw_input:
        return {}
    if isinstance(raw_input, dict):
        return {str(k).strip().lower(): str(v).strip().upper() for k, v in raw_input.items() if v}

    text = str(raw_input).strip()
    if not text:
        return {}

    # 1. Intentar extraer un objeto JSON `{...}` si está presente
    json_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if json_match:
        try:
            parsed_json = json.loads(json_match.group(0))
            if isinstance(parsed_json, dict):
                return {
                    str(k).strip().lower(): str(v).strip().upper()
                    for k, v in parsed_json.items()
                    if v is not None
                }
        except json.JSONDecodeError:
            pass

    # 2. Buscar patrones explícitos tipo `q1: B`, `Pregunta 1: B`, `1) B`, `1. B`, `1: B`
    extracted: dict[str, str] = {}
    for idx in range(1, 5):
        pattern = re.compile(
            rf"(?:q|pregunta\s*)?0?{idx}\s*[\)\.\:\-\=]?\s*[\"']?([A-Da-d])\b",
            re.IGNORECASE,
        )
        match = pattern.search(text)
        if match:
            extracted[f"q{idx}"] = match.group(1).upper()

    if extracted:
        return extracted

    # 3. Fallback: si el usuario pegó simplemente 4 letras separadas por coma/espacio (ej: "B, C, C, B")
    letters = re.findall(r"\b([A-Da-d])\b", text)
    if len(letters) == 4:
        return {f"q{i+1}": letters[i].upper() for i in range(4)}

    return {}


def evaluate_metalinguistic_answers(
    raw_or_dict_answers: str | dict[str, str] | None,
    ground_truth: dict[str, dict[str, Any]] = GROUND_TRUTH_META,
) -> MetaEvaluationResult:
    """
    Evalúa las respuestas del modelo para Q1-Q4 contra la clave oficial `GROUND_TRUTH_META`.
    Comparación estricta de cadenas ignorando mayúsculas y espacios.
    """
    total_questions = len(ground_truth)
    normalized = parse_meta_input(raw_or_dict_answers)

    score = 0
    detalles: list[QuestionDetail] = []

    for q_id, gt_info in ground_truth.items():
        expected = str(gt_info.get("respuesta_correcta", "")).strip().upper()
        pregunta_desc = str(gt_info.get("pregunta", q_id))
        actual = normalized.get(q_id.strip().lower(), "")

        is_correct = bool(expected) and (actual == expected)
        if is_correct:
            score += 1

        detalles.append(
            QuestionDetail(
                question_id=q_id.upper(),
                pregunta=pregunta_desc,
                respuesta_esperada=expected,
                respuesta_modelo=actual if actual else "(Vacío)",
                es_correcta=is_correct,
            )
        )

    accuracy = (score / total_questions) * 100.0 if total_questions > 0 else 0.0
    return MetaEvaluationResult(
        puntaje_absoluto=score,
        total_preguntas=total_questions,
        precision_porcentaje=accuracy,
        detalles=detalles,
    )


# ============================================================================
# MÓDULO B: DETECCIÓN DE REFRAMING ASINDÉTICO («No es X, es Y»)
# ============================================================================


def _clean_inline_whitespace(text: str) -> str:
    """Normaliza espacios y saltos de línea internos en una oración."""
    return re.sub(r"\s+", " ", text).strip()


def detect_reframing_in_doc(doc: Doc) -> list[ReframingMatch]:
    """
    Detecta construcciones de reframing asindético («No es X, es Y» / «No se trata de X, se trata de Y»)
    dentro de un documento spaCy.
    """
    matches: list[ReframingMatch] = []
    for sent in doc.sents:
        sent_text = _clean_inline_whitespace(sent.text)
        if not sent_text:
            continue
        for match in PATRON_REFRAMING.finditer(sent_text):
            matches.append(
                ReframingMatch(
                    fragmento_coincidente=match.group(0).strip(),
                    oracion_contexto=sent_text,
                )
            )
    return matches


# ============================================================================
# MÓDULO C: DETECCIÓN DE DOS PUNTOS ORACIONALES (YUXTAPOSICIÓN EXPLICATIVA)
# ============================================================================


def _is_header_or_short_line(sent: Span, sent_text: str) -> bool:
    """Descarta títulos, encabezados o líneas demasiado breves."""
    non_punct_tokens = [t for t in sent if not t.is_punct and not t.is_space]
    if len(non_punct_tokens) < 5:
        return True
    if sent_text.lstrip().startswith("#"):
        return True
    return False


def detect_colon_clauses_in_doc(doc: Doc, nlp: Language) -> list[ColonClauseMatch]:
    """
    Criterio lingüístico estricto con spaCy:
    1. Segmenta el documento en oraciones (`doc.sents`).
    2. Filtra aquellas oraciones que contengan `:` en línea (descartando líneas cortas o encabezados).
    3. Divide la oración por el primer `:` en `segmento_izq` y `segmento_der`.
    4. Procesa `segmento_der` con spaCy: debe contener al menos un token con `pos_ in ("VERB", "AUX")`
       y tener más de 3 tokens (descartando taxativamente listas y aposiciones nominales como
       «modelos: Servio y Macrobio»).
    """
    matches: list[ColonClauseMatch] = []

    for sent in doc.sents:
        raw_lines = [line.strip() for line in sent.text.splitlines() if line.strip()]
        for raw_line in raw_lines:
            if ":" not in raw_line:
                continue

            clean_line = _clean_inline_whitespace(raw_line)
            if _is_header_or_short_line(sent, clean_line):
                continue

            segmento_izq, segmento_der = clean_line.split(":", 1)
            segmento_izq = segmento_izq.strip()
            segmento_der = segmento_der.strip()

            if not segmento_izq or not segmento_der:
                continue

            doc_izq = nlp(segmento_izq)
            doc_der = nlp(segmento_der)

            tiene_verbo_izq = any(t.pos_ in ("VERB", "AUX") for t in doc_izq if not t.is_space)
            if not tiene_verbo_izq:
                continue

            tokens_der = [t for t in doc_der if not t.is_space and not t.is_punct]
            verbos_der = [
                t.text for t in doc_der if t.pos_ in ("VERB", "AUX") and not t.is_space
            ]

            if len(tokens_der) > 3 and len(verbos_der) >= 1:
                matches.append(
                    ColonClauseMatch(
                        oracion_completa=clean_line,
                        segmento_izquierdo=segmento_izq,
                        segmento_derecho=segmento_der,
                        verbos_detectados_der=verbos_der,
                    )
                )

    return matches


# ============================================================================
# MÓDULO D: NORMALIZACIÓN DE MÉTRICAS
# ============================================================================


def count_words_in_doc(doc: Doc) -> int:
    """Total de tokens en el texto que cumplen `not token.is_punct and not token.is_space`."""
    return sum(1 for token in doc if not token.is_punct and not token.is_space)


def calculate_density_per_500w(occurrences: int, total_words: int) -> float:
    """
    Fórmula de densidad cada 500 palabras:
        Densidad = (Total de ocurrencias / Total de palabras) * 500
    Maneja división por cero en caso de textos vacíos.
    """
    if total_words <= 0 or occurrences <= 0:
        return 0.0
    return (occurrences / total_words) * 500.0


# ============================================================================
# PIPELINE EN MEMORIA CON ACTIVACIÓN INDEPENDIENTE DE PRUEBAS
# ============================================================================


def audit_text_in_memory(
    model_name: str,
    raw_text: str,
    nlp: Language | None,
    meta_answers: str | dict[str, str] | None = None,
    run_meta: bool = True,
    run_style: bool = True,
) -> ModelAuditResult:
    """
    Ejecuta las pruebas seleccionadas (`run_meta` y/o `run_style`) en memoria.
    Si `run_style=False`, no requiere procesar texto con spaCy.
    """
    meta_result = (
        evaluate_metalinguistic_answers(meta_answers)
        if run_meta
        else MetaEvaluationResult(puntaje_absoluto=0, total_preguntas=len(GROUND_TRUTH_META), precision_porcentaje=0.0)
    )

    cleaned_text = raw_text.strip()
    if not run_style or not cleaned_text or nlp is None:
        return ModelAuditResult(
            modelo=model_name,
            meta=meta_result,
            palabras_totales=0,
            oraciones_totales=0,
            reframing_matches=[],
            densidad_reframing_500w=0.0,
            dos_puntos_matches=[],
            densidad_dos_puntos_500w=0.0,
        )

    doc = nlp(cleaned_text)
    total_words = count_words_in_doc(doc)
    total_sents = sum(1 for _ in doc.sents)
    reframing_matches = detect_reframing_in_doc(doc)
    colon_matches = detect_colon_clauses_in_doc(doc, nlp)

    densidad_reframing = calculate_density_per_500w(len(reframing_matches), total_words)
    densidad_dos_puntos = calculate_density_per_500w(len(colon_matches), total_words)

    return ModelAuditResult(
        modelo=model_name,
        meta=meta_result,
        palabras_totales=total_words,
        oraciones_totales=total_sents,
        reframing_matches=reframing_matches,
        densidad_reframing_500w=densidad_reframing,
        dos_puntos_matches=colon_matches,
        densidad_dos_puntos_500w=densidad_dos_puntos,
    )


def compare_models_in_memory(
    models_inputs: list[tuple[str, str, str | dict[str, str] | None]],
    nlp: Language | None,
    run_meta: bool = True,
    run_style: bool = True,
) -> tuple[pd.DataFrame, list[ModelAuditResult]]:
    """
    Recibe una lista de tuplas `(nombre_modelo, texto_ensayo, respuestas_meta)` y devuelve
    el DataFrame comparativo adaptado a las pruebas activas y la lista de resultados detallados.
    """
    results: list[ModelAuditResult] = [
        audit_text_in_memory(
            name,
            text,
            nlp,
            meta,
            run_meta=run_meta,
            run_style=run_style,
        )
        for name, text, meta in models_inputs
    ]

    df_summary = pd.DataFrame(
        [r.to_summary_dict(include_meta=run_meta, include_style=run_style) for r in results]
    )
    return df_summary, results


def format_detailed_audit_log(
    results: list[ModelAuditResult],
    spacy_model_used: str = SPACY_MODEL_NAME,
    run_meta: bool = True,
    run_style: bool = True,
) -> str:
    """Genera un informe detallado en texto plano según las pruebas activadas."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = [
        "=" * 88,
        "BENCHMARK DE EVALUACIÓN METALINGÜÍSTICA Y ESTILOMETRÍA PARA LLMs",
        f"Fecha: {timestamp}" + (f" | Modelo spaCy: {spacy_model_used}" if run_style else ""),
        "=" * 88,
        "",
    ]

    for res in results:
        lines.append("-" * 88)
        lines.append(f"MODELO: {res.modelo}")
        lines.append("-" * 88)

        if run_meta:
            lines.append(
                f"[EVALUACIÓN METALINGÜÍSTICA (MULTIPLE CHOICE)]: "
                f"{res.meta.puntaje_absoluto}/{res.meta.total_preguntas} "
                f"({res.meta.precision_porcentaje:.1f}%)"
            )
            for det in res.meta.detalles:
                estado = "CORRECTO" if det.es_correcta else "INCORRECTO"
                lines.append(
                    f"  - {det.question_id} ({det.pregunta}): "
                    f"Modelo='{det.respuesta_modelo}' | Clave='{det.respuesta_esperada}' -> [{estado}]"
                )
            lines.append("")

        if run_style:
            lines.append(
                f"[ESTADÍSTICAS DE CORPUS]: {res.palabras_totales} palabras | {res.oraciones_totales} oraciones"
            )
            lines.append("")

            lines.append(
                f"[TIC 1 - REFRAMING ASINDÉTICO («No es X, es Y»)]: "
                f"{res.total_reframing} ocurrencias (Densidad: {res.densidad_reframing_500w:.2f} / 500w)"
            )
            if res.reframing_matches:
                for idx, m in enumerate(res.reframing_matches, 1):
                    lines.append(f"  {idx}. Fragmento: «{m.fragmento_coincidente}»")
                    lines.append(f"     Oración: {m.oracion_contexto}")
            else:
                lines.append("  (Sin ocurrencias detectadas)")
            lines.append("")

            lines.append(
                f"[TIC 2 - DOS PUNTOS ORACIONALES ([Oración] : [Oración con verbo])]: "
                f"{res.total_dos_puntos} ocurrencias (Densidad: {res.densidad_dos_puntos_500w:.2f} / 500w)"
            )
            if res.dos_puntos_matches:
                for idx, m in enumerate(res.dos_puntos_matches, 1):
                    verbos = ", ".join(m.verbos_detectados_der)
                    lines.append(f"  {idx}. Oración: {m.oracion_completa}")
                    lines.append(f"     -> 2° proposición: «{m.segmento_derecho}» (Verbos: {verbos})")
            else:
                lines.append("  (Sin ocurrencias detectadas)")
            lines.append("")

    return "\n".join(lines)
