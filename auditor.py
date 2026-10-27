"""
auditor.py - Motor en memoria de evaluación metalingüística, estilometría y detección de errores para LLMs.

Permite activar o desactivar de forma independiente las pruebas:
- Prueba 1 (Módulo A): evaluación metalingüística objetiva (opción múltiple Q1-Q4 contra ground truth).
- Prueba 2 (Módulos B, C y D): auditoría estilística en prosa (reframing asindético/adversativo y
  dos puntos oracionales con spaCy, normalizados cada 500 palabras).
- Prueba 3 (Módulo E): detección y corrección de errores de redacción y ortografía (sobre los 3 textos
  extensos de ~1000 palabras con errores claros y sutiles, o sobre un texto propio con su clave).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
import re
import subprocess
import sys
from typing import TYPE_CHECKING, Any, Final, Pattern
import unicodedata

import pandas as pd

try:
    import spacy
except ImportError:  # Versión web (stlite/Pyodide): spaCy no puede ejecutarse en el navegador
    spacy = None

import nlp_ligero

if TYPE_CHECKING:
    from spacy.language import Language
    from spacy.tokens import Doc, Span

from textos_errores import (
    TEXTO_1_CONTENIDO,
    TEXTO_2_CONTENIDO,
    TEXTO_3_CONTENIDO,
    PlantedError,
)

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
            "¿Cuál de las siguientes oraciones presenta una concordancia gramaticalmente "
            "normativa en construcciones partitivas con atributo o predicativo?"
        ),
        "opciones": {
            "A": "La mayoría de los asistentes quedó sumamente cansados tras la jornada.",
            "B": "Un grupo de investigadores quedaron muy satisfechos con los resultados obtenidos.",
            "C": "La mitad de los libros fue trasladados al depósito central.",
            "D": "El resto de los participantes permaneció callados durante toda la sesión.",
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

GROUND_TRUTH_META_FORMAL: Final[dict[str, dict[str, Any]]] = {
    "q1": {
        "pregunta": "Agramaticalidad estructural (*) vs. variación no normativa",
        "enunciado": (
            "En gramática descriptiva y formal se distingue entre variantes no normativas (sancionadas por la "
            "academia pero producidas espontáneamente por hablantes nativos) y oraciones estructuralmente "
            "agramaticales (*) que violan principios sintácticos profundos. ¿Cuál de las siguientes oraciones "
            "es estrictamente AGRAMATICAL (*) en español y no una mera variante coloquial o no normativa?"
        ),
        "opciones": {
            "A": "Habían demasiados estudiantes esperando en el pasillo de la facultad.",
            "B": "¿Qué artículo me preguntaste quién había escrito para el congreso?",
            "C": "La vecina que le presté el diccionario ayer todavía no me lo devolvió.",
            "D": "Me alegro que hayas podido resolver el problema a tiempo.",
        },
        "respuesta_correcta": "B",
    },
    "q2": {
        "pregunta": "Inacusatividad vs. inergatividad en cláusulas absolutas",
        "enunciado": (
            "Las construcciones absolutas de participio («Participio + sujeto posverbal») son gramaticales "
            "con verbos inacusativos (cuyo sujeto se origina como argumento interno), pero resultan "
            "agramaticales (*) con verbos inergativos puros (cuyo único argumento es externo/agente). "
            "Según este principio estructural, ¿cuál de las siguientes oraciones es AGRAMATICAL (*)?"
        ),
        "opciones": {
            "A": "Llegados los delegados extranjeros, comenzó la sesión plenaria.",
            "B": "Ocurrido el accidente en la autopista, se desviaron todos los vehículos.",
            "C": "Sonreídos los niños durante el acto, el director dio por finalizado el festejo.",
            "D": "Nacidos los cachorros en primavera, el veterinario recomendó esperar un mes.",
        },
        "respuesta_correcta": "C",
    },
    "q3": {
        "pregunta": "Teoría del ligamiento (Principio C) y mando-c",
        "enunciado": (
            "Según el Principio C de la teoría del ligamiento, una expresión referencial (como un nombre propio) "
            "no puede ser correferente con un pronombre que la mande-c (c-command). ¿En cuál de las siguientes "
            "oraciones es ESTRUCTURALMENTE IMPOSIBLE que el pronombre «él» (o «lo») y «Martín» refieran a la misma persona?"
        ),
        "opciones": {
            "A": "Él aseguró en la reunión que Martín presentaría la renuncia mañana.",
            "B": "Cuando él entró al aula, Martín saludó a todos los presentes.",
            "C": "La profesora que lo dirigió en la tesis dijo que Martín ganaría la beca.",
            "D": "Martín aseguró en la reunión que él presentaría la renuncia mañana.",
        },
        "respuesta_correcta": "A",
    },
    "q4": {
        "pregunta": "Polaridad negativa y alcance sintáctico (mando-c)",
        "enunciado": (
            "Las expresiones de polaridad negativa estricta (como «pegar ojo» o «mover un dedo») requieren estar "
            "bajo el mando-c directo de un inductor negativo o no verídico; si la negación está incrustada dentro "
            "de una cláusula relativa que modifica al sujeto, no manda-c al predicado matriz. ¿Cuál de las siguientes "
            "oraciones es AGRAMATICAL (*) por violar esta condición estructural?"
        ),
        "opciones": {
            "A": "Ningún médico del hospital movió un dedo para evitar la sanción.",
            "B": "Dudo que el acusado haya movido un dedo para ocultar las pruebas.",
            "C": "Tampoco anoche pudo pegar ojo el paciente de la habitación contigua.",
            "D": "El médico que no estuvo de guardia anoche pegó ojo durante toda la madrugada.",
        },
        "respuesta_correcta": "D",
    },
}

GROUND_TRUTH_VARIEDADES: Final[dict[str, dict[str, Any]]] = {
    "q1": {
        "pregunta": "Léxico regional platense («pollajería»)",
        "enunciado": (
            'La palabra "pollajería" para designar al lugar donde se puede comprar pollo y huevo corresponde a:'
        ),
        "opciones": {
            "A": "Buenos Aires",
            "B": "Concordia",
            "C": "La Plata",
            "D": "Montevideo",
        },
        "respuesta_correcta": "C",
    },
    "q2": {
        "pregunta": "Léxico coloquial rioplatense («bondi»)",
        "enunciado": 'El término "bondi" se utiliza para designar:',
        "opciones": {
            "A": "Un tren",
            "B": "Un colectivo",
            "C": "Un problema",
            "D": "Una solución",
        },
        "respuesta_correcta": "B",
    },
    "q3": {
        "pregunta": "Fraseología y lunfardo rioplatense",
        "enunciado": (
            '¿Cuál de estas expresiones no corresponde al lunfardo rioplatense para decir "no tengo dinero"?'
        ),
        "opciones": {
            "A": "estoy corto de guita",
            "B": "no tengo un mango partido al medio",
            "C": "estoy seco",
            "D": "me quedé sin un duro",
        },
        "respuesta_correcta": "D",
    },
    "q4": {
        "pregunta": "Fonética y tonada cordobesa (alargamiento vocálico pretónico)",
        "enunciado": (
            'Si la frase "Yo soy cordobés" fuese pronunciada por un hablante de español oriundo de Córdoba '
            '(Argentina), decime cuál sílaba alargaría de la palabra "cordobés":'
        ),
        "opciones": {
            "A": "Cor-",
            "B": "-do-",
            "C": "-bés",
            "D": "Ninguna, todas se pronuncian igual",
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

META_FORMAL_DEMO_MODELO_A: Final[str] = '{"q1": "B", "q2": "C", "q3": "A", "q4": "D"}'
META_FORMAL_DEMO_MODELO_B: Final[str] = '{"q1": "A", "q2": "C", "q3": "B", "q4": "D"}'

VARIEDADES_DEMO_MODELO_A: Final[str] = '{"q1": "C", "q2": "B", "q3": "D", "q4": "B"}'
VARIEDADES_DEMO_MODELO_B: Final[str] = '{"q1": "A", "q2": "B", "q3": "D", "q4": "A"}'

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


def get_metalinguistic_prompt_text(
    ground_truth: dict[str, dict[str, Any]] = GROUND_TRUTH_META,
) -> str:
    """Genera el texto completo del prompt para el test metalingüístico (1A normativo o 1B formal) listo para copiar."""
    if ground_truth is GROUND_TRUTH_META_FORMAL:
        instruction = (
            'Responde únicamente en formato JSON {"q1": "...", "q2": "...", "q3": "...", "q4": "..."} '
            "seleccionando la opción correcta según los principios de la gramática formal y descriptiva "
            "del español (juicios de agramaticalidad estructural *, inacusatividad, ligamiento y polaridad):"
        )
    else:
        instruction = (
            'Responde únicamente en formato JSON {"q1": "...", "q2": "...", "q3": "...", "q4": "..."} '
            "seleccionando la opción gramaticalmente normativa según la NGLE de la RAE/ASALE:"
        )
    lines = [
        instruction,
        "",
    ]
    for q_id, q_data in ground_truth.items():
        lines.append(f"{q_id.upper()} ({q_data['pregunta']}): {q_data['enunciado']}")
        for opt_k, opt_v in q_data["opciones"].items():
            lines.append(f"  {opt_k}) {opt_v}")
        lines.append("")
    return "\n".join(lines).strip()


def get_variedades_prompt_text() -> str:
    """Genera el texto completo del prompt para la Prueba 4 (variedades rioplatense y cordobesa) listo para copiar."""
    lines = [
        'Responde únicamente en formato JSON {"q1": "...", "q2": "...", "q3": "...", "q4": "..."} '
        "en base a tu conocimiento de las variedades del español de Argentina (variante rioplatense y variante cordobesa):",
        "",
    ]
    for q_id, q_data in GROUND_TRUTH_VARIEDADES.items():
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
    """Ocurrencia detectada por el Módulo B (Reframing Asindético/Adversativo)."""

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
class ParsedErrorCorrection:
    """Ítem individual extraído de la respuesta de corrección de un modelo."""

    numero: int
    error_identificado: str
    correccion_propuesta: str
    explicacion: str
    linea_original: str
    tiene_flecha: bool = True


@dataclass(frozen=True)
class PlantedErrorMatchDetail:
    """Resultado de verificación de un error plantado contra la respuesta del modelo."""

    error_plantado: PlantedError
    detectado: bool
    evidencia_modelo: str


@dataclass(frozen=True)
class ErrorDetectionResult:
    """Resultado consolidado del Módulo E (Detección y Corrección de Errores)."""

    items_extraidos: list[ParsedErrorCorrection] = field(default_factory=list)
    items_sin_correspondencia: list[ParsedErrorCorrection] = field(default_factory=list)
    tiene_ground_truth: bool = False
    total_plantados: int = 0
    detectados_total: int = 0
    cobertura_porcentaje: float = 0.0
    precision_aprox_porcentaje: float = 0.0
    claros_total: int = 0
    claros_detectados: int = 0
    sutiles_total: int = 0
    sutiles_detectados: int = 0
    detalles_plantados: list[PlantedErrorMatchDetail] = field(default_factory=list)
    aviso_formato: bool = False


@dataclass(frozen=True)
class CustomQuestionSpec:
    """Especificación de un ítem creado en vivo durante la clase para la Prueba 4."""

    id: str  # "q1", "q2", ...
    fenomeno: str
    enunciado: str
    modo: str  # "multiple_choice" | "frase_clave"
    opciones: dict[str, str] = field(default_factory=dict)  # Solo para multiple_choice
    respuesta_correcta: str = ""  # Letra ("A"-"D") o expresión clave esperada


@dataclass(frozen=True)
class CustomQuestionDetail:
    """Resultado individual por ítem de la Prueba 4 creada en clase."""

    question_id: str
    fenomeno: str
    enunciado: str
    respuesta_esperada: str
    respuesta_modelo: str
    es_correcta: bool


@dataclass(frozen=True)
class CustomTestResult:
    """Resultado consolidado de la Prueba 4 creada en vivo."""

    evaluado: bool = False
    puntaje_absoluto: int = 0
    total_preguntas: int = 0
    precision_porcentaje: float = 0.0
    detalles: list[CustomQuestionDetail] = field(default_factory=list)


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
    errores_result: ErrorDetectionResult = field(default_factory=ErrorDetectionResult)
    variedades_result: MetaEvaluationResult = field(
        default_factory=lambda: MetaEvaluationResult(0, 4, 0.0)
    )
    custom_result: CustomTestResult = field(default_factory=CustomTestResult)

    @property
    def total_reframing(self) -> int:
        return len(self.reframing_matches)

    @property
    def total_dos_puntos(self) -> int:
        return len(self.dos_puntos_matches)

    def to_summary_dict(
        self,
        include_meta: bool = True,
        include_style: bool = True,
        include_errors: bool = False,
        include_variedades: bool = False,
        include_custom: bool = False,
    ) -> dict[str, Any]:
        """Convierte el resultado al esquema dinámico según las pruebas activas."""
        row: dict[str, Any] = {"Modelo": self.modelo}
        if include_meta:
            row["Puntaje meta (0-4)"] = self.meta.puntaje_absoluto
            row["Precisión meta (%)"] = round(self.meta.precision_porcentaje, 2)
        if include_style:
            row["Palabras totales"] = self.palabras_totales
            row["Reframing (N)"] = self.total_reframing
            row["Densidad reframing / 500w"] = round(self.densidad_reframing_500w, 2)
            row["Dos puntos oracionales (N)"] = self.total_dos_puntos
            row["Densidad dos puntos / 500w"] = round(self.densidad_dos_puntos_500w, 2)
        if include_errors:
            if self.errores_result.tiene_ground_truth:
                row["Errores detectados"] = (
                    f"{self.errores_result.detectados_total} / {self.errores_result.total_plantados}"
                )
                row["Cobertura (%)"] = round(self.errores_result.cobertura_porcentaje, 1)
                row["Precisión aprox. (%)"] = round(self.errores_result.precision_aprox_porcentaje, 1)
                row["Claros detectados"] = (
                    f"{self.errores_result.claros_detectados} / {self.errores_result.claros_total}"
                )
                row["Sutiles detectados"] = (
                    f"{self.errores_result.sutiles_detectados} / {self.errores_result.sutiles_total}"
                )
                row["Ítems sin correspondencia"] = len(self.errores_result.items_sin_correspondencia)
            row["Ítems reportados por el modelo"] = len(self.errores_result.items_extraidos)
        if include_variedades:
            row["Puntaje variedades (0-4)"] = self.variedades_result.puntaje_absoluto
            row["Precisión variedades (%)"] = round(self.variedades_result.precision_porcentaje, 2)
        if include_custom and self.custom_result.evaluado:
            row["Puntaje prueba 5"] = (
                f"{self.custom_result.puntaje_absoluto} / {self.custom_result.total_preguntas}"
            )
            row["Precisión prueba 5 (%)"] = round(self.custom_result.precision_porcentaje, 1)
        return row


# ============================================================================
# CARGA DEL MODELO SPACY
# ============================================================================


def load_spacy_model(preferred_model: str = SPACY_MODEL_NAME) -> tuple[Language, str]:
    """
    Carga el modelo de spaCy (`es_core_news_lg`).
    Si no está instalado localmente, intenta descargarlo automáticamente.
    Si spaCy no está disponible (versión web en el navegador), usa el analizador heurístico de `nlp_ligero`.
    """
    if spacy is None:
        return nlp_ligero.load(), nlp_ligero.NOMBRE_MODELO_LIGERO

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
    """Extrae las respuestas de q1, q2, q3, q4 a partir de JSON o texto libre pegado."""
    if not raw_input:
        return {}
    if isinstance(raw_input, dict):
        return {str(k).strip().lower(): str(v).strip().upper() for k, v in raw_input.items() if v}

    text = str(raw_input).strip()
    if not text:
        return {}

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

    letters = re.findall(r"\b([A-Da-d])\b", text)
    if len(letters) == 4:
        return {f"q{i+1}": letters[i].upper() for i in range(4)}

    return {}


def evaluate_metalinguistic_answers(
    raw_or_dict_answers: str | dict[str, str] | None,
    ground_truth: dict[str, dict[str, Any]] = GROUND_TRUTH_META,
) -> MetaEvaluationResult:
    """Evalúa las respuestas del modelo para Q1-Q4 contra la clave oficial."""
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
# MÓDULO B: DETECCIÓN DE REFRAMING ASINDÉTICO Y ADVERSATIVO
# ============================================================================


def _clean_inline_whitespace(text: str) -> str:
    """Normaliza espacios y saltos de línea internos en una oración."""
    return re.sub(r"\s+", " ", text).strip()


def detect_reframing_in_doc(doc: Doc) -> list[ReframingMatch]:
    """Detecta construcciones de reframing («No es X, es Y» / «No era solo X, sino Y»)."""
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
    Detecta con spaCy oraciones del tipo:
        [Oración con verbo] : [Oración con verbo conjugado (forma finita)]
    descartando listas, aposiciones nominales y segmentos derechos con solo formas no personales.
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
                t.text
                for t in doc_der
                if t.pos_ in ("VERB", "AUX")
                and "Fin" in t.morph.get("VerbForm")
                and not t.is_space
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
    """Densidad cada 500 palabras con protección contra división por cero."""
    if total_words <= 0 or occurrences <= 0:
        return 0.0
    return (occurrences / total_words) * 500.0


# ============================================================================
# MÓDULO E: EVALUACIÓN DE DETECCIÓN Y CORRECCIÓN DE ERRORES
# ============================================================================


def normalize_error_text(text: str) -> str:
    """
    Normaliza cadenas para el matching de la Prueba 3:
    - Unicode NFC y minúsculas
    - Quita comillas simples/dobles/tipográficas, «», asteriscos y backticks
    - Colapsa espacios múltiples y elimina el espacio antes de coma
    - Conserva los puntos (necesario para casos como 'indexación.aquellos')
    """
    if not text:
        return ""
    norm = unicodedata.normalize("NFC", text).lower()
    norm = re.sub(r"[\"'“”‘’«»*`]", "", norm)
    norm = re.sub(r"\s+", " ", norm).strip()
    norm = re.sub(r"\s+,", ",", norm)
    return norm


_NORMALIZED_SOURCE_TEXTS: Final[tuple[str, ...]] = (
    normalize_error_text(TEXTO_1_CONTENIDO),
    normalize_error_text(TEXTO_2_CONTENIDO),
    normalize_error_text(TEXTO_3_CONTENIDO),
)


def _is_copied_source_paragraph(line: str) -> bool:
    """
    Verifica si una línea sin formato de lista ni flecha es simplemente un párrafo o
    oración larga copiada literalmente del texto original sin corrección.
    """
    norm_line = normalize_error_text(line)
    if len(norm_line) >= 60 and any(norm_line in src for src in _NORMALIZED_SOURCE_TEXTS):
        return True
    return False


def _parse_single_line_as_item(line: str, numero: int) -> ParsedErrorCorrection:
    """Convierte una línea en un objeto `ParsedErrorCorrection` analizando flechas y barras."""
    list_match = re.match(r"^(?:(\d+)[\.\)\-\:]|\-|\*|\•)\s*(.+)$", line)
    content = list_match.group(2).strip() if list_match else line
    num = int(list_match.group(1)) if (list_match and list_match.group(1)) else numero

    explicacion = ""
    if "|" in content:
        main_part, explicacion = [p.strip() for p in content.split("|", 1)]
    else:
        main_part = content

    arrow_split = re.split(
        r"\s*(?:->|→|=>|—>|por:|corrección:)\s*",
        main_part,
        maxsplit=1,
        flags=re.IGNORECASE,
    )
    if len(arrow_split) == 2:
        err_part = arrow_split[0].strip(" \"'“”‘’«»*`")
        corr_part_raw = arrow_split[1].strip()
        if not explicacion and (" - " in corr_part_raw or ": " in corr_part_raw):
            sub_parts = re.split(r"\s+(?:\-|\—|:)\s+", corr_part_raw, maxsplit=1)
            corr_part = sub_parts[0].strip(" \"'“”‘’«»*`")
            explicacion = sub_parts[1].strip() if len(sub_parts) > 1 else ""
        else:
            corr_part = corr_part_raw.strip(" \"'“”‘’«»*`")
        return ParsedErrorCorrection(
            numero=num,
            error_identificado=err_part,
            correccion_propuesta=corr_part,
            explicacion=explicacion,
            linea_original=line,
            tiene_flecha=True,
        )

    return ParsedErrorCorrection(
        numero=num,
        error_identificado=main_part.strip(" \"'“”‘’«»*`"),
        correccion_propuesta="",
        explicacion=explicacion,
        linea_original=line,
        tiene_flecha=False,
    )


def parse_error_correction_list(raw_response: str) -> list[ParsedErrorCorrection]:
    """
    Parsea la lista numerada o con viñetas devuelta por el LLM con los errores y correcciones.
    Soporta tanto el formato estructurado (`1. "error" -> "corrección" | explicación`) como
    variantes con flechas (`->`, `→`, `=>`, `—>`, `por:`, `corrección:`).
    """
    if not raw_response or not raw_response.strip():
        return []

    items: list[ParsedErrorCorrection] = []
    lines = [line.strip() for line in raw_response.strip().splitlines() if line.strip()]

    for line in lines:
        list_match = re.match(r"^(?:(\d+)[\.\)\-\:]|\-|\*|\•)\s*(.+)$", line)
        has_arrow = bool(
            re.search(r"(?:->|→|=>|—>|por:|corrección:)", line, flags=re.IGNORECASE)
        )
        if not list_match and not has_arrow:
            continue

        items.append(_parse_single_line_as_item(line, len(items) + 1))

    return items


def evaluate_error_detection(
    raw_model_response: str,
    planted_errors: tuple[PlantedError, ...] | None = None,
) -> ErrorDetectionResult:
    """
    Evalúa la respuesta de un modelo en el Benchmark de Detección de Errores:
    - Hace el matching exclusivamente contra ítems extraídos (sin buscar en el cuerpo libre).
    - Si el parser no extrae ítems de una respuesta no vacía, trata cada línea no vacía
      como un ítem (descartando párrafos copiados literalmente del texto fuente) y activa `aviso_formato=True`.
    - Asignación uno a uno entre ítems y errores plantados mediante `anclas` y desempate por `correcciones`.
    """
    if not raw_model_response or not raw_model_response.strip():
        if not planted_errors:
            return ErrorDetectionResult()
        claros_total = sum(1 for e in planted_errors if e.categoria.lower() == "claro")
        sutiles_total = sum(1 for e in planted_errors if e.categoria.lower() == "sutil")
        return ErrorDetectionResult(
            tiene_ground_truth=True,
            total_plantados=len(planted_errors),
            claros_total=claros_total,
            sutiles_total=sutiles_total,
            detalles_plantados=[
                PlantedErrorMatchDetail(
                    error_plantado=pe,
                    detectado=False,
                    evidencia_modelo="(No detectado)",
                )
                for pe in planted_errors
            ],
        )

    parsed_items = parse_error_correction_list(raw_model_response)
    aviso_formato = False

    if not parsed_items:
        aviso_formato = True
        raw_lines = [line.strip() for line in raw_model_response.strip().splitlines() if line.strip()]
        parsed_items = [
            _parse_single_line_as_item(line, idx)
            for idx, line in enumerate(raw_lines, start=1)
            if not _is_copied_source_paragraph(line)
        ]

    if not planted_errors:
        return ErrorDetectionResult(
            items_extraidos=parsed_items,
            tiene_ground_truth=False,
            aviso_formato=aviso_formato,
        )

    # Asignación uno a uno: cada ítem acredita a lo sumo un error plantado y viceversa
    accredited_by: dict[int, ParsedErrorCorrection] = {}
    items_sin_correspondencia: list[ParsedErrorCorrection] = []

    for item in parsed_items:
        target_norm = (
            normalize_error_text(item.error_identificado)
            if item.tiene_flecha
            else normalize_error_text(item.linea_original)
        )
        corr_norm = (
            normalize_error_text(item.correccion_propuesta)
            if item.tiene_flecha
            else normalize_error_text(item.linea_original)
        )

        candidatos: list[PlantedError] = [
            p_err
            for p_err in planted_errors
            if p_err.orden not in accredited_by
            and any(normalize_error_text(ancla) in target_norm for ancla in p_err.anclas)
        ]

        if not candidatos:
            items_sin_correspondencia.append(item)
            continue

        if len(candidatos) == 1:
            elegido = candidatos[0]
        else:
            con_correccion = [
                p_err
                for p_err in candidatos
                if any(normalize_error_text(c) in corr_norm for c in p_err.correcciones)
            ]
            if con_correccion:
                elegido = min(con_correccion, key=lambda e: e.orden)
            else:
                elegido = min(candidatos, key=lambda e: e.orden)

        accredited_by[elegido.orden] = item

    detalles: list[PlantedErrorMatchDetail] = []
    claros_total = sum(1 for e in planted_errors if e.categoria.lower() == "claro")
    sutiles_total = sum(1 for e in planted_errors if e.categoria.lower() == "sutil")
    claros_det = 0
    sutiles_det = 0

    for p_err in planted_errors:
        matched_item = accredited_by.get(p_err.orden)
        is_detected = matched_item is not None
        if is_detected:
            if p_err.categoria.lower() == "claro":
                claros_det += 1
            else:
                sutiles_det += 1

        detalles.append(
            PlantedErrorMatchDetail(
                error_plantado=p_err,
                detectado=is_detected,
                evidencia_modelo=matched_item.linea_original if matched_item else "(No detectado)",
            )
        )

    total_plantados = len(planted_errors)
    detectados_total = claros_det + sutiles_det
    cobertura = (detectados_total / total_plantados) * 100.0 if total_plantados > 0 else 0.0

    total_listados = len(parsed_items)
    con_correspondencia = total_listados - len(items_sin_correspondencia)
    precision_aprox = (
        (con_correspondencia / total_listados) * 100.0 if total_listados > 0 else 0.0
    )

    return ErrorDetectionResult(
        items_extraidos=parsed_items,
        items_sin_correspondencia=items_sin_correspondencia,
        tiene_ground_truth=True,
        total_plantados=total_plantados,
        detectados_total=detectados_total,
        cobertura_porcentaje=cobertura,
        precision_aprox_porcentaje=precision_aprox,
        claros_total=claros_total,
        claros_detectados=claros_det,
        sutiles_total=sutiles_total,
        sutiles_detectados=sutiles_det,
        detalles_plantados=detalles,
        aviso_formato=aviso_formato,
    )


# ============================================================================
# MÓDULO F: PRUEBA 4 (CONSTRUCTOR DE PRUEBA EN VIVO CON LA CLASE)
# ============================================================================


def _normalize_option_letters(raw_val: object) -> str:
    """
    Extrae las letras de opción válidas (A, B, C, D) de una cadena o lista,
    las ordena alfabéticamente sin duplicados y las une con ', ' (ej.: 'A' o 'A, C').
    """
    if isinstance(raw_val, (list, tuple, set)):
        combined = " ".join(str(x) for x in raw_val)
    else:
        combined = str(raw_val or "")
    letters = sorted({m.upper() for m in re.findall(r"\b([A-Da-d])\b", combined)})
    return ", ".join(letters)


def build_custom_test_prompt(specs: list[CustomQuestionSpec] | tuple[CustomQuestionSpec, ...]) -> str:
    """
    Construye automáticamente el prompt estructurado para la Prueba 4 creada en clase,
    según los ítems definidos por el docente y los estudiantes (soporta 1 o varias opciones correctas).
    """
    if not specs:
        return ""

    modo = specs[0].modo
    if modo == "multiple_choice":
        has_multi_answer = any(
            len(re.findall(r"\b[A-Da-d]\b", s.respuesta_correcta)) > 1 for s in specs
        )
        keys_example = ", ".join(f'"{s.id}": "..."' for s in specs)
        if has_multi_answer:
            header_instruction = (
                f"Responde únicamente en formato JSON {{{keys_example}}} indicando la opción o las opciones "
                'válidas para cada ítem (si un ítem tiene más de una opción válida, indícalas separadas por coma, por ejemplo "A, C"):'
            )
        else:
            header_instruction = (
                f"Responde únicamente en formato JSON {{{keys_example}}} seleccionando la opción "
                "gramaticalmente normativa o correcta para cada ítem:"
            )
        lines = [
            header_instruction,
            "",
        ]
        for s in specs:
            lines.append(f"{s.id.upper()} ({s.fenomeno}): {s.enunciado}")
            for opt_k, opt_v in s.opciones.items():
                lines.append(f"  {opt_k}) {opt_v}")
            lines.append("")
        return "\n".join(lines).strip()

    # Modo "frase_clave" (respuesta breve / reescritura)
    lines = [
        "Resuelve los siguientes ítems lingüísticos respondiendo en una lista numerada "
        "(1., 2., etc.), una respuesta por línea:",
        "",
    ]
    for idx, s in enumerate(specs, start=1):
        lines.append(f"{idx}. ({s.fenomeno}) {s.enunciado}")
    return "\n".join(lines).strip()


def _parse_custom_choice_input(raw_input: str, num_questions: int) -> dict[str, str]:
    """
    Extrae respuestas A/B/C/D (simples o múltiples, ej. 'A' o 'A, C')
    para cualquier cantidad de preguntas (q1..qN).
    """
    if not raw_input or not raw_input.strip():
        return {}
    text = raw_input.strip()

    json_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            if isinstance(parsed, dict):
                return {
                    str(k).strip().lower(): _normalize_option_letters(v)
                    for k, v in parsed.items()
                    if v is not None and _normalize_option_letters(v)
                }
        except json.JSONDecodeError:
            pass

    extracted: dict[str, str] = {}
    for idx in range(1, num_questions + 1):
        pattern = re.compile(
            rf"(?:^|\n)\s*(?:q|pregunta\s*)?0?{idx}\s*[\)\.\:\-\=]\s*([^\n]+)",
            re.IGNORECASE,
        )
        m = pattern.search(text)
        if m:
            norm_letters = _normalize_option_letters(m.group(1))
            if norm_letters:
                extracted[f"q{idx}"] = norm_letters

    if extracted:
        return extracted

    if num_questions == 1:
        norm_all = _normalize_option_letters(text)
        if norm_all:
            return {"q1": norm_all}

    letters = [m.upper() for m in re.findall(r"\b([A-Da-d])\b", text)]
    if len(letters) == num_questions:
        return {f"q{i+1}": letters[i] for i in range(num_questions)}

    return {}


def evaluate_custom_test(
    raw_response: str,
    specs: list[CustomQuestionSpec] | tuple[CustomQuestionSpec, ...] | None,
) -> CustomTestResult:
    """
    Evalúa de forma determinista las respuestas de un modelo en la Prueba 4 creada en clase.
    Soporta opción múltiple con una o varias respuestas correctas ('multiple_choice')
    y coincidencia de expresión clave ('frase_clave').
    """
    if not specs:
        return CustomTestResult(evaluado=False)

    total_q = len(specs)
    modo = specs[0].modo
    detalles: list[CustomQuestionDetail] = []
    score = 0

    if modo == "multiple_choice":
        parsed_ans = _parse_custom_choice_input(raw_response, total_q)
        for s in specs:
            expected = _normalize_option_letters(s.respuesta_correcta)
            actual = _normalize_option_letters(parsed_ans.get(s.id.lower(), ""))
            is_ok = bool(expected) and (actual == expected)
            if is_ok:
                score += 1
            detalles.append(
                CustomQuestionDetail(
                    question_id=s.id.upper(),
                    fenomeno=s.fenomeno,
                    enunciado=s.enunciado,
                    respuesta_esperada=expected if expected else "(Sin definir)",
                    respuesta_modelo=actual if actual else "(Vacío)",
                    es_correcta=is_ok,
                )
            )
    else:
        # Modo "frase_clave": verifica línea por línea (o por número de ítem) si contiene la expresión esperada
        raw_lines = [line.strip() for line in (raw_response or "").strip().splitlines() if line.strip()]
        for idx, s in enumerate(specs):
            expected_raw = s.respuesta_correcta.strip()
            alternativas = [
                normalize_error_text(alt)
                for alt in re.split(r"[;|]", expected_raw)
                if normalize_error_text(alt)
            ]
            model_line = raw_lines[idx] if idx < len(raw_lines) else (raw_response or "").strip()
            model_norm = normalize_error_text(model_line)
            is_ok = bool(alternativas) and any(alt in model_norm for alt in alternativas)
            if is_ok:
                score += 1
            detalles.append(
                CustomQuestionDetail(
                    question_id=s.id.upper(),
                    fenomeno=s.fenomeno,
                    enunciado=s.enunciado,
                    respuesta_esperada=expected_raw,
                    respuesta_modelo=model_line if model_line else "(Vacío)",
                    es_correcta=is_ok,
                )
            )

    precision = (score / total_q) * 100.0 if total_q > 0 else 0.0
    return CustomTestResult(
        evaluado=True,
        puntaje_absoluto=score,
        total_preguntas=total_q,
        precision_porcentaje=precision,
        detalles=detalles,
    )


# ============================================================================
# PIPELINE EN MEMORIA CON ACTIVACIÓN INDEPENDIENTE DE LAS 4 PRUEBAS
# ============================================================================


def audit_text_in_memory(
    model_name: str,
    raw_text: str,
    nlp: Language | None,
    meta_answers: str | dict[str, str] | None = None,
    error_response: str = "",
    planted_errors: tuple[PlantedError, ...] | None = None,
    variedades_answers: str | dict[str, str] | None = None,
    custom_response: str = "",
    custom_specs: list[CustomQuestionSpec] | None = None,
    run_meta: bool = True,
    run_style: bool = True,
    run_errors: bool = False,
    run_variedades: bool = False,
    run_custom: bool = False,
    meta_ground_truth: dict[str, dict[str, Any]] = GROUND_TRUTH_META,
) -> ModelAuditResult:
    """
    Ejecuta las pruebas seleccionadas (`run_meta`, `run_style`, `run_errors`, `run_variedades`, `run_custom`) en memoria.
    """
    meta_result = (
        evaluate_metalinguistic_answers(meta_answers, ground_truth=meta_ground_truth)
        if run_meta
        else MetaEvaluationResult(
            puntaje_absoluto=0,
            total_preguntas=len(meta_ground_truth),
            precision_porcentaje=0.0,
        )
    )

    errores_res = (
        evaluate_error_detection(error_response, planted_errors)
        if run_errors
        else ErrorDetectionResult()
    )

    variedades_res = (
        evaluate_metalinguistic_answers(variedades_answers, ground_truth=GROUND_TRUTH_VARIEDADES)
        if run_variedades
        else MetaEvaluationResult(
            puntaje_absoluto=0,
            total_preguntas=len(GROUND_TRUTH_VARIEDADES),
            precision_porcentaje=0.0,
        )
    )

    custom_res = (
        evaluate_custom_test(custom_response, custom_specs)
        if run_custom
        else CustomTestResult()
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
            errores_result=errores_res,
            variedades_result=variedades_res,
            custom_result=custom_res,
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
        errores_result=errores_res,
        variedades_result=variedades_res,
        custom_result=custom_res,
    )


def compare_models_in_memory(
    models_inputs: list[tuple[str, ...]],
    nlp: Language | None,
    planted_errors: tuple[PlantedError, ...] | None = None,
    custom_specs: list[CustomQuestionSpec] | None = None,
    run_meta: bool = True,
    run_style: bool = True,
    run_errors: bool = False,
    run_variedades: bool = False,
    run_custom: bool = False,
    meta_ground_truth: dict[str, dict[str, Any]] = GROUND_TRUTH_META,
) -> tuple[pd.DataFrame, list[ModelAuditResult]]:
    """
    Recibe una lista de tuplas `(nombre_modelo, texto_ensayo, respuestas_meta, respuestas_errores[, respuestas_custom[, respuestas_variedades]])`
    y devuelve el DataFrame comparativo adaptado a las pruebas activas y la lista de resultados.
    """
    results: list[ModelAuditResult] = []
    for entry in models_inputs:
        name = entry[0]
        text = entry[1] if len(entry) > 1 else ""
        meta = entry[2] if len(entry) > 2 else None
        err_resp = entry[3] if len(entry) > 3 else ""
        cust_resp = entry[4] if len(entry) > 4 else ""
        var_resp = entry[5] if len(entry) > 5 else ""
        results.append(
            audit_text_in_memory(
                name,
                text,
                nlp,
                meta,
                error_response=err_resp,
                planted_errors=planted_errors,
                variedades_answers=var_resp,
                custom_response=cust_resp,
                custom_specs=custom_specs,
                run_meta=run_meta,
                run_style=run_style,
                run_errors=run_errors,
                run_variedades=run_variedades,
                run_custom=run_custom,
                meta_ground_truth=meta_ground_truth,
            )
        )

    df_summary = pd.DataFrame(
        [
            r.to_summary_dict(
                include_meta=run_meta,
                include_style=run_style,
                include_errors=run_errors,
                include_variedades=run_variedades,
                include_custom=run_custom,
            )
            for r in results
        ]
    )
    return df_summary, results


def format_detailed_audit_log(
    results: list[ModelAuditResult],
    spacy_model_used: str = SPACY_MODEL_NAME,
    run_meta: bool = True,
    run_style: bool = True,
    run_errors: bool = False,
    run_variedades: bool = False,
    run_custom: bool = False,
) -> str:
    """Genera un informe detallado en texto plano según las pruebas activadas."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = [
        "=" * 88,
        "BENCHMARK EN VIVO: EVALUACIÓN METALINGÜÍSTICA, ESTILOMETRÍA Y DETECCIÓN DE ERRORES",
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
                f"[PRUEBA 1 - EVALUACIÓN METALINGÜÍSTICA]: "
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
                f"[PRUEBA 2 - ESTADÍSTICAS DE CORPUS]: {res.palabras_totales} palabras | {res.oraciones_totales} oraciones"
            )
            lines.append(
                f"  * Tic 1 - Reframing: {res.total_reframing} ocurr. ({res.densidad_reframing_500w:.2f} / 500w)"
            )
            for idx, m in enumerate(res.reframing_matches, 1):
                lines.append(f"    {idx}. Fragmento: «{m.fragmento_coincidente}» | Oración: {m.oracion_contexto}")
            lines.append(
                f"  * Tic 2 - Dos puntos oracionales: {res.total_dos_puntos} ocurr. ({res.densidad_dos_puntos_500w:.2f} / 500w)"
            )
            for idx, m in enumerate(res.dos_puntos_matches, 1):
                verbos = ", ".join(m.verbos_detectados_der)
                lines.append(f"    {idx}. Oración: {m.oracion_completa} (Verbos 2° prop.: {verbos})")
            lines.append("")

        if run_errors:
            er = res.errores_result
            if er.tiene_ground_truth:
                lines.append(
                    f"[PRUEBA 3 - DETECCIÓN DE ERRORES]: {er.detectados_total}/{er.total_plantados} "
                    f"(Cobertura: {er.cobertura_porcentaje:.1f}% | Precisión aprox.: {er.precision_aprox_porcentaje:.1f}%) | "
                    f"Claros: {er.claros_detectados}/{er.claros_total} | "
                    f"Sutiles: {er.sutiles_detectados}/{er.sutiles_total} | "
                    f"Ítems sin correspondencia: {len(er.items_sin_correspondencia)}"
                )
                for det_e in er.detalles_plantados:
                    est = "DETECTADO" if det_e.detectado else "OMITIDO"
                    lines.append(
                        f"  - #{det_e.error_plantado.orden} [{det_e.error_plantado.categoria}] "
                        f"«{det_e.error_plantado.fragmento_erroneo}» -> [{est}] | Evidencia: {det_e.evidencia_modelo}"
                    )
            else:
                lines.append(f"[PRUEBA 3 - DETECCIÓN DE ERRORES (TEXTO PROPIO)]: {len(er.items_extraidos)} ítems reportados")
                for item in er.items_extraidos:
                    lines.append(f"  {item.numero}. {item.error_identificado} -> {item.correccion_propuesta}")
            lines.append("")

        if run_variedades:
            vr = res.variedades_result
            lines.append(
                f"[PRUEBA 4 - RECONOCIMIENTO DE VARIEDADES DEL ESPAÑOL]: "
                f"{vr.puntaje_absoluto}/{vr.total_preguntas} ({vr.precision_porcentaje:.1f}%)"
            )
            for det_v in vr.detalles:
                est_v = "CORRECTO" if det_v.es_correcta else "INCORRECTO"
                lines.append(
                    f"  - {det_v.question_id} ({det_v.pregunta}): "
                    f"Modelo='{det_v.respuesta_modelo}' | Clave='{det_v.respuesta_esperada}' -> [{est_v}]"
                )
            lines.append("")

        if run_custom and res.custom_result.evaluado:
            cr = res.custom_result
            lines.append(
                f"[PRUEBA 5 - PRUEBA CREADA EN CLASE]: "
                f"{cr.puntaje_absoluto}/{cr.total_preguntas} ({cr.precision_porcentaje:.1f}%)"
            )
            for det_c in cr.detalles:
                est_c = "CORRECTO" if det_c.es_correcta else "INCORRECTO"
                lines.append(
                    f"  - {det_c.question_id} ({det_c.fenomeno}): "
                    f"Modelo='{det_c.respuesta_modelo}' | Clave='{det_c.respuesta_esperada}' -> [{est_c}]"
                )
            lines.append("")

    return "\n".join(lines)
