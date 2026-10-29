"""
correccion_texto.py - Prueba 3B: corrección de un texto breve evaluada por ediciones (estilo ERRANT / BEA-2019).

El modelo devuelve el texto completo corregido. Python alinea palabra por palabra el texto original con
la versión del modelo y clasifica cada cambio:
- Errores plantados: bien corregido, mal corregido u omitido.
- Cambios fuera de los errores plantados: sobrecorrecciones (cambiar algo que estaba bien).
- Trampas: construcciones correctas que suelen "corregirse" por error («solo», «sino», «de que» obligatorio).

Métricas: precisión, cobertura (recall) y F0.5, que pondera la precisión el doble que la cobertura, como en
los benchmarks de corrección gramatical (CoNLL-2014, BEA-2019): corregir de más se penaliza más que omitir.

Formato del texto de la prueba (marcado en línea):
- `{erróneo=>corrección|alternativa}`: error plantado con sus correcciones aceptadas (`{,=>}` = suprimir).
- `[[texto]]`: trampa, texto correcto que no debe modificarse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import difflib
import re
from typing import Final, Pattern

# ============================================================================
# BANCO DE TEXTOS BREVES (MARCADO EN LÍNEA)
# ============================================================================


@dataclass(frozen=True)
class ErrorPlantadoBreve:
    """Metadatos de un error plantado, en el mismo orden en que aparece en el marcado."""

    categoria: str  # "Claro" | "Sutil"
    fenomeno: str


@dataclass(frozen=True)
class TrampaBreve:
    """Metadatos de una trampa (construcción correcta que no debe modificarse)."""

    fenomeno: str


@dataclass(frozen=True)
class TextoBreveMarcado:
    id: str
    titulo: str
    marcado: str
    errores: tuple[ErrorPlantadoBreve, ...]
    trampas: tuple[TrampaBreve, ...]


TEXTO_BREVE_1: Final[TextoBreveMarcado] = TextoBreveMarcado(
    id="breve_1",
    titulo="Texto breve 1: Digitalización de la biblioteca (~170 palabras, 8 errores, 4 trampas)",
    marcado=(
        "La biblioteca central de la facultad inició en marzo un ambicioso plan de digitalización. "
        "{Habían=>Había} pocos ejemplares disponibles de las obras más consultadas, y los responsables "
        "decidieron ofrecerlas no [[solo]] en papel, [[sino]] también en formato digital. Los estudiantes de "
        "los primeros años{,=>} consultan ahora los textos {atravez=>a través} del catálogo en línea. "
        "[[Consultando]] ese catálogo, cualquier usuario puede reservar un libro en pocos minutos.\n\n"
        "Los bibliotecarios {opinan de que=>opinan que} la experiencia fue positiva, aunque pronto se dieron "
        "cuenta [[de que]] faltaban escáneres. Estamos seguros {que=>de que} el presupuesto del año próximo "
        "permitirá resolverlo; en todo caso, debería {a ver=>haber} más personal en el turno noche. El proyecto "
        "avanzó con rapidez durante el invierno{, concluyendo=>, y concluyó|y concluyó|, y terminó|y terminó|"
        ", hasta concluir|hasta concluir} en diciembre. Todavía no sabemos {cuando=>cuándo} llegarán los "
        "equipos nuevos, pero la comunidad ya celebra el cambio."
    ),
    errores=(
        ErrorPlantadoBreve("Claro", "'Haber' impersonal pluralizado"),
        ErrorPlantadoBreve("Sutil", "Coma entre sujeto y verbo"),
        ErrorPlantadoBreve("Claro", "Ortografía de la locución 'a través de'"),
        ErrorPlantadoBreve("Sutil", "Dequeísmo"),
        ErrorPlantadoBreve("Sutil", "Queísmo ('estar seguro de que')"),
        ErrorPlantadoBreve("Claro", "Homófonos 'a ver' / 'haber'"),
        ErrorPlantadoBreve("Sutil", "Gerundio de posterioridad"),
        ErrorPlantadoBreve("Claro", "Tilde diacrítica en interrogativa indirecta"),
    ),
    trampas=(
        TrampaBreve("'solo' adverbio sin tilde (norma de 2010)"),
        TrampaBreve("'sino' adversativo correcto"),
        TrampaBreve("Gerundio modal/condicional válido"),
        TrampaBreve("'darse cuenta de que' (el 'de' es obligatorio)"),
    ),
)

BANCO_TEXTOS_BREVES: Final[dict[str, TextoBreveMarcado]] = {TEXTO_BREVE_1.id: TEXTO_BREVE_1}

# ============================================================================
# TOKENIZACIÓN Y PARSEO DEL MARCADO
# ============================================================================

PATRON_TOKEN: Final[Pattern[str]] = re.compile(r"\d+(?:[.,]\d+)*|\w+|[^\w\s]", re.UNICODE)
PATRON_MARCA: Final[Pattern[str]] = re.compile(r"\{([^{}]*?)=>([^{}]*)\}|\[\[([^\[\]]+)\]\]")

# Variantes tipográficas que no cuentan como cambio (comillas curvas, puntos suspensivos, guiones)
_EQUIVALENCIAS_TIPOGRAFICAS: Final[dict[str, str]] = {
    "“": '"', "”": '"', "«": '"', "»": '"', "‘": "'", "’": "'", "…": "...", "–": "-", "—": "-",
}


def normalizar_tipografia(texto: str) -> str:
    for original, reemplazo in _EQUIVALENCIAS_TIPOGRAFICAS.items():
        texto = texto.replace(original, reemplazo)
    return texto


def tokenizar(texto: str) -> list[str]:
    """Palabras, números y signos de puntuación como tokens separados (los espacios no cuentan)."""
    return PATRON_TOKEN.findall(normalizar_tipografia(texto))


@dataclass(frozen=True)
class ZonaError:
    """Error plantado ubicado en los tokens del texto original: tokens [inicio, fin)."""

    orden: int
    categoria: str
    fenomeno: str
    inicio: int
    fin: int
    erroneo: str
    correcciones: tuple[str, ...]  # la primera es la corrección de referencia


@dataclass(frozen=True)
class ZonaTrampa:
    orden: int
    fenomeno: str
    inicio: int
    fin: int
    texto: str


@dataclass(frozen=True)
class PruebaCorreccion:
    """Texto breve listo para evaluar: original, versión corregida de referencia y zonas marcadas."""

    id: str
    titulo: str
    texto_original: str
    texto_referencia: str
    tokens_original: tuple[str, ...]
    errores: tuple[ZonaError, ...]
    trampas: tuple[ZonaTrampa, ...]


def construir_prueba(texto: TextoBreveMarcado) -> PruebaCorreccion:
    """Convierte el marcado en línea en texto original, texto de referencia y zonas de tokens."""
    partes_original: list[str] = []
    partes_referencia: list[str] = []
    tokens: list[str] = []
    errores: list[ZonaError] = []
    trampas: list[ZonaTrampa] = []
    pos = 0

    def agregar_comun(fragmento: str) -> None:
        partes_original.append(fragmento)
        partes_referencia.append(fragmento)
        tokens.extend(tokenizar(fragmento))

    for marca in PATRON_MARCA.finditer(texto.marcado):
        agregar_comun(texto.marcado[pos : marca.start()])
        pos = marca.end()
        if marca.group(3) is not None:
            contenido = marca.group(3)
            inicio = len(tokens)
            agregar_comun(contenido)
            meta = texto.trampas[len(trampas)]
            trampas.append(ZonaTrampa(len(trampas) + 1, meta.fenomeno, inicio, len(tokens), contenido))
            continue

        erroneo = marca.group(1)
        correcciones = tuple(marca.group(2).split("|"))
        inicio = len(tokens)
        partes_original.append(erroneo)
        partes_referencia.append(correcciones[0])
        tokens.extend(tokenizar(erroneo))
        meta = texto.errores[len(errores)]
        errores.append(
            ZonaError(len(errores) + 1, meta.categoria, meta.fenomeno, inicio, len(tokens), erroneo, correcciones)
        )
    agregar_comun(texto.marcado[pos:])

    if len(errores) != len(texto.errores) or len(trampas) != len(texto.trampas):
        raise ValueError(f"{texto.id}: el marcado no coincide con la cantidad de errores o trampas declarados")

    return PruebaCorreccion(
        id=texto.id,
        titulo=texto.titulo,
        texto_original="".join(partes_original),
        texto_referencia="".join(partes_referencia),
        tokens_original=tuple(tokens),
        errores=tuple(errores),
        trampas=tuple(trampas),
    )


# ============================================================================
# EVALUACIÓN POR EDICIONES
# ============================================================================


@dataclass(frozen=True)
class Edicion:
    """Cambio del modelo respecto del original: tokens [inicio, fin) del original -> texto nuevo."""

    inicio: int
    fin: int
    antes: str
    despues: str


@dataclass(frozen=True)
class ResultadoError:
    zona: ZonaError
    estado: str  # "corregido" | "mal_corregido" | "omitido"
    version_modelo: str  # cómo quedó el fragmento en el texto del modelo


@dataclass(frozen=True)
class ResultadoTrampa:
    zona: ZonaTrampa
    respetada: bool
    version_modelo: str


@dataclass(frozen=True)
class ResultadoCorreccion:
    evaluado: bool = False
    errores: list[ResultadoError] = field(default_factory=list)
    trampas: list[ResultadoTrampa] = field(default_factory=list)
    sobrecorrecciones: list[Edicion] = field(default_factory=list)  # incluye las trampas modificadas
    verdaderos_positivos: int = 0
    falsos_positivos: int = 0
    falsos_negativos: int = 0
    precision: float = 0.0
    cobertura: float = 0.0
    f05: float = 0.0
    similitud_global: float = 0.0
    aviso_reescritura: bool = False
    texto_modelo_limpio: str = ""


PATRON_SECCION_NOTAS: Final[Pattern[str]] = re.compile(
    r"^\s*(?:#+\s*)?(?:\*\*)?(?:cambios|correcciones|notas?|explicaci[oó]n|observaciones|resumen de)[^:\n]{0,40}:",
    re.IGNORECASE,
)


def limpiar_respuesta_modelo(respuesta: str) -> str:
    """
    Quita lo que no es el texto corregido: encabezados del tipo «Aquí está el texto corregido:»,
    bloques de código, separadores y una sección final de notas o cambios realizados.
    """
    lineas = [ln for ln in respuesta.strip().splitlines() if not ln.strip().startswith("```")]
    es_relleno = lambda ln: (  # noqa: E731
        not ln.strip()
        or re.fullmatch(r"\s*(?:-{3,}|\*{3,}|_{3,})\s*", ln) is not None
        or (len(ln) < 90 and ln.rstrip().endswith(":"))
    )
    while lineas and es_relleno(lineas[0]):
        lineas.pop(0)
    for i, linea in enumerate(lineas[1:], start=1):
        if PATRON_SECCION_NOTAS.match(linea):
            lineas = lineas[:i]
            break
    while lineas and es_relleno(lineas[-1]):
        lineas.pop()
    return "\n".join(lineas).strip()


def _une(tokens: list[str] | tuple[str, ...]) -> str:
    """Une tokens para mostrarlos, sin espacio antes de los signos de cierre."""
    texto = " ".join(tokens)
    texto = re.sub(r"\s+([,.;:!?)\]»”])", r"\1", texto)
    return re.sub(r"([¿¡(\[«“])\s+", r"\1", texto)


def _se_superpone(tag_i1: int, tag_i2: int, inicio: int, fin: int) -> bool:
    """¿Un cambio en [tag_i1, tag_i2) del original toca la zona [inicio, fin)? Las inserciones en el borde cuentan."""
    if tag_i1 == tag_i2:  # inserción
        return inicio <= tag_i1 <= fin
    if inicio == fin:  # zona vacía (error por omisión)
        return tag_i1 <= inicio <= tag_i2
    return tag_i1 < fin and tag_i2 > inicio


def _mapear(posicion: int, opcodes: list[tuple[str, int, int, int, int]], hacia_el_final: bool) -> int:
    """Traduce una posición de token del original a la posición equivalente en el texto del modelo."""
    candidatos: list[int] = []
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal" and i1 <= posicion <= i2:
            candidatos.append(j1 + (posicion - i1))
        elif tag != "equal":
            if posicion == i1:
                candidatos.append(j1)
            if posicion == i2:
                candidatos.append(j2)
            if i1 < posicion < i2:  # dentro de un cambio: se toma el cambio entero
                candidatos.append(j2 if hacia_el_final else j1)
    return max(candidatos) if hacia_el_final else min(candidatos)


def evaluar_correccion(prueba: PruebaCorreccion, respuesta_modelo: str) -> ResultadoCorreccion:
    """Compara el texto corregido por el modelo con el original y clasifica cada edición."""
    texto_modelo = limpiar_respuesta_modelo(respuesta_modelo or "")
    if not texto_modelo:
        return ResultadoCorreccion()

    original = list(prueba.tokens_original)
    modelo = tokenizar(texto_modelo)
    matcher = difflib.SequenceMatcher(None, original, modelo, autojunk=False)
    opcodes = matcher.get_opcodes()
    cambios = [op for op in opcodes if op[0] != "equal"]
    usados: set[int] = set()

    resultados_errores: list[ResultadoError] = []
    for zona in prueba.errores:
        tocan = [k for k, (_, i1, i2, _, _) in enumerate(cambios) if _se_superpone(i1, i2, zona.inicio, zona.fin)]
        inicio = min([zona.inicio] + [cambios[k][1] for k in tocan])
        fin = max([zona.fin] + [cambios[k][2] for k in tocan])
        version = modelo[_mapear(inicio, opcodes, False) : _mapear(fin, opcodes, True)]
        if not tocan:
            estado = "omitido"
        else:
            usados.update(tocan)
            # Cada corrección aceptada se compara en el mismo contexto ampliado que abarcó el cambio del modelo
            aceptadas = [
                original[inicio : zona.inicio] + tokenizar(c) + original[zona.fin : fin] for c in zona.correcciones
            ]
            estado = "corregido" if version in aceptadas else "mal_corregido"
        resultados_errores.append(ResultadoError(zona, estado, _une(version)))

    sobrecorrecciones = [
        Edicion(i1, i2, _une(original[i1:i2]), _une(modelo[j1:j2]))
        for k, (_, i1, i2, j1, j2) in enumerate(cambios)
        if k not in usados
    ]

    resultados_trampas: list[ResultadoTrampa] = []
    for zona in prueba.trampas:
        tocada = any(_se_superpone(e.inicio, e.fin, zona.inicio, zona.fin) for e in sobrecorrecciones)
        inicio_m = _mapear(zona.inicio, opcodes, False)
        fin_m = _mapear(zona.fin, opcodes, True)
        resultados_trampas.append(ResultadoTrampa(zona, not tocada, _une(modelo[inicio_m:fin_m])))

    vp = sum(1 for r in resultados_errores if r.estado == "corregido")
    mal = sum(1 for r in resultados_errores if r.estado == "mal_corregido")
    fp = mal + len(sobrecorrecciones)
    fn = len(prueba.errores) - vp
    precision = vp / (vp + fp) if (vp + fp) else 0.0
    cobertura = vp / (vp + fn) if (vp + fn) else 0.0
    f05 = (1.25 * precision * cobertura / (0.25 * precision + cobertura)) if (precision + cobertura) else 0.0
    similitud = matcher.ratio()

    return ResultadoCorreccion(
        evaluado=True,
        errores=resultados_errores,
        trampas=resultados_trampas,
        sobrecorrecciones=sobrecorrecciones,
        verdaderos_positivos=vp,
        falsos_positivos=fp,
        falsos_negativos=fn,
        precision=precision * 100.0,
        cobertura=cobertura * 100.0,
        f05=f05 * 100.0,
        similitud_global=similitud * 100.0,
        # Si el modelo resumió o reescribió el texto, la alineación por ediciones deja de ser significativa
        aviso_reescritura=similitud < 0.85,
        texto_modelo_limpio=texto_modelo,
    )


def formatear_informe_correccion(prueba: PruebaCorreccion, nombre_modelo: str, res: ResultadoCorreccion) -> str:
    """Bloque de texto plano con el resultado de la Prueba 3B para el informe descargable."""
    if not res.evaluado:
        return f"[PRUEBA 3B - CORRECCIÓN DE TEXTO BREVE] {nombre_modelo}: sin texto para evaluar.\n"
    estados = {"corregido": "BIEN CORREGIDO", "mal_corregido": "MAL CORREGIDO", "omitido": "OMITIDO"}
    lineas = [
        f"[PRUEBA 3B - CORRECCIÓN DE TEXTO BREVE] {nombre_modelo} ({prueba.titulo})",
        f"  F0.5: {res.f05:.1f}% | Precisión: {res.precision:.1f}% | Cobertura: {res.cobertura:.1f}% "
        f"| VP: {res.verdaderos_positivos} FP: {res.falsos_positivos} FN: {res.falsos_negativos} "
        f"| Trampas respetadas: {sum(t.respetada for t in res.trampas)}/{len(res.trampas)}",
    ]
    for r in res.errores:
        lineas.append(
            f"  - #{r.zona.orden} [{r.zona.categoria}] «{r.zona.erroneo.strip()}» -> {estados[r.estado]} "
            f"(versión del modelo: «{r.version_modelo}»)"
        )
    for t in res.trampas:
        lineas.append(f"  - Trampa «{t.zona.texto}»: {'respetada' if t.respetada else f'modificada a «{t.version_modelo}»'}")
    for e in res.sobrecorrecciones:
        lineas.append(f"  - Sobrecorrección: «{e.antes}» -> «{e.despues}»")
    return "\n".join(lineas) + "\n"


def construir_prompt_correccion(prueba: PruebaCorreccion) -> str:
    """Prompt que pide el texto completo corregido con cambios mínimos."""
    return (
        "Actúa como corrector profesional de textos en español según la norma panhispánica (RAE / ASALE).\n"
        "Corrige el siguiente texto: ortografía, acentuación, puntuación, gramática (concordancia, régimen "
        "preposicional, queísmo, dequeísmo, gerundios) e impropiedades léxicas.\n\n"
        "INSTRUCCIONES OBLIGATORIAS:\n"
        "- Haz solo las correcciones necesarias (edición mínima): no reformules, no cambies el estilo ni el "
        "vocabulario si son correctos.\n"
        "- Devuelve ÚNICAMENTE el texto completo corregido, con los mismos párrafos, sin comentarios, sin "
        "explicaciones, sin listas de cambios y sin marcar las correcciones.\n\n"
        "TEXTO:\n"
        "---\n"
        f"{prueba.texto_original}\n"
        "---"
    )


# Respuestas de demostración: A corrige todo salvo el gerundio y cae en la trampa de «sólo»;
# B omite errores sutiles, corrige mal uno y reformula una frase correcta
_PRUEBA_DEMO: Final[PruebaCorreccion] = construir_prueba(TEXTO_BREVE_1)
DEMO_CORRECCION_MODELO_A: Final[str] = (
    _PRUEBA_DEMO.texto_referencia.replace("no solo en papel", "no sólo en papel").replace(
        "invierno, y concluyó en diciembre", "invierno, concluyendo en diciembre"
    )
)
DEMO_CORRECCION_MODELO_B: Final[str] = (
    _PRUEBA_DEMO.texto_original.replace("Habían", "Había")
    .replace("atravez", "através")
    .replace("debería a ver", "debería haber")
    .replace("en pocos minutos", "en apenas unos minutos")
)
