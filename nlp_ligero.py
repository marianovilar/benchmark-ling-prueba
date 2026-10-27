"""
nlp_ligero.py - Analizador morfosintáctico liviano (sin spaCy) para la versión web en GitHub Pages.

spaCy y `es_core_news_lg` (~550 MB) no pueden ejecutarse dentro del navegador (stlite/Pyodide), así que
la versión web usa este reemplazo heurístico. Imita solo la parte de la interfaz de spaCy que usa
`auditor.py`:
- `nlp(texto)` devuelve un `Doc` iterable de tokens y con `.sents` (oraciones como `Span`).
- Cada token expone `.text`, `.is_punct`, `.is_space`, `.pos_` y `.morph.get("VerbForm")`.

La detección de verbos se basa en un léxico de formas verbales del español (`datos/lexico_verbal_es.json.gz`,
generado desde `spacy-lookups-data`, licencia MIT) más reglas de contexto para descartar homógrafos
nominales («la cuenta», «el uso»). Es menos precisa que el etiquetado estadístico de spaCy.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import gzip
import json
from pathlib import Path
import re
from typing import Final, Iterator

NOMBRE_MODELO_LIGERO: Final[str] = "nlp_ligero (heurístico, versión web)"
RUTA_LEXICO: Final[Path] = Path(__file__).resolve().parent / "datos" / "lexico_verbal_es.json.gz"

PATRON_TOKEN: Final[re.Pattern[str]] = re.compile(
    r"\s+"
    r"|\d+(?:[.,]\d+)*"
    r"|[^\W\d_]+(?:['’-][^\W\d_]+)*"
    r"|\.{3}"
    r"|[^\w\s]"
    r"|_+",
)
# Fin de oración: signo final (más comillas o paréntesis de cierre) seguido de espacio, o un párrafo en blanco
PATRON_FIN_ORACION: Final[re.Pattern[str]] = re.compile(r"[.!?…]+[\"'»”’)\]]*(?=\s)|\n\s*\n")

# Palabras gramaticales que el léxico confunde con formas verbales («para» → parir, «sobre» → sobrar)
NO_VERBALES: Final[frozenset[str]] = frozenset(
    {
        "para", "como", "sobre", "entre", "bajo", "así", "embargo", "medio", "cerca", "junto", "acerca",
        "arriba", "abajo", "encima", "adelante", "adentro", "afuera", "apenas", "conforme", "salvo",
        "pese", "aparte", "respecto", "torno", "supuesto", "ademas", "dos", "uno", "cara", "nada",
        "mientras", "mediante", "durante", "según", "tras", "ante", "contra", "hacia", "hasta", "desde",
        "sino", "pues", "luego", "cuando", "donde", "aunque", "porque", "entonces", "también", "tampoco",
        "siempre", "nunca", "casi", "solo", "sólo", "muy", "más", "menos", "tan", "tanto", "bien", "mal",
        "todo", "toda", "todos", "todas", "otro", "otra", "otros", "otras", "cada", "mismo", "misma",
        "propio", "propia", "dicho", "dicha", "fin", "vez", "veces", "río", "acuerdo", "manera",
    }
)

# Contexto izquierdo que indica que la palabra siguiente es un sustantivo y no un verbo («la cuenta», «el uso»)
DETERMINANTES: Final[frozenset[str]] = frozenset(
    {
        "el", "la", "los", "las", "un", "una", "unos", "unas", "lo",
        "este", "esta", "estos", "estas", "ese", "esa", "esos", "esas", "aquel", "aquella", "aquellos", "aquellas",
        "mi", "mis", "tu", "tus", "su", "sus", "nuestro", "nuestra", "nuestros", "nuestras",
        "vuestro", "vuestra", "vuestros", "vuestras", "cuyo", "cuya", "cuyos", "cuyas",
        "cada", "todo", "toda", "todos", "todas", "otro", "otra", "otros", "otras", "algún", "alguna",
        "algunos", "algunas", "ningún", "ninguna", "mucho", "mucha", "muchos", "muchas", "poco", "poca",
        "pocos", "pocas", "varios", "varias", "tal", "tales", "dicho", "dicha", "dichos", "dichas",
        "mismo", "misma", "mismos", "mismas", "cierto", "cierta", "ciertos", "ciertas", "gran", "buen",
        "mal", "primer", "tercer", "mayor", "menor", "mejor", "peor", "nuevo", "nueva", "nuevos", "nuevas",
        "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve", "diez",
    }
)
PREPOSICIONES: Final[frozenset[str]] = frozenset(
    {
        "a", "al", "ante", "bajo", "con", "contra", "de", "del", "desde", "durante", "en", "entre", "hacia",
        "hasta", "mediante", "para", "por", "según", "sin", "sobre", "tras",
    }
)

# Terminaciones exclusivamente verbales: no hay sustantivos frecuentes que las compartan
PATRON_SUFIJO_VERBAL_INEQUIVOCO: Final[re.Pattern[str]] = re.compile(
    r"(?:"
    r"aba|abas|ábamos|abais|aban"  # pretérito imperfecto (-ar)
    r"|ía|ías|íamos|íais|ían"  # pretérito imperfecto (-er/-ir) y condicional
    r"|ó|ió|aron|ieron|eron|aste|iste|asteis|isteis|amos|emos|imos"  # pretérito perfecto simple y 1.ª plural
    r"|ré|rás|rá|remos|réis|rán"  # futuro
    r"|ara|aras|áramos|arais|aran|iera|ieras|iéramos|ierais|ieran"  # subjuntivo imperfecto (-ra)
    r"|ase|ases|ásemos|aseis|asen|iese|ieses|iésemos|ieseis|iesen"  # subjuntivo imperfecto (-se)
    r"|áis|éis"
    r")$"
)

# Formas finitas muy frecuentes que casi nunca funcionan como sustantivo (cuentan aun tras «la/lo/los/las»)
FINITAS_FRECUENTES: Final[frozenset[str]] = frozenset(
    {
        "es", "son", "era", "eran", "fue", "fueron", "sea", "sean", "será", "serán", "sería", "serían",
        "está", "están", "estaba", "estaban", "esté", "estén", "hay", "ha", "han", "he", "hemos", "había",
        "habían", "haya", "hayan", "tiene", "tienen", "tenía", "tenían", "tenga", "tengan", "puede", "pueden",
        "podía", "podían", "pueda", "puedan", "debe", "deben", "debía", "debían", "hace", "hacen", "hizo",
        "dice", "dicen", "dijo", "va", "van", "iba", "iban", "sigue", "siguen", "permite", "permiten",
        "implica", "implican", "requiere", "requieren", "exige", "exigen", "contiene", "contienen",
        "propone", "proponen", "supone", "suponen", "constituye", "constituyen", "conlleva", "conllevan",
        "parece", "parecen", "resulta", "resultan", "depende", "dependen", "consiste", "consisten",
        "existe", "existen", "ofrece", "ofrecen", "revela", "revelan", "demuestra", "demuestran",
        "representa", "representan", "significa", "significan", "produce", "producen", "genera", "generan",
        "incluye", "incluyen", "obliga", "obligan", "sirve", "sirven", "necesita", "necesitan",
    }
)
CLITICOS_AMBIGUOS: Final[frozenset[str]] = frozenset({"la", "las", "lo", "los"})

FORMAS_AUXILIARES: Final[frozenset[str]] = frozenset(
    {
        "es", "son", "era", "eran", "fue", "fueron", "sea", "sean", "será", "serán", "sería", "serían",
        "soy", "eres", "somos", "fui", "fuimos", "fuera", "fueran", "fuese", "fuesen",
        "está", "están", "estaba", "estaban", "esté", "estén", "estoy", "estás", "estamos", "estuvo",
        "estuvieron", "estará", "estarán", "estaría", "estarían",
        "ha", "han", "he", "has", "hemos", "había", "habían", "haya", "hayan", "habrá", "habrán", "habría",
        "habrían", "hubo", "hubiera", "hubieran", "hubiese", "hubiesen",
    }
)

PATRON_GERUNDIO: Final[re.Pattern[str]] = re.compile(r"(?:ando|iendo|yendo)(?:se|me|te|nos|lo|la|los|las|le|les)*$")
PATRON_GERUNDIO_ACENTUADO: Final[re.Pattern[str]] = re.compile(r"(?:ándo|iéndo|yéndo)(?:se|me|te|nos|lo|la|los|las|le|les)+$")
PATRON_PARTICIPIO: Final[re.Pattern[str]] = re.compile(r"(?:ad|id|íd)(?:o|a|os|as)$")
# Sustantivos y adjetivos frecuentes terminados en -ar/-er que el léxico registra como infinitivos
NO_INFINITIVOS: Final[frozenset[str]] = frozenset(
    {
        "lugar", "hogar", "mujer", "escolar", "militar", "popular", "similar", "particular", "familiar",
        "singular", "regular", "nuclear", "solar", "polar", "celular", "titular", "par", "mar", "bar",
        "collar", "altar", "azar", "pilar", "ayer", "taller", "alfiler", "secular", "molecular",
        "muscular", "vascular", "lunar", "estelar", "ejemplar", "peculiar", "auxiliar",
    }
)
NO_GERUNDIOS: Final[frozenset[str]] = frozenset({"cuando", "mando", "blando", "bando", "comando", "fernando", "orlando", "rolando", "segundo", "mundo", "fondo", "profundo"})


@lru_cache(maxsize=1)
def _cargar_lexico() -> tuple[frozenset[str], frozenset[str]]:
    """Devuelve (infinitivos, formas flexionadas) del léxico verbal comprimido."""
    contenido = RUTA_LEXICO.read_bytes()
    # Algunos servidores web descomprimen el .gz por su cuenta (Content-Encoding); en ese caso ya es JSON plano
    if contenido[:2] == b"\x1f\x8b":
        contenido = gzip.decompress(contenido)
    datos = json.loads(contenido.decode("utf-8"))
    return frozenset(datos["inf"]), frozenset(datos["fin"])


class Morfologia:
    """Equivalente mínimo de `spacy.tokens.MorphAnalysis`."""

    __slots__ = ("_rasgos",)

    def __init__(self, rasgos: dict[str, str] | None = None) -> None:
        self._rasgos = rasgos or {}

    def get(self, rasgo: str) -> list[str]:
        valor = self._rasgos.get(rasgo)
        return [valor] if valor else []

    def __str__(self) -> str:
        return "|".join(f"{k}={v}" for k, v in self._rasgos.items())


@dataclass(slots=True)
class Token:
    """Equivalente mínimo de `spacy.tokens.Token`."""

    text: str
    idx: int
    is_space: bool
    is_punct: bool
    pos_: str = "X"
    morph: Morfologia = Morfologia()

    def __str__(self) -> str:
        return self.text


class Span:
    """Equivalente mínimo de `spacy.tokens.Span` (una oración)."""

    def __init__(self, doc: Doc, start: int, end: int) -> None:
        self.doc = doc
        self.start = start
        self.end = end

    @property
    def text(self) -> str:
        """Texto de la oración sin el espacio final que la separa de la siguiente."""
        tokens = [t for t in self if not t.is_space]
        if not tokens:
            return ""
        return self.doc.text[tokens[0].idx : tokens[-1].idx + len(tokens[-1].text)]

    def __iter__(self) -> Iterator[Token]:
        return iter(self.doc.tokens[self.start : self.end])

    def __len__(self) -> int:
        return self.end - self.start


class Doc:
    """Equivalente mínimo de `spacy.tokens.Doc`."""

    def __init__(self, text: str, tokens: list[Token], limites_oraciones: list[tuple[int, int]]) -> None:
        self.text = text
        self.tokens = tokens
        self._limites_oraciones = limites_oraciones

    @property
    def sents(self) -> Iterator[Span]:
        for start, end in self._limites_oraciones:
            yield Span(self, start, end)

    def __iter__(self) -> Iterator[Token]:
        return iter(self.tokens)

    def __len__(self) -> int:
        return len(self.tokens)


class AnalizadorLigero:
    """Sustituto invocable de `spacy.language.Language`: `nlp(texto) -> Doc`."""

    def __init__(self) -> None:
        self.infinitivos, self.finitas = _cargar_lexico()

    def __call__(self, text: str) -> Doc:
        tokens = [
            Token(
                text=m.group(0),
                idx=m.start(),
                is_space=m.group(0).isspace(),
                is_punct=not m.group(0).isspace() and not any(c.isalnum() for c in m.group(0)),
            )
            for m in PATRON_TOKEN.finditer(text)
        ]
        limites = self._segmentar(text, tokens)
        for start, end in limites:
            self._etiquetar_oracion(tokens[start:end])
        return Doc(text, tokens, limites)

    @staticmethod
    def _segmentar(text: str, tokens: list[Token]) -> list[tuple[int, int]]:
        """Divide en oraciones por signos de cierre seguidos de espacio y por párrafos en blanco."""
        cortes = sorted({m.end() for m in PATRON_FIN_ORACION.finditer(text)})
        limites: list[tuple[int, int]] = []
        inicio = 0
        i_corte = 0
        for i, tok in enumerate(tokens):
            # El espacio posterior al cierre queda en la oración anterior, como en spaCy
            while i_corte < len(cortes) and cortes[i_corte] <= tok.idx and not tok.is_space:
                if i > inicio:
                    limites.append((inicio, i))
                inicio = i
                i_corte += 1
        if len(tokens) > inicio:
            limites.append((inicio, len(tokens)))
        # Las oraciones formadas solo por espacios se adosan a la anterior, como hace spaCy
        fusionados: list[tuple[int, int]] = []
        for start, end in limites:
            if fusionados and all(t.is_space for t in tokens[start:end]):
                fusionados[-1] = (fusionados[-1][0], end)
            else:
                fusionados.append((start, end))
        return fusionados

    def _etiquetar_oracion(self, tokens: list[Token]) -> None:
        palabras = [t for t in tokens if not t.is_space]
        for i, tok in enumerate(palabras):
            if tok.is_punct:
                tok.pos_ = "PUNCT"
                continue
            previa = palabras[i - 1].text.lower() if i > 0 and not palabras[i - 1].is_punct else ""
            inicio_oracion = i == 0 or palabras[i - 1].text in ("«", '"', "“", "(", "¿", "¡", "—", "-")
            tok.pos_, verb_form = self._clasificar(tok.text, previa, inicio_oracion)
            if verb_form:
                tok.morph = Morfologia({"VerbForm": verb_form})

    def _clasificar(self, palabra: str, previa: str, inicio_oracion: bool) -> tuple[str, str]:
        """Devuelve (pos_, VerbForm) para una palabra según el léxico y su contexto izquierdo."""
        if palabra[0].isdigit():
            return "NUM", ""
        if palabra[0].isupper() and not inicio_oracion:
            return "PROPN", ""
        forma = palabra.lower()
        if forma in NO_VERBALES or forma.endswith("mente"):
            return "X", ""

        if forma in self.finitas:
            if self._es_finita_en_contexto(forma, previa):
                return ("AUX" if forma in FORMAS_AUXILIARES else "VERB"), "Fin"
            return "NOUN", ""
        if forma in self.infinitivos and forma not in NO_INFINITIVOS:
            # «el deber», «su poder» son sustantivos; «al llegar», «para mejorar», infinitivos
            return ("NOUN", "") if previa in DETERMINANTES else ("VERB", "Inf")
        if forma not in NO_GERUNDIOS and (PATRON_GERUNDIO.search(forma) or PATRON_GERUNDIO_ACENTUADO.search(forma)):
            return "VERB", "Ger"
        if PATRON_PARTICIPIO.search(forma) and previa in FORMAS_AUXILIARES:
            return "VERB", "Part"
        return "X", ""

    @staticmethod
    def _es_finita_en_contexto(forma: str, previa: str) -> bool:
        """Descarta homógrafos nominales («la cuenta», «su uso») salvo formas inequívocamente verbales."""
        if previa not in DETERMINANTES and previa not in PREPOSICIONES:
            return True
        if previa in CLITICOS_AMBIGUOS and forma in FINITAS_FRECUENTES:
            return True
        return previa in CLITICOS_AMBIGUOS and bool(PATRON_SUFIJO_VERBAL_INEQUIVOCO.search(forma))


def load() -> AnalizadorLigero:
    """Crea el analizador liviano (equivalente a `spacy.load`)."""
    return AnalizadorLigero()
