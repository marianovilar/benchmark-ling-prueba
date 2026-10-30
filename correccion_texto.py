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
    titulo="Texto 1: Digitalización de la biblioteca (~170 palabras, 8 errores, 4 trampas)",
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

TEXTO_BREVE_2: Final[TextoBreveMarcado] = TextoBreveMarcado(
    id="breve_2",
    titulo="Texto 2: Mudanza del archivo municipal (~450 palabras, 18 errores, 10 trampas) — más difícil",
    marcado=(
        "El Archivo Histórico Municipal atraviesa desde hace dos años una mudanza que, según sus responsables, "
        "no tiene precedentes en la ciudad. [[Debe de haber]] unos cuarenta mil legajos repartidos entre el "
        "edificio viejo y el nuevo depósito, y cada uno requiere un registro detallado antes del traslado. Para "
        "coordinar la tarea, la dirección redactó un [[guion]] de trabajo que distribuye los fondos por series y "
        "fija plazos semanales. {Se necesita=>Se necesitan|Hacen falta} más de veinte voluntarios para cumplir ese "
        "calendario, pero hasta ahora [[solo]] se inscribieron doce.\n\n"
        "Los primeros meses fueron los más difíciles. {Hubieron=>Hubo} protestas de algunos investigadores, que "
        "temían perder el acceso a la documentación durante meses, y la directora tuvo que reunirse con ellos "
        "varias veces. {La=>Le} dijeron, con cierta dureza, que el cronograma no contemplaba las consultas "
        "urgentes. [[Este]] fue, quizás, el primer aprendizaje del equipo: una mudanza no mueve solo cajas, "
        "[[sino]] también hábitos y expectativas. {A parte=>Aparte} de eso, la institución no había logrado "
        "{proveer=>prever|anticipar|calcular|estimar} el costo real de los materiales de embalaje, que se duplicó en pocas semanas.\n\n"
        "Los técnicos del área de conservación preventiva del archivo y de la biblioteca universitaria "
        "asociada{,=>} recomendaron revisar cada caja antes de cerrarla. [[Revisando]] las cajas de la serie "
        "notarial, una archivista encontró un expediente del siglo XVIII que nadie había catalogado. El documento "
        "fue trasladado de inmediato a una sala climatizada{, siendo digitalizado=>, y fue digitalizado|"
        "y fue digitalizado|, y se digitalizó|y se digitalizó} meses más tarde por un equipo externo. Hoy "
        "{se haya=>se halla} en la sala de consulta, a disposición del público. Por suerte, el hallazgo resultó "
        "{inicuo=>inocuo|inofensivo} para el cronograma: no demoró el traslado ni un solo día.\n\n"
        "Según el último informe, {van a haber=>va a haber} cambios en el horario de atención mientras dure la "
        "segunda etapa. {Es posible de que=>Es posible que} la sala de lectura cierre por completo durante el "
        "verano, aunque la directora está convencida [[de que]] podrá habilitarse una consulta digital "
        "[[provisoria]]. Muchos usuarios todavía no entienden {el por qué=>el porqué} de estas restricciones, y "
        "{aun=>aún|todavía} no se publicó un comunicado oficial que las explique. Mientras tanto, en {este aula=>esta "
        "aula} del primer piso se reúne cada lunes el equipo de coordinación, y en la entrada colocaron una "
        "{vaya=>valla} para ordenar la fila de quienes retiran copias.\n\n"
        "La directora recuerda aquella primera etapa con humor. «{Me acuerdo que=>Me acuerdo de que|Recuerdo que} nadie sabía "
        "por dónde empezar», cuenta. «La experiencia {a sido=>ha sido} agotadora, pero el equipo aprendió a pedir "
        "ayuda a tiempo». [[Si no]] llegan nuevos fondos, la última etapa se postergará hasta el año próximo; "
        "[[aun así]], el equipo confía en terminar el traslado antes de que venza el contrato de alquiler. Esta "
        "{ves=>vez}, asegura la directora, el archivo dejará de ser un depósito de papeles viejos para convertirse "
        "en un espacio de consulta abierto a toda la comunidad."
    ),
    errores=(
        ErrorPlantadoBreve("Sutil", "Concordancia en pasiva refleja con sujeto plural"),
        ErrorPlantadoBreve("Claro", "'Haber' impersonal pluralizado"),
        ErrorPlantadoBreve("Sutil", "Laísmo"),
        ErrorPlantadoBreve("Claro", "'Aparte' (adverbio) escrito en dos palabras"),
        ErrorPlantadoBreve("Sutil", "Paronimia 'proveer' / 'prever'"),
        ErrorPlantadoBreve("Sutil", "Coma entre sujeto (largo) y verbo"),
        ErrorPlantadoBreve("Sutil", "Gerundio de posterioridad"),
        ErrorPlantadoBreve("Claro", "Homófonos 'haya' / 'halla'"),
        ErrorPlantadoBreve("Sutil", "Paronimia 'inicuo' / 'inocuo'"),
        ErrorPlantadoBreve("Sutil", "'Haber' impersonal pluralizado en perífrasis"),
        ErrorPlantadoBreve("Sutil", "Dequeísmo"),
        ErrorPlantadoBreve("Claro", "'Porqué' sustantivo"),
        ErrorPlantadoBreve("Sutil", "Tilde diacrítica de 'aún' (= todavía)"),
        ErrorPlantadoBreve("Sutil", "Demostrativo ante femenino con /a/ tónica"),
        ErrorPlantadoBreve("Claro", "Homófonos 'vaya' / 'valla'"),
        ErrorPlantadoBreve("Sutil", "Queísmo ('acordarse de que')"),
        ErrorPlantadoBreve("Claro", "'Ha' (verbo haber) sin hache"),
        ErrorPlantadoBreve("Claro", "'Vez' con s"),
    ),
    trampas=(
        TrampaBreve("'Deber de' + infinitivo de probabilidad (correcto)"),
        TrampaBreve("'guion' sin tilde (norma de 2010)"),
        TrampaBreve("'solo' adverbio sin tilde (norma de 2010)"),
        TrampaBreve("Demostrativo 'este' sin tilde (norma de 2010)"),
        TrampaBreve("'sino' adversativo correcto"),
        TrampaBreve("Gerundio de simultaneidad válido"),
        TrampaBreve("'estar convencido de que' (el 'de' es obligatorio)"),
        TrampaBreve("'provisoria' (americanismo válido)"),
        TrampaBreve("'si no' condicional correcto"),
        TrampaBreve("'aun así' (= incluso así) sin tilde"),
    ),
)

TEXTO_BREVE_3: Final[TextoBreveMarcado] = TextoBreveMarcado(
    id="breve_3",
    titulo="Texto 3: Restauración del teatro del barrio (~1000 palabras, 45 errores, 19 trampas) — el más exigente",
    marcado=(
        "{Hacen=>Hace} casi tres años que el Teatro Alameda permanece cerrado. El edificio, inaugurado en "
        "1912, [[debe de]] rondar los cuatro mil metros cuadrados, y durante décadas fue el centro de la vida"
        " cultural del barrio. En {Marzo=>marzo} de 2023, un temporal {infringió=>infligió|causó|provocó} "
        "graves daños a la cubierta, y el municipio {tubo=>tuvo} que clausurar la sala de manera preventiva. "
        "Desde entonces, los vecinos reclaman una restauración integral, y no [[solo]] una reparación de "
        "emergencia que permita reabrir las puertas por unos meses.\n\n"
        "La comisión de vecinos y comerciantes de las cuadras que rodean el viejo teatro{,=>} presentó en "
        "abril un petitorio con más de seis mil firmas. «{Habemos=>Somos} muchos los que crecimos viendo "
        "funciones en esa sala», explicó su presidenta, que {piensa de que=>piensa que} el edificio forma "
        "parte de la identidad del barrio. Durante la reunión con el Concejo Deliberante, los integrantes de "
        "la comisión {le=>les} explicaron a los funcionarios que la demora agravaba el deterioro y que, "
        "{contra más=>cuanto más|mientras más} se postergara la obra, más cara resultaría. La propuesta, sin "
        "embargo, pasó [[desapercibida]] en los medios locales durante varias semanas.\n\n"
        "Los estudios técnicos {rebelaron=>revelaron} un panorama mucho más grave de lo previsto. Las vigas "
        "del escenario estaban podridas y la instalación eléctrica no cumplía ninguna norma vigente. "
        "[[Mientras más]] avanzaba el relevamiento, más problemas aparecían. En el subsuelo, los técnicos "
        "encontraron una caja {conteniendo=>que contenía|con} los planos originales del edificio, que nadie "
        "había visto desde los años cuarenta. El hallazgo {fué=>fue} celebrado por los historiadores, porque "
        "permitió reconstruir con precisión las molduras de los palcos y el diseño {exhuberante=>exuberante} "
        "del telón de boca.\n\n"
        "La obra quedó en manos de un arquitecto {que su=>cuyo} estudio ya había restaurado otras dos salas "
        "históricas de la provincia. El mismo profesional [[guio]] a los periodistas por el edificio en una "
        "recorrida que duró casi dos horas. «No {se=>sé} si alguna vez vi un deterioro semejante», admitió "
        "frente al escenario. «[[Aun]] los técnicos más experimentados se sorprendieron. {Sobretodo=>Sobre "
        "todo|Sobre todo,} nos preocupa la cúpula, que podría ceder si no se refuerza antes del invierno». "
        "[[Aquel]] fue el primer anuncio oficial sobre el riesgo estructural, y la noticia se difundió "
        "rápidamente gracias a un [[video]] que grabaron los propios vecinos.\n\n"
        "El financiamiento fue el tema más discutido. El presupuesto original [[preveía]] una inversión de "
        "cuatro millones, pero la inflación y los imprevistos lo duplicaron en pocos meses. "
        "{Preveyendo=>Previendo} nuevas subas, el Concejo aprobó una partida adicional en julio{, firmándose "
        "el convenio=>, y el convenio se firmó|y el convenio se firmó} dos semanas después. {Sin embargo=>Sin"
        " embargo,} el dinero no alcanzaba para todo. Algunos concejales aseguran que, {si el municipio "
        "habría=>si el municipio hubiera|si el municipio hubiese} aceptado la oferta de un inversor privado, "
        "la obra ya estaría terminada; otros sostienen que esa propuesta habría convertido el teatro en un "
        "salón de eventos.\n\n"
        "El debate tuvo también un costado judicial. El {ex intendente=>exintendente}, que impulsó el "
        "proyecto original, fue {absorbido=>absuelto} el año pasado de las acusaciones de sobreprecios que "
        "pesaban sobre aquella licitación, y hoy apoya la restauración desde la oposición. El [[ex secretario"
        " general]] del sindicato de actores se sumó al reclamo con una carta abierta. A la actual intendenta"
        " le {preocupa=>preocupan} los plazos más que el costo, y por eso exigió que, {apartir=>a partir} de "
        "enero, {habrán=>habrá} inspecciones mensuales con un informe público de avance.\n\n"
        "Las obras comenzaron finalmente en septiembre. Un grupo de jóvenes del barrio se formó en un taller "
        "municipal de carpintería, y los docentes destacaron su {actitud=>aptitud} para el trabajo con "
        "maderas nobles. [[Trabajando]] en turnos rotativos, los aprendices lograron recuperar buena parte de"
        " las butacas originales. Según el contrato, la empresa constructora {debe de=>debe} entregar la "
        "primera etapa antes del verano, [[con la condición de que]] contrate mano de obra local. En una "
        "asamblea reciente, la mayoría de los vecinos [[apoyaron]] esa cláusula, aunque muchos dudan de que "
        "la empresa la cumpla.\n\n"
        "El taller despertó, además, vocaciones inesperadas. «Yo no sabía nada de carpintería, pero el "
        "profesor me enseñó a tener paciencia», cuenta uno de los aprendices, de diecinueve años. "
        "«{Ojala=>Ojalá} pueda seguir trabajando acá cuando terminen las obras». Su instructor asegura que "
        "[[ese]] es el mayor logro del proyecto: que los jóvenes sientan el teatro como algo propio y no como"
        " un edificio ajeno. Según sus registros, {han habido=>ha habido} más de cuarenta inscriptos desde la"
        " primera convocatoria, aunque [[solo]] quince completaron la formación.\n\n"
        "La restauración incluirá no solo la sala principal, {si no=>sino} también el foyer y la antigua "
        "confitería, que volverá a funcionar como café literario. El proyecto se [[adecua]] a la normativa de"
        " accesibilidad: habrá rampas, un ascensor y {mas=>más} de veinte ubicaciones para personas con "
        "movilidad reducida. La directora del teatro se {enteró que=>enteró de que} el ascensor no cabía en "
        "el hueco previsto cuando la obra ya estaba avanzada, y el arquitecto tuvo que rediseñar el acceso "
        "lateral. Todavía no está claro {por que=>por qué} ese error no se detectó antes, pero la empresa "
        "[[recién]] lo admitió la semana pasada.\n\n"
        "La {abertura=>apertura} de la temporada está prevista para abril del año próximo, {a pesar que=>a "
        "pesar de que} todavía {faltan=>falta} definir algunos detalles de la programación. "
        "{Tambien=>También} se planea un ciclo de teatro independiente con entradas a precios populares. La "
        "propuesta final {satisfació=>satisfizo} a casi todos los sectores, y la comisión considera que es la"
        " {solución más óptima=>solución óptima|mejor solución} para un edificio de estas "
        "características.\n\n"
        "Para los vecinos más antiguos, la espera tiene un sabor especial. «Nos {han echo=>han hecho} muchas "
        "promesas», recuerda una mujer de ochenta años que vive enfrente del teatro desde su infancia. "
        "«Cuando éramos chicos {ibamos=>íbamos} todos los domingos, y la sala nunca estaba {bacía=>vacía}». "
        "Cuando un funcionario intentó minimizar los retrasos durante la última asamblea, la mujer no se "
        "{cayó=>calló}: le recordó {un sin número=>un sinnúmero} de anuncios incumplidos y le pidió una fecha"
        " concreta. Nadie le respondió en ese momento. Para el barrio, el Alameda no es un edificio "
        "cualquiera, [[sino]] un lugar de encuentro entre generaciones. [[Si no]] se cumplen los plazos, la "
        "comisión volverá a movilizarse; mientras tanto, cada sábado organiza visitas guiadas por la vereda "
        "para que el {publico=>público} pueda seguir de cerca los avances de la obra."
    ),
    errores=(
        ErrorPlantadoBreve("Sutil", "'Hacer' impersonal pluralizado"),
        ErrorPlantadoBreve("Claro", "Mayúscula en nombre de mes"),
        ErrorPlantadoBreve("Sutil", "Paronimia 'infringir' / 'infligir'"),
        ErrorPlantadoBreve("Claro", "Homófonos 'tubo' / 'tuvo'"),
        ErrorPlantadoBreve("Sutil", "Coma entre sujeto (largo) y verbo"),
        ErrorPlantadoBreve("Sutil", "'Habemos' por 'somos'"),
        ErrorPlantadoBreve("Sutil", "Dequeísmo"),
        ErrorPlantadoBreve("Sutil", "'Le' por 'les' con complemento indirecto plural duplicado"),
        ErrorPlantadoBreve("Sutil", "'Contra más' (vulgarismo) por 'cuanto más'"),
        ErrorPlantadoBreve("Claro", "Homófonos 'rebelar' / 'revelar'"),
        ErrorPlantadoBreve("Sutil", "Gerundio especificativo (complemento de un sustantivo)"),
        ErrorPlantadoBreve("Claro", "Tilde en el monosílabo 'fue'"),
        ErrorPlantadoBreve("Claro", "'Exuberante' sin hache"),
        ErrorPlantadoBreve("Sutil", "Quesuismo ('que su' por 'cuyo')"),
        ErrorPlantadoBreve("Claro", "Tilde diacrítica de 'sé' (verbo saber)"),
        ErrorPlantadoBreve("Claro", "'Sobre todo' (locución adverbial) en dos palabras"),
        ErrorPlantadoBreve("Sutil", "Conjugación de 'prever' ('preveyendo' por 'previendo')"),
        ErrorPlantadoBreve("Sutil", "Gerundio de posterioridad"),
        ErrorPlantadoBreve("Sutil", "Coma tras conector discursivo ('sin embargo')"),
        ErrorPlantadoBreve("Sutil", "Condicional en la prótasis ('si habría' por 'si hubiera')"),
        ErrorPlantadoBreve("Sutil", "Prefijo 'ex-' unido a base simple (norma de 2010)"),
        ErrorPlantadoBreve("Sutil", "Paronimia 'absorber' / 'absolver'"),
        ErrorPlantadoBreve("Sutil", "Concordancia con sujeto pospuesto ('le preocupan los plazos')"),
        ErrorPlantadoBreve("Claro", "'A partir' en dos palabras"),
        ErrorPlantadoBreve("Claro", "'Haber' impersonal pluralizado"),
        ErrorPlantadoBreve("Sutil", "Paronimia 'actitud' / 'aptitud'"),
        ErrorPlantadoBreve("Sutil", "'Deber de' con valor de obligación"),
        ErrorPlantadoBreve("Claro", "Tilde de 'ojalá'"),
        ErrorPlantadoBreve("Sutil", "'Haber' impersonal pluralizado en tiempo compuesto"),
        ErrorPlantadoBreve("Claro", "'Sino' adversativo escrito 'si no'"),
        ErrorPlantadoBreve("Claro", "Tilde de 'más' (adverbio de cantidad)"),
        ErrorPlantadoBreve("Sutil", "Queísmo ('enterarse de que')"),
        ErrorPlantadoBreve("Claro", "Tilde de 'por qué' en interrogativa indirecta"),
        ErrorPlantadoBreve("Sutil", "Paronimia 'abertura' / 'apertura'"),
        ErrorPlantadoBreve("Sutil", "Queísmo ('a pesar de que')"),
        ErrorPlantadoBreve("Sutil", "Concordancia con sujeto de infinitivo ('falta definir')"),
        ErrorPlantadoBreve("Claro", "Tilde de 'también'"),
        ErrorPlantadoBreve("Sutil", "Conjugación de 'satisfacer' ('satisfació' por 'satisfizo')"),
        ErrorPlantadoBreve("Sutil", "Superlativo redundante ('más óptima')"),
        ErrorPlantadoBreve("Claro", "Homófonos 'echo' / 'hecho'"),
        ErrorPlantadoBreve("Claro", "Tilde de 'íbamos'"),
        ErrorPlantadoBreve("Claro", "Homófonos 'bacía' / 'vacía'"),
        ErrorPlantadoBreve("Claro", "Homófonos 'cayó' / 'calló'"),
        ErrorPlantadoBreve("Sutil", "'Sinnúmero' (sustantivo) en una palabra"),
        ErrorPlantadoBreve("Claro", "Tilde de 'público' (sustantivo)"),
    ),
    trampas=(
        TrampaBreve("'Deber de' + infinitivo de probabilidad (correcto)"),
        TrampaBreve("'solo' adverbio sin tilde (norma de 2010)"),
        TrampaBreve("'Pasar desapercibido' (uso válido, equivale a 'inadvertido')"),
        TrampaBreve("'Mientras más' (variante válida de 'cuanto más')"),
        TrampaBreve("'guio' sin tilde (monosílabo ortográfico, norma de 2010)"),
        TrampaBreve("'aun' (= incluso) sin tilde"),
        TrampaBreve("Demostrativo 'aquel' sin tilde (norma de 2010)"),
        TrampaBreve("'video' llano (forma americana válida)"),
        TrampaBreve("Conjugación correcta de 'prever' ('preveía')"),
        TrampaBreve("'ex' separado ante base pluriverbal ('ex secretario general')"),
        TrampaBreve("Gerundio modal válido"),
        TrampaBreve("'con la condición de que' (el 'de' es obligatorio)"),
        TrampaBreve("Concordancia en plural con 'la mayoría de' + sustantivo plural (válida)"),
        TrampaBreve("Demostrativo 'ese' sin tilde (norma de 2010)"),
        TrampaBreve("'solo' adverbio sin tilde (norma de 2010)"),
        TrampaBreve("'adecua' (acentuación válida, igual que 'adecúa')"),
        TrampaBreve("'recién' ante verbo conjugado (americanismo válido)"),
        TrampaBreve("'sino' adversativo correcto"),
        TrampaBreve("'si no' condicional correcto"),
    ),
)

BANCO_TEXTOS_BREVES: Final[dict[str, TextoBreveMarcado]] = {
    TEXTO_BREVE_1.id: TEXTO_BREVE_1,
    TEXTO_BREVE_2.id: TEXTO_BREVE_2,
    TEXTO_BREVE_3.id: TEXTO_BREVE_3,
}

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


def version_parcial(texto: TextoBreveMarcado, corregir: set[int], reemplazos: tuple[tuple[str, str], ...] = ()) -> str:
    """
    Versión del texto con solo algunos errores corregidos (los de `corregir`, por número de orden) y
    reemplazos adicionales sobre el resultado. Sirve para armar respuestas de demostración y tests.
    """
    numero = 0

    def resolver(marca: re.Match[str]) -> str:
        nonlocal numero
        if marca.group(3) is not None:
            return marca.group(3)
        numero += 1
        return marca.group(2).split("|")[0] if numero in corregir else marca.group(1)

    resultado = PATRON_MARCA.sub(resolver, texto.marcado)
    for viejo, nuevo in reemplazos:
        resultado = resultado.replace(viejo, nuevo, 1)
    return resultado


# Respuestas de demostración por texto: (modelo A, modelo B)
DEMOS_CORRECCION: Final[dict[str, tuple[str, str]]] = {
    # A corrige todo salvo el gerundio y cae en la trampa de «sólo»;
    # B omite errores sutiles, corrige mal uno y reformula una frase correcta
    TEXTO_BREVE_1.id: (
        version_parcial(TEXTO_BREVE_1, {1, 2, 3, 4, 5, 6, 8}, (("no solo en papel", "no sólo en papel"),)),
        version_parcial(
            TEXTO_BREVE_1,
            {1, 6},
            (("atravez", "através"), ("en pocos minutos", "en apenas unos minutos")),
        ),
    ),
    # A se le escapan cuatro errores sutiles y cambia «guion» y «provisoria»;
    # B corrige solo errores claros y agrega tildes que la norma actual no lleva
    TEXTO_BREVE_2.id: (
        version_parcial(
            TEXTO_BREVE_2,
            set(range(1, 19)) - {6, 7, 10, 14},
            (("un guion de", "un guión de"), ("digital provisoria", "digital provisional")),
        ),
        version_parcial(
            TEXTO_BREVE_2,
            {2, 4, 8, 12, 15, 17, 18},
            (("ahora solo se", "ahora sólo se"), ("Este fue", "Éste fue"), ("aun así", "aún así")),
        ),
    ),
    # A corrige casi todo, pero se le escapan ocho errores sutiles y cae en cuatro trampas;
    # B corrige solo los errores claros, corrige mal uno sutil y agrega tildes que la norma actual no lleva
    TEXTO_BREVE_3.id: (
        version_parcial(
            TEXTO_BREVE_3,
            set(range(1, 46)) - {5, 8, 9, 20, 27, 29, 36, 39},
            (
                ("pasó desapercibida", "pasó inadvertida"),
                ("profesional guio", "profesional guió"),
                ("un video que", "un vídeo que"),
                ("Aun los técnicos", "Aún los técnicos"),
            ),
        ),
        version_parcial(
            TEXTO_BREVE_3,
            {2, 4, 10, 12, 13, 15, 16, 24, 25, 28, 30, 31, 33, 37, 40, 41, 42, 43, 45},
            (
                ("fue absorbido", "fue absolvido"),
                ("y no solo una", "y no sólo una"),
                ("Aquel fue", "Aquél fue"),
                ("Mientras más avanzaba", "Cuanto más avanzaba"),
            ),
        ),
    ),
}
DEMO_CORRECCION_MODELO_A, DEMO_CORRECCION_MODELO_B = DEMOS_CORRECCION[TEXTO_BREVE_1.id]
