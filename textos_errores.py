"""
textos_errores.py - Banco de 3 textos extensos (~950-1050 palabras) con errores plantados
(claros y sutiles) y generador del prompt estructurado para el Benchmark de Detección de Errores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class PlantedError:
    """Representa un error plantado en orden de aparición dentro de un texto de prueba."""

    orden: int
    categoria: str  # "Claro" | "Sutil"
    fenomeno: str
    fragmento_erroneo: str
    correccion_esperada: str
    claves_busqueda: tuple[str, ...]  # Subcadenas para verificar si el modelo identificó el error


@dataclass(frozen=True)
class ErrorBenchmarkText:
    """Texto extenso (~1000 palabras) con su catálogo ordenado de errores plantados."""

    id: str
    titulo: str
    descripcion: str
    texto: str
    errores: tuple[PlantedError, ...]


# ============================================================================
# TEXTO 1: IMPRENTA Y CULTURA ESCRITA EN EL SIGLO DE ORO (~980 PALABRAS)
# ============================================================================

TEXTO_1_CONTENIDO: Final[str] = """La difusión de la imprenta de tipos móviles en la península ibérica durante los siglos XVI y XVII transformó de manera irreversible los modos de producir, poner en circulación y leer los textos literarios y doctrinales. Atravez de una red cada vez más densa de talleres asentados en ciudades universitarias y comerciales como Salamanca, Alcalá de Henares, Sevilla, Madrid y Barcelona, el libro impreso dejó de ser un objeto suntuario reservado a las élites eclesiásticas para convertirse en un artefacto cotidiano que circulaba en plazas, colegios y aposentos privados. Sin embargo, este proceso de expansión material no supuso la desaparición inmediata del manuscrito ni la uniformidad absoluta de los impresos, sino la convivencia compleja de prácticas caligráficas, orales y tipográficas que moldearon la cultura del Siglo de Oro.

Para comprender la materialidad de aquellos volúmenes, resulta indispensable asomarse al funcionamiento interno de una oficina tipográfica de la época. En cada taller habían muchos operarios especializados que dividían las tareas según una jerarquía estricta: el regente coordinaba el trabajo diario, los cajistas o componedores seleccionaban a gran velocidad los caracteres de plomo invertidos desde los cajetines, los tiradores accionaban el husillo de la prensa de madera y los batidores entintaban las formas mediante balas de cuero rellenas de lana. Los componedores tipográficos de los talleres madrileños, organizaban las cajas siguiendo esquemas que facilitaban la localización rápida de las letras más frecuentes del castellano, aunque el desgaste constante de las fundiciones y la escasez de papel obligaban a menudo a reutilizar tipos defectuosos o a abreviar palabras al final de la línea para ajustar la justificación del renglón.

Esa intervención constante de múltiples manos explica el echo de que ninguna edición antigua sea un reflejo transparente del manuscrito autógrafo entregado por el autor. Antes siquiera de llegar a las manos del cajista, el original de imprenta debía atravesar una serie de filtros institucionales y técnicos que alteraban su fisonomía lingüística y ortográfica. Muchos dramaturgos y poetas pensaban de que la puntuación y la ortografía del libro impreso eran responsabilidad exclusiva de los oficiales del taller, por lo que entregaban cuadernos con escasa demarcación gráfica y confiaban en que el componedor introdujera las comas, los puntos y las mayúsculas según las costumbres editoriales vigentes en cada ciudad. De este modo, la sintaxis impresa respondía tanto a las decisiones rítmicas del escritor como a las necesidades mecánicas del reparto de pliegos y cuadernillos.

La legislación real ejercía, además, una vigilancia permanente sobre todo lo que salía de las prensas. Desde la pragmática sancionada en 1558 bajo el reinado de Felipe II, ningún libro podía imprimirse sin obtener previamente la licencia del Consejo Real y sin someter cada pliego impreso al cotejo meticuloso del corrector oficial, quien verificaba que el volumen coincidiera con el ejemplar rubricado. Sino se conservaban los pliegos originales debidamente firmados por el escribano de cámara, el impresor se exponía a penas severísimas que iban desde la confiscación de los moldes y la pérdida de la tirada entera hasta el destierro o la inhabilitación perpetua. Pese a tales controles, el mercado clandestino floreció en numerosas provincias: se falsificaban pies de imprenta, se fechaban en el extranjero ediciones compuestas en prensas locales y se perseguían a los impresores clandestinos mediante visitas de inspección que rara vez lograban contener el tráfico de pliegos sueltos, comedias y romances populares.

En el terreno propiamente literario, las consecuencias de esta economía editorial fueron decisivas para la configuración de los géneros modernos. La prosa narrativa, el teatro comercial y la lírica impresa se adaptaron a los formatos materiales que demandaba un público urbano ávido de novedades. Un caso paradigmático lo ofrece la historia editorial de la obra cervantina: el taller madrileño de Juan de la Cuesta trabajó a ritmo acelerado durante el otoño de 1604, publicándose la edición príncipe del Quijote en enero de 1605, convirtiéndose décadas más tarde en el modelo canónico de la novela europea moderna. Aquella primera tirada, plagada de erratas de composición, cambios de epígrafes e inconsistencias que el propio autor comentaría con ironía en la segunda parte de 1615, demuestra hasta qué punto el texto áureo era una realidad abierta e inestable, sujeta a las contingencias del trabajo manual en la imprenta.

Junto a los libros de gran formato destinados a las bibliotecas eruditas, circulaba una producción exhuberante de impresos efímeros que alimentaba la lectura colectiva en voz alta. Relaciones de sucesos, cartillas escolares, almanaques, pronósticos astrológicos y pliegos de cordel se vendían por pocos maravedíes en las gradas de las iglesias y en las ferias comarcales. Los libreros financiaban estas tiradas económicas con la condición que el taller entregara los pliegos en pocos días, lo que reducía al mínimo las tareas de corrección de pruebas y multiplicaba las variantes entre los distintos ejemplares de una misma emisión. A menudo, cuando el corrector advertía un descuido grave a mitad del tiraje, detenía momentáneamente la prensa para enmendar la forma de plomo, pero los pliegos ya estampados no se desperdiciaban jamás debido al elevado costo del papel de trapo, dando lugar a ejemplares híbridos donde coexisten estados corregidos y sin corregir.

La autoridad eclesiástica intervenía asimismo mediante los índices de libros prohibidos y expurgados, cuyo propósito no era únicamente vedar la circulación de obras heréticas, sino también infringir un castigo ejemplar a quienes difundieran doctrinas contrarias a la ortodoxia tridentina. Los calificadores del Santo Oficio tachaban pasajes enteros en los volúmenes ya encuadernados o exigían suprimir hojas específicas antes de autorizar la venta pública. Así, entre la censura estatal, el celo inquisitorial, las urgencias comerciales de los mercaderes de libros y las rutinas mecánicas de los cajistas, el texto del Siglo de Oro llegó hasta nosotros como un palimpsesto colectivo cuya lectura crítica exige hoy conjugar la sensibilidad literaria con el rigor de la bibliografía material y la historia de la lengua."""

ERRORES_TEXTO_1: Final[tuple[PlantedError, ...]] = (
    PlantedError(
        orden=1,
        categoria="Claro",
        fenomeno="Ortografía léxica y locución preposicional",
        fragmento_erroneo="Atravez de una red",
        correccion_esperada="A través de una red",
        claves_busqueda=("atravez", "a través"),
    ),
    PlantedError(
        orden=2,
        categoria="Claro",
        fenomeno="Pluralización indebida de 'haber' impersonal",
        fragmento_erroneo="En cada taller habían muchos operarios",
        correccion_esperada="En cada taller había muchos operarios",
        claves_busqueda=("habían muchos", "había muchos"),
    ),
    PlantedError(
        orden=3,
        categoria="Sutil",
        fenomeno="Coma entre sujeto y verbo (coma criminal)",
        fragmento_erroneo="Los componedores tipográficos de los talleres madrileños, organizaban",
        correccion_esperada="Los componedores tipográficos de los talleres madrileños organizaban",
        claves_busqueda=("talleres madrileños, organizaban", "madrileños, organizaban", "sujeto y verbo"),
    ),
    PlantedError(
        orden=4,
        categoria="Claro",
        fenomeno="Confusión de homófonos ('echo' / 'hecho')",
        fragmento_erroneo="el echo de que ninguna edición",
        correccion_esperada="el hecho de que ninguna edición",
        claves_busqueda=("el echo", "el hecho"),
    ),
    PlantedError(
        orden=5,
        categoria="Sutil",
        fenomeno="Dequeísmo",
        fragmento_erroneo="pensaban de que la puntuación",
        correccion_esperada="pensaban que la puntuación",
        claves_busqueda=("pensaban de que", "pensaban que", "dequeísmo", "dequeismo"),
    ),
    PlantedError(
        orden=6,
        categoria="Claro",
        fenomeno="Confusión 'sino' adversativo por 'si no' condicional",
        fragmento_erroneo="Sino se conservaban los pliegos",
        correccion_esperada="Si no se conservaban los pliegos",
        claves_busqueda=("sino se conservaban", "si no se conservaban"),
    ),
    PlantedError(
        orden=7,
        categoria="Sutil",
        fenomeno="Impersonal refleja con CD de persona pluralizado",
        fragmento_erroneo="se perseguían a los impresores clandestinos",
        correccion_esperada="se perseguía a los impresores clandestinos",
        claves_busqueda=("se perseguían a los", "se perseguía a los", "perseguían a los impresores"),
    ),
    PlantedError(
        orden=8,
        categoria="Sutil",
        fenomeno="Gerundio de posterioridad temporal",
        fragmento_erroneo="publicándose la edición príncipe del Quijote en enero de 1605, convirtiéndose décadas más tarde",
        correccion_esperada="y se convirtió décadas más tarde (o: que se convertiría décadas más tarde)",
        claves_busqueda=("convirtiéndose décadas", "gerundio de posterioridad", "publicándose la edición"),
    ),
    PlantedError(
        orden=9,
        categoria="Claro",
        fenomeno="Ortografía ('exhuberante' con h intercalada indebida)",
        fragmento_erroneo="producción exhuberante",
        correccion_esperada="producción exuberante",
        claves_busqueda=("exhuberante", "exuberante"),
    ),
    PlantedError(
        orden=10,
        categoria="Sutil",
        fenomeno="Queísmo en locución conjuntiva ('con la condición de que')",
        fragmento_erroneo="con la condición que el taller entregara",
        correccion_esperada="con la condición de que el taller entregara",
        claves_busqueda=("condición que el", "condición de que", "queísmo", "queismo"),
    ),
    PlantedError(
        orden=11,
        categoria="Sutil",
        fenomeno="Impropiedad léxica / paronimia ('infringir' por 'infligir')",
        fragmento_erroneo="infringir un castigo ejemplar",
        correccion_esperada="infligir un castigo ejemplar",
        claves_busqueda=("infringir un castigo", "infligir un castigo", "infringir", "infligir"),
    ),
)


# ============================================================================
# TEXTO 2: ARCHIVOS DIGITALES, OBSOLESCENCIA Y MEMORIA HISTÓRICA (~970 PALABRAS)
# ============================================================================

TEXTO_2_CONTENIDO: Final[str] = """La transición acelerada desde los repositorios en papel hacia las infraestructuras de almacenamiento en la nube ha modificado de raíz las condiciones de posibilidad de la investigación histórica y filológica contemporánea. Durante milenios, la supervivencia de los documentos dependió de la durabilidad física de soportes como el pergamino, el papiro o el papel de trapo, cuya degradación química era lenta y predecible. Hoy, en cambio, nos enfrentamos a una paradoja inédita: producimos un volumen diario de registros escritos, audiovisuales y administrativos infinitamente superior al de cualquier otra época de la humanidad, pero la vida útil de los formatos electrónicos en que se inscriben apenas alcanza unas pocas décadas sin una intervención técnica constante.

Para que el patrimonio documental del siglo XXI no se diluya en la ilegibilidad informática, debe a ver un protocolo público de preservación digital que garantice la migración periódica de los archivos antes de que el software y el hardware con que fueron creados queden fuera de circulación. La historia reciente de la computación ofrece numerosos ejemplos de esta fragilidad estructural. En muchas reparticiones estatales y bibliotecas universitarias habían cientos de cintas magnéticas, disquetes de cinco pulgadas y discos ópticos grabados durante las décadas de 1980 y 1990 cuyos datos resultan hoy inaccesibles porque ya no existen las lectoras físicas ni los sistemas operativos capaces de decodificar su estructura binaria original.

El problema no se reduce únicamente a la degradación física de los dispositivos de silicio o de los recubrimientos magnéticos. Los responsables técnicos de los grandes centros de documentación nacional, advierten con frecuencia que el mayor peligro radica en el cierre propietario de los estándares de codificación. Cuando una institución cultural confía la totalidad de su catálogo digitalizado a un formato cerrado o dependiente de una única corporación tecnológica, hipoteca su autonomía futura. Por ello, los archiveros insisten en que los repositorios académicos deben absorver únicamente estándares abiertos y bien documentados, tales como texto plano codificado en UTF-8, esquemas XML-TEI para ediciones críticas o formatos de imagen sin compresión con pérdida.

Pese a estas recomendaciones metodológicas, resulta evidente de que las políticas de digitalización masiva suelen priorizar la velocidad de captura fotográfica por encima de la descripción catalográfica rigurosa y de la curaduría de metadatos. En numerosos proyectos financiados con fondos de emergencia, se escanean miles de legajos coloniales o colecciones hemerográficas del siglo XIX sin incorporar capas de reconocimiento óptico de caracteres revisadas por especialistas ni índices topográficos consistentes. De este modo, el investigador que consulta el repositorio remoto se encuentra ante millones de imágenes estáticas desprovistas de contexto archivístico, llendo a ciegas entre carpetas numeradas de manera arbitraria que dificultan reconstruir la procedencia orgánica de cada expediente.

A esta precariedad descriptiva se suma una transformación silenciosa en los hábitos heurísticos de las nuevas generaciones de humanistas. El investigador formado exclusivamente en entornos de búsqueda por palabras clave corre el riesgo de no ser plenamente conciente de todo aquello que permanece fuera del radar de los motores de indexación.aquellos fondos documentales que aún aguardan en estanterías provinciales, archivos parroquiales o colecciones sindicales sin digitalizar tienden a volverse invisibles para la historiografía reciente, generando un sesgo de disponibilidad que sobrerrepresenta las fuentes accesibles en línea en detrimento de los testimonios periféricos o de difícil acceso material.

Las administraciones públicas han intentado responder a este desafío mediante marcos regulatorios específicos. Hace pocos años, el ministerio del área promulgó una resolución regulando la guarda perpetua de los expedientes administrativos electrónicos y estableciendo plazos obligatorios para su transferencia al Archivo General. Sin embargo, la normativa omitió preveer las partidas presupuestarias indispensables para contratar personal especializado en preservación digital y para auditar la integridad criptográfica de los paquetes de datos a largo plazo. Sin recursos humanos estables ni servidores redundantes distribuidos geográficamente, la legislación corre el peligro de quedar reducida a una declaración de buenas intenciones incapaz de frenar la pérdida paulatina de la memoria institucional.

Otro aspecto crítico concierne a la gobernanza de los datos culturales cuando intervienen consorcios privados en la digitalización del patrimonio público. En diversas ocasiones, convenios firmados sin el debido asesoramiento jurídico han permitido que empresas externas pasen a detentar legítimamente durante diez o quince años los derechos exclusivos de explotación comercial sobre las reproducciones digitales de manuscritos y periódicos que pertenecen al dominio público. Semejante restricción contradice la misión democratizadora de las bibliotecas públicas y levanta barreras económicas que perjudican de manera directa a los investigadores de universidades con menor capacidad financiera.

Frente a este panorama complejo, la formación de los futuros filólogos, historiadores y bibliotecólogos exige integrar los saberes tradicionales de la diplomática, la paleografía y la ecdótica con las herramientas contemporáneas de la informática humanística. No basta con celebrar el acceso instantáneo a millones de páginas desde la pantalla doméstica; es preciso interrogar críticamente la arquitectura técnica de los archivos digitales, auditar sus criterios de selección y asegurar que las decisiones tecnológicas del presente no comprometan el derecho de las generaciones venideras a interrogar su propio pasado documental."""

ERRORES_TEXTO_2: Final[tuple[PlantedError, ...]] = (
    PlantedError(
        orden=1,
        categoria="Claro",
        fenomeno="Confusión 'a ver' (preposición + verbo ver) por 'haber' (infinitivo)",
        fragmento_erroneo="debe a ver un protocolo público",
        correccion_esperada="debe haber un protocolo público",
        claves_busqueda=("debe a ver", "debe haber", "a ver un protocolo"),
    ),
    PlantedError(
        orden=2,
        categoria="Claro",
        fenomeno="Pluralización indebida de 'haber' impersonal",
        fragmento_erroneo="habían cientos de cintas magnéticas",
        correccion_esperada="había cientos de cintas magnéticas",
        claves_busqueda=("habían cientos", "había cientos"),
    ),
    PlantedError(
        orden=3,
        categoria="Sutil",
        fenomeno="Coma entre sujeto y verbo (coma criminal)",
        fragmento_erroneo="Los responsables técnicos de los grandes centros de documentación nacional, advierten",
        correccion_esperada="Los responsables técnicos de los grandes centros de documentación nacional advierten",
        claves_busqueda=("documentación nacional, advierten", "nacional, advierten", "sujeto y verbo"),
    ),
    PlantedError(
        orden=4,
        categoria="Claro",
        fenomeno="Ortografía verbal ('absorver' con v por 'absorber' con b)",
        fragmento_erroneo="deben absorver únicamente estándares",
        correccion_esperada="deben absorber únicamente estándares",
        claves_busqueda=("absorver", "absorber"),
    ),
    PlantedError(
        orden=5,
        categoria="Sutil",
        fenomeno="Dequeísmo en oración subordinada sustantiva de sujeto",
        fragmento_erroneo="resulta evidente de que las políticas",
        correccion_esperada="resulta evidente que las políticas",
        claves_busqueda=("evidente de que", "evidente que", "dequeísmo", "dequeismo"),
    ),
    PlantedError(
        orden=6,
        categoria="Claro",
        fenomeno="Ortografía de gerundio irregular ('llendo' por 'yendo')",
        fragmento_erroneo="llendo a ciegas entre carpetas",
        correccion_esperada="yendo a ciegas entre carpetas",
        claves_busqueda=("llendo", "yendo"),
    ),
    PlantedError(
        orden=7,
        categoria="Claro",
        fenomeno="Ortografía ('conciente' por 'consciente')",
        fragmento_erroneo="plenamente conciente de todo aquello",
        correccion_esperada="plenamente consciente de todo aquello",
        claves_busqueda=("conciente", "consciente"),
    ),
    PlantedError(
        orden=8,
        categoria="Claro",
        fenomeno="Falta de espacio tras punto y minúscula inicial de oración",
        fragmento_erroneo="indexación.aquellos fondos documentales",
        correccion_esperada="indexación. Aquellos fondos documentales",
        claves_busqueda=("indexación.aquellos", ".aquellos", "aquellos fondos"),
    ),
    PlantedError(
        orden=9,
        categoria="Sutil",
        fenomeno="Gerundio especificativo / del Boletín Oficial ('resolución regulando')",
        fragmento_erroneo="promulgó una resolución regulando la guarda perpetua y estableciendo",
        correccion_esperada="promulgó una resolución que regula la guarda perpetua y establece",
        claves_busqueda=("resolución regulando", "gerundio especificativo", "gerundio"),
    ),
    PlantedError(
        orden=10,
        categoria="Claro",
        fenomeno="Ultracorrección morfológica ('preveer' por 'prever')",
        fragmento_erroneo="la normativa omitió preveer las partidas",
        correccion_esperada="la normativa omitió prever las partidas",
        claves_busqueda=("preveer", "prever"),
    ),
    PlantedError(
        orden=11,
        categoria="Sutil",
        fenomeno="Impropiedad léxica ('detentar legítimamente' — detentar implica posesión ilegítima)",
        fragmento_erroneo="pasen a detentar legítimamente durante diez o quince años los derechos",
        correccion_esperada="pasen a ostentar / ejercer / poseer legítimamente los derechos",
        claves_busqueda=("detentar legítimamente", "detentar", "ostentar"),
    ),
)


# ============================================================================
# TEXTO 3: LECTURA PROFUNDA, PANTALLAS Y ATENCIÓN COGNITIVA (~970 PALABRAS)
# ============================================================================

TEXTO_3_CONTENIDO: Final[str] = """El desplazamiento progresivo de la lectura en papel hacia las pantallas conectadas ha suscitado durante la última década un intenso debate entre neurocientíficos, psicólogos cognitivos, lingüistas y docentes de todos los niveles educativos. A diferencia de la adquisición del lenguaje oral, que constituye una capacidad biológicamente programada en la especie humana desde edades tempranas, el cerebro lector es una conquista cultural que exige reorganizar circuitos neuronales originalmente destinados al reconocimiento visual de objetos y a la articulación fonológica. Precisamente porque ese circuito de lectura profunda no es innato sino plástico, las características físicas y ergonómicas del soporte en que leemos influyen de manera directa sobre la calidad de nuestra comprensión textual.

Diversas investigaciones empíricas realizadas en universidades europeas y latinoamericanas han intentado hayar evidencias precisas acerca de cómo varía el procesamiento semántico cuando un mismo texto expositivo o argumentativo se aborda en páginas impresas frente a dispositivos electrónicos retroiluminados. Los resultados acumulados muestran que la lectura en pantalla tiende a favorecer estrategias de barrido superficial, caza de palabras clave y fragmentación de la atención. Este fenómeno se acentúa sobretodo cuando el lector interactúa con entornos multitarea donde las notificaciones emergentes, los hipervínculos y el desplazamiento continuo del cursor compiten por los recursos limitados de la memoria de trabajo.

Los equipos de investigación en psicolingüística experimental de varias universidades públicas, han observado además una marcada discrepancia entre el rendimiento real de los sujetos y su autopercepción subjetiva. Muchos jóvenes universitarios consideran que leen con mayor rapidez y eficacia en sus teléfonos o portátiles, pero al someterse a pruebas de inferencia causal, reconstrucción de argumentos complejos o evocación del orden expositivo obtienen puntajes significativamente inferiores a quienes leyeron el mismo ensayo en papel. Para comprender porque se produce esta ilusión de competencia lectora, es necesario distinguir entre la mera decodificación léxica de oraciones aisladas y la construcción pausada de un modelo mental integrado del texto completo.

En el soporte impreso, la materialidad tridimensional del volumen ofrece un mapa espacial y propioceptivo que auxilia de manera silenciosa a la memoria episódica: recordamos involuntariamente que una definición decisiva se encontraba en el ángulo superior izquierdo de una página par o hacia el último cuarto del libro. En la pantalla digital, en cambio, esa topografía tangible desaparece bajo una superficie única y deslizante, lo que incrementa la carga cognitiva necesaria para orientarse dentro de arquitecturas argumentativas extensas. La escases de puntos de anclaje físico dificulta volver atrás para releer un pasaje ambiguo o para cotejar una premisa inicial con sus conclusiones finales.

Las consecuencias pedagógicas de esta mutación cultural se perciben con nitidez en las aulas de enseñanza media y superior. No cabe duda que la exposición constante a flujos informativos breves, veloces y altamente estimulantes altera los umbrales de tolerancia frente a la complejidad sintáctica y la densidad conceptual. Un lector habituado al ritmo entrecortado de las redes sociales experimenta a menudo fatiga o impaciencia ante períodos subordinados extensos, vocabulario abstracto o desarrollos filosóficos que demoran varios párrafos en formular su tesis central. Dicha impaciencia conlleva a una pérdida gradual de la lectura crítica y reflexiva, aquella en la que el individuo suspende momentáneamente la prisa informativa para dialogar en silencio con las razones del autor.

En un estudio longitudinal muy citado en la bibliografía reciente, los especialistas seleccionaron una cohorte de trescientos ingresantes universitarios en marzo, evaluando en diciembre del mismo año su evolución en tareas de síntesis argumentativa y detección de contradicciones lógicas. Durante ese período, se entrevistaron a los estudiantes de distintas carreras humanísticas y científicas para registrar sus hábitos cotidianos de estudio, constatándose que aquellos que combinaban apuntes manuscritos con lecturas en papel retenían mejor las relaciones jerárquicas entre ideas principales y secundarias. Por el contrario, un estudiante que su práctica de estudio se limitaba a subrayar mecánicamente archivos digitales en la pantalla solía reproducir fragmentos literales sin lograr articular una reformulación conceptual propia.

Hacen ya más de quince años que especialistas como Maryanne Wolf advirtieron sobre el riesgo de atrofiar la dimensión contemplativa e inferencial del cerebro lector si las instituciones educativas abandonaban sin reflexión crítica el soporte impreso. Sin embargo, el diagnóstico riguroso del problema no debe conducir a una nostalgia tecnofóbica ni al rechazo simplista de las herramientas digitales, cuyas ventajas para la búsqueda bibliográfica, la accesibilidad de personas con discapacidad visual y la circulación democrática del conocimiento resultan incuestionables. El verdadero desafío pedagógico consiste en formar lectores llamados «bialfabetizados», capaces de aprovechar la agilidad instrumental de los medios electrónicos sin renunciar a la lentitud deliberada que reclama el pensamiento complejo.

A menudo se afirma que el discurso público contemporáneo adolece de serenidad analítica y de escucha atenta ante los matices del argumento ajeno. Recuperar espacios curriculares protegidos para la lectura lineal, silenciosa y sostenida de textos extensos constituye, por tanto, mucho más que una destreza escolar: representa una condición indispensable para preservar la autonomía intelectual de los ciudadanos frente a la economía de la atención fragmentada."""

ERRORES_TEXTO_3: Final[tuple[PlantedError, ...]] = (
    PlantedError(
        orden=1,
        categoria="Claro",
        fenomeno="Confusión ortográfica ('hayar' con y por 'hallar' con ll)",
        fragmento_erroneo="intentado hayar evidencias precisas",
        correccion_esperada="intentado hallar evidencias precisas",
        claves_busqueda=("hayar", "hallar"),
    ),
    PlantedError(
        orden=2,
        categoria="Claro",
        fenomeno="Confusión 'sobretodo' (sustantivo: abrigo) por 'sobre todo' (locución adverbial)",
        fragmento_erroneo="se acentúa sobretodo cuando el lector",
        correccion_esperada="se acentúa sobre todo cuando el lector",
        claves_busqueda=("sobretodo", "sobre todo"),
    ),
    PlantedError(
        orden=3,
        categoria="Sutil",
        fenomeno="Coma entre sujeto y verbo (coma criminal)",
        fragmento_erroneo="Los equipos de investigación en psicolingüística experimental de varias universidades públicas, han observado",
        correccion_esperada="Los equipos de investigación en psicolingüística experimental de varias universidades públicas han observado",
        claves_busqueda=("universidades públicas, han observado", "públicas, han observado", "sujeto y verbo"),
    ),
    PlantedError(
        orden=4,
        categoria="Claro",
        fenomeno="Confusión 'porque' conjunción causal por 'por qué' interrogativo indirecto",
        fragmento_erroneo="Para comprender porque se produce esta ilusión",
        correccion_esperada="Para comprender por qué se produce esta ilusión",
        claves_busqueda=("comprender porque", "comprender por qué", "por qué se produce"),
    ),
    PlantedError(
        orden=5,
        categoria="Claro",
        fenomeno="Ortografía ('escases' con s por 'escasez' con z)",
        fragmento_erroneo="La escases de puntos de anclaje",
        correccion_esperada="La escasez de puntos de anclaje",
        claves_busqueda=("escases", "escasez"),
    ),
    PlantedError(
        orden=6,
        categoria="Sutil",
        fenomeno="Queísmo en locución ('no cabe duda de que')",
        fragmento_erroneo="No cabe duda que la exposición constante",
        correccion_esperada="No cabe duda de que la exposición constante",
        claves_busqueda=("no cabe duda que", "no cabe duda de que", "queísmo", "queismo"),
    ),
    PlantedError(
        orden=7,
        categoria="Claro",
        fenomeno="Cruce sintáctico de régimen ('conlleva a' por 'conlleva' transitivo o 'lleva a')",
        fragmento_erroneo="Dicha impaciencia conlleva a una pérdida gradual",
        correccion_esperada="Dicha impaciencia conlleva una pérdida gradual (o: lleva a una pérdida)",
        claves_busqueda=("conlleva a", "conlleva una"),
    ),
    PlantedError(
        orden=8,
        categoria="Sutil",
        fenomeno="Gerundio de posterioridad temporal",
        fragmento_erroneo="seleccionaron una cohorte de trescientos ingresantes universitarios en marzo, evaluando en diciembre",
        correccion_esperada="seleccionaron una cohorte en marzo y evaluaron en diciembre",
        claves_busqueda=("evaluando en diciembre", "gerundio de posterioridad"),
    ),
    PlantedError(
        orden=9,
        categoria="Sutil",
        fenomeno="Discordancia en impersonal refleja con CD humano ('se entrevistaron a los estudiantes')",
        fragmento_erroneo="se entrevistaron a los estudiantes de distintas carreras",
        correccion_esperada="se entrevistó a los estudiantes (o pasiva refleja: fueron entrevistados los estudiantes)",
        claves_busqueda=("se entrevistaron a los", "se entrevistó a los", "entrevistaron a los estudiantes"),
    ),
    PlantedError(
        orden=10,
        categoria="Sutil",
        fenomeno="Quesuismo ('que su' en lugar del relativo posesivo 'cuya')",
        fragmento_erroneo="un estudiante que su práctica de estudio se limitaba",
        correccion_esperada="un estudiante cuya práctica de estudio se limitaba",
        claves_busqueda=("que su práctica", "cuya práctica", "quesuismo"),
    ),
    PlantedError(
        orden=11,
        categoria="Claro",
        fenomeno="Pluralización indebida de 'hacer' impersonal temporal ('hacen ya más de quince años')",
        fragmento_erroneo="Hacen ya más de quince años",
        correccion_esperada="Hace ya más de quince años",
        claves_busqueda=("hacen ya más", "hace ya más", "hacen ya"),
    ),
    PlantedError(
        orden=12,
        categoria="Sutil",
        fenomeno="Impropiedad léxica ('adolece de serenidad' usado como 'carece de', cuando adolecer significa padecer un defecto)",
        fragmento_erroneo="adolece de serenidad analítica y de escucha atenta",
        correccion_esperada="carece de serenidad analítica (o: adolece de falta de serenidad)",
        claves_busqueda=("adolece de serenidad", "carece de serenidad", "adolecer"),
    ),
)


BANCO_TEXTOS_ERRORES: Final[dict[str, ErrorBenchmarkText]] = {
    "texto_1": ErrorBenchmarkText(
        id="texto_1",
        titulo="Texto 1: La imprenta y la cultura escrita en el Siglo de Oro (~980 palabras, 11 errores)",
        descripcion="Contiene 5 errores claros (ortografía, 'haber' impersonal, homófonos) y 6 sutiles (coma sujeto-verbo, dequeísmo, queísmo, gerundio de posterioridad, impersonal con 'se', paronimia).",
        texto=TEXTO_1_CONTENIDO,
        errores=ERRORES_TEXTO_1,
    ),
    "texto_2": ErrorBenchmarkText(
        id="texto_2",
        titulo="Texto 2: Archivos digitales, obsolescencia y memoria histórica (~970 palabras, 11 errores)",
        descripcion="Contiene 7 errores claros ('a ver'/'haber', 'absorver', 'llendo', 'conciente', 'preveer', puntuación) y 4 sutiles (coma sujeto-verbo, dequeísmo, gerundio del Boletín Oficial, impropiedad de 'detentar').",
        texto=TEXTO_2_CONTENIDO,
        errores=ERRORES_TEXTO_2,
    ),
    "texto_3": ErrorBenchmarkText(
        id="texto_3",
        titulo="Texto 3: Lectura profunda, pantallas y atención cognitiva (~970 palabras, 12 errores)",
        descripcion="Contiene 6 errores claros ('hayar', 'sobretodo', 'porque'/'por qué', 'escases', 'conlleva a', 'hacen años') y 6 sutiles (coma sujeto-verbo, queísmo, gerundio de posterioridad, 'se entrevistaron a', quesuismo, 'adolecer de').",
        texto=TEXTO_3_CONTENIDO,
        errores=ERRORES_TEXTO_3,
    ),
}

DEMO_RESPUESTA_ERRORES_MODELO_A: Final[str] = """1. "Atravez de una red" -> "A través de una red" | Error ortográfico: la locución preposicional se escribe en dos palabras, con tilde y terminada en 's'.
2. "En cada taller habían muchos operarios" -> "En cada taller había muchos operarios" | Error de concordancia: el verbo 'haber' impersonal se conjuga siempre en tercera persona del singular.
3. "Los componedores tipográficos de los talleres madrileños, organizaban" -> "Los componedores tipográficos de los talleres madrileños organizaban" | Error de puntuación (coma entre sujeto y verbo): no debe separarse el sujeto de su verbo principal con una coma.
4. "el echo de que ninguna edición" -> "el hecho de que ninguna edición" | Error ortográfico por homofonía: sustantivo derivado del verbo 'hacer' (se escribe con 'h').
5. "pensaban de que la puntuación" -> "pensaban que la puntuación" | Error sintáctico (dequeísmo): el verbo 'pensar' selecciona un complemento directo sin preposición 'de'.
6. "Sino se conservaban los pliegos" -> "Si no se conservaban los pliegos" | Error morfosintáctico: corresponde la conjunción condicional 'si' seguida del adverbio de negación 'no'.
7. "se perseguían a los impresores clandestinos" -> "se perseguía a los impresores clandestinos" | Error en construcción impersonal refleja: al llevar complemento directo de persona con 'a', el verbo debe ir en singular.
8. "publicándose la edición príncipe del Quijote en enero de 1605, convirtiéndose décadas más tarde" -> "y se convirtió décadas más tarde" | Uso incorrecto del gerundio de posterioridad temporal.
9. "producción exhuberante" -> "producción exuberante" | Error ortográfico: 'exuberante' se escribe sin 'h' intercalada.
10. "con la condición que el taller entregara" -> "con la condición de que el taller entregara" | Error sintáctico (queísmo): omisión indebida de la preposición 'de' exigida por el sustantivo 'condición'.
11. "infringir un castigo ejemplar" -> "infligir un castigo ejemplar" | Impropiedad léxica por paronimia: 'infligir' significa imponer un castigo, mientras que 'infringir' significa quebrantar una ley."""

DEMO_RESPUESTA_ERRORES_MODELO_B: Final[str] = """1. "Atravez de una red" -> "A través de una red" | Error ortográfico en la locución 'a través de'.
2. "habían muchos operarios" -> "había muchos operarios" | El verbo 'haber' existencial no lleva plural.
3. "el echo de que" -> "el hecho de que" | Confusión entre 'echo' (echar) y 'hecho' (hacer).
4. "pensaban de que" -> "pensaban que" | Dequeísmo.
5. "Sino se conservaban" -> "Si no se conservaban" | Debe escribirse separado por ser condicional negativo.
6. "producción exhuberante" -> "producción exuberante" | 'Exuberante' no lleva 'h' intermedia."""


def build_error_detection_prompt(texto_a_evaluar: str) -> str:
    """
    Construye el prompt estructurado listo para copiar y pegar en los LLMs,
    solicitando una lista numerada en estricto orden de aparición con el error y su corrección.
    """
    return (
        "Actúa como un corrector de estilo y editor filológico experto en normativa panhispánica (RAE / ASALE).\n"
        "Analiza minuciosamente el siguiente texto y detecta TODOS los errores de ortografía, acentuación, "
        "puntuación (como comas entre sujeto y verbo), morfosintaxis (concordancia, verbos impersonales, "
        "queísmo, dequeísmo, quesuismo, gerundios mal empleados, régimen preposicional) e impropiedades léxicas.\n\n"
        "INSTRUCCIÓN DE FORMATO OBLIGATORIA:\n"
        "Devuelve ÚNICAMENTE una lista numerada siguiendo el estricto ORDEN DE APARICIÓN en el texto, "
        "respetando este esquema exacto en cada línea:\n"
        '1. "[Fragmento exacto con el error]" -> "[Fragmento corregido]" | [Explicación breve del error]\n'
        '2. "[Fragmento exacto con el error]" -> "[Fragmento corregido]" | [Explicación breve del error]\n\n'
        "TEXTO A AUDITAR:\n"
        "---\n"
        f"{texto_a_evaluar.strip()}\n"
        "---"
    )
