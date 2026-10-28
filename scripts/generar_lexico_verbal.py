"""
scripts/generar_lexico_verbal.py - Genera `datos/lexico_verbal_es.json.gz` para `nlp_ligero.py`.

Fuente: tabla de lematización del español de `spacy-lookups-data` (licencia MIT,
https://github.com/explosion/spacy-lookups-data). Se conservan:
- "inf": lemas verbales (infinitivos terminados en -ar, -er, -ir, -ír).
- "fin": formas verbales flexionadas que no son infinitivo, gerundio ni participio
  (candidatas a forma finita).

Uso (solo hace falta volver a ejecutarlo si se quiere regenerar el léxico):
    pip download spacy-lookups-data --no-deps -d /tmp/lk
    python scripts/generar_lexico_verbal.py /tmp/lk/spacy_lookups_data-*.whl
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
import re
import sys
import zipfile

PATRON_LEMA_VERBAL = re.compile(r"(?:ar|er|ir|ír)$")
PATRON_NO_FINITA = re.compile(
    r"(?:ndo|ndose|[aií]d[oa]s?"  # gerundios y participios regulares
    r"|(?:scrit|puest|hech|dich|abiert|cubiert|vuelt|suelt|muert|vist|frit|impres)[oa]s?)$"  # participios irregulares
)
SALIDA = Path(__file__).resolve().parent.parent / "datos" / "lexico_verbal_es.json.gz"


def main(ruta_wheel: str) -> None:
    with zipfile.ZipFile(ruta_wheel) as wheel:
        with wheel.open("spacy_lookups_data/data/es_lemma_lookup.json.gz") as fh:
            tabla: dict[str, str] = json.loads(gzip.decompress(fh.read()))

    # Los infinitivos no llevan tilde salvo en -ír (reír, oír): así se descartan «carácter», «cáncer», «dólar»
    infinitivos = sorted(
        {
            lema
            for lema in tabla.values()
            if PATRON_LEMA_VERBAL.search(lema) and lema.isalpha() and not re.search(r"[áéíóú]", lema[:-2])
        }
    )
    finitas = sorted(
        forma
        for forma, lema in tabla.items()
        if PATRON_LEMA_VERBAL.search(lema)
        and forma != lema
        and forma.isalpha()
        and forma.islower()
        and not PATRON_NO_FINITA.search(forma)
    )

    datos = json.dumps({"inf": infinitivos, "fin": finitas}, ensure_ascii=False, separators=(",", ":"))
    # mtime=0 para que el archivo generado sea reproducible byte a byte
    SALIDA.write_bytes(gzip.compress(datos.encode("utf-8"), compresslevel=9, mtime=0))
    print(f"{SALIDA}: {len(infinitivos)} infinitivos, {len(finitas)} formas flexionadas")


if __name__ == "__main__":
    main(sys.argv[1])
