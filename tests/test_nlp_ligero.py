"""
tests/test_nlp_ligero.py - Tests del analizador heurístico que reemplaza a spaCy en la versión web.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import nlp_ligero
from auditor import (
    TEXTO_DEMO_MODELO_A,
    TEXTO_DEMO_MODELO_B,
    count_words_in_doc,
    detect_colon_clauses_in_doc,
    detect_reframing_in_doc,
)


@pytest.fixture(scope="module")
def nlp():
    return nlp_ligero.load()


def test_infinitivo_tras_dos_puntos_no_cuenta(nlp) -> None:
    texto = "El plan de la cátedra propone una meta clara: mejorar la calidad de los textos académicos."
    assert detect_colon_clauses_in_doc(nlp(texto), nlp) == []


def test_verbo_conjugado_tras_dos_puntos_si_cuenta(nlp) -> None:
    texto = "La glosa marginal operaba bajo un principio claro: el texto clásico contenía una verdad latente."
    matches = detect_colon_clauses_in_doc(nlp(texto), nlp)
    assert [m.verbos_detectados_der for m in matches] == [["contenía"]]


def test_homografos_nominales_tras_determinante_no_cuentan(nlp) -> None:
    """«el precio», «el uso» no son verbos aunque el léxico registre precio (preciar) y uso (usar)."""
    texto = "El informe final tiene tres ejes: el precio, la calidad del producto y el uso responsable."
    assert detect_colon_clauses_in_doc(nlp(texto), nlp) == []


def test_demo_a_y_demo_b(nlp) -> None:
    matches_a = detect_colon_clauses_in_doc(nlp(TEXTO_DEMO_MODELO_A), nlp)
    assert [m.verbos_detectados_der for m in matches_a] == [["contenía", "debía"], ["conlleva"]]
    assert detect_colon_clauses_in_doc(nlp(TEXTO_DEMO_MODELO_B), nlp) == []


def test_segmentacion_y_conteo(nlp) -> None:
    doc = nlp(TEXTO_DEMO_MODELO_A)
    oraciones = [s.text for s in doc.sents]
    assert len(oraciones) == 9
    assert all(o == o.strip() for o in oraciones)
    assert count_words_in_doc(doc) == 148
    assert len(detect_reframing_in_doc(doc)) == 2
