"""
tests/test_prueba2.py - Tests de aceptación para la Prueba 2 (dos puntos con verbo conjugado en forma finita).
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from auditor import (
    SPACY_MODEL_NAME,
    TEXTO_DEMO_MODELO_A,
    TEXTO_DEMO_MODELO_B,
    detect_colon_clauses_in_doc,
    load_spacy_model,
)


@pytest.fixture(scope="module")
def nlp_es():
    nlp, _ = load_spacy_model(SPACY_MODEL_NAME)
    return nlp


def test_infinitivo_tras_dos_puntos_no_cuenta(nlp_es) -> None:
    """
    'El plan de la cátedra propone una meta clara: mejorar la calidad de los textos académicos.'
    -> 0 coincidencias.
    """
    texto = "El plan de la cátedra propone una meta clara: mejorar la calidad de los textos académicos."
    doc = nlp_es(texto)
    matches = detect_colon_clauses_in_doc(doc, nlp_es)
    assert len(matches) == 0


def test_verbo_conjugado_tras_dos_puntos_si_cuenta(nlp_es) -> None:
    """
    'La glosa marginal operaba bajo un principio claro: el texto clásico contenía una verdad latente.'
    -> 1 coincidencia.
    """
    texto = "La glosa marginal operaba bajo un principio claro: el texto clásico contenía una verdad latente."
    doc = nlp_es(texto)
    matches = detect_colon_clauses_in_doc(doc, nlp_es)
    assert len(matches) == 1
    assert matches[0].verbos_detectados_der == ["contenía"]


def test_demo_a_y_demo_b_dos_puntos(nlp_es) -> None:
    """
    Demo A: sigue dando 2 coincidencias; en la primera, los verbos listados son 'contenía, debía'
    (sin 'ser' ni 'actualizada'). Demo B: 0.
    """
    doc_a = nlp_es(TEXTO_DEMO_MODELO_A)
    matches_a = detect_colon_clauses_in_doc(doc_a, nlp_es)
    assert len(matches_a) == 2
    assert matches_a[0].verbos_detectados_der == ["contenía", "debía"]
    assert "ser" not in matches_a[0].verbos_detectados_der
    assert "actualizada" not in matches_a[0].verbos_detectados_der
    assert matches_a[1].verbos_detectados_der == ["conlleva"]

    doc_b = nlp_es(TEXTO_DEMO_MODELO_B)
    matches_b = detect_colon_clauses_in_doc(doc_b, nlp_es)
    assert len(matches_b) == 0
