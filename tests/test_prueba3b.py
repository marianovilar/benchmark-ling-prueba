"""
tests/test_prueba3b.py - Tests de la Prueba 3B (corrección de texto breve evaluada por ediciones).
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from correccion_texto import (
    BANCO_TEXTOS_BREVES,
    DEMO_CORRECCION_MODELO_A,
    DEMO_CORRECCION_MODELO_B,
    TEXTO_BREVE_1,
    construir_prueba,
    evaluar_correccion,
    formatear_informe_correccion,
)


@pytest.fixture(scope="module")
def prueba():
    return construir_prueba(TEXTO_BREVE_1)


def _estados(res) -> list[str]:
    return [r.estado for r in res.errores]


def test_marcado_del_banco(prueba) -> None:
    for texto in BANCO_TEXTOS_BREVES.values():
        construir_prueba(texto)  # lanza ValueError si el marcado no coincide con los metadatos
    assert len(prueba.errores) == 8
    assert len(prueba.trampas) == 4
    assert "{" not in prueba.texto_original and "[[" not in prueba.texto_original
    assert "Habían pocos" in prueba.texto_original
    assert "Había pocos" in prueba.texto_referencia


def test_texto_de_referencia_puntua_perfecto(prueba) -> None:
    res = evaluar_correccion(prueba, prueba.texto_referencia)
    assert _estados(res) == ["corregido"] * 8
    assert res.sobrecorrecciones == []
    assert all(t.respetada for t in res.trampas)
    assert res.f05 == 100.0


def test_texto_original_sin_cambios_omite_todo(prueba) -> None:
    res = evaluar_correccion(prueba, prueba.texto_original)
    assert _estados(res) == ["omitido"] * 8
    assert res.sobrecorrecciones == []
    assert res.f05 == 0.0


def test_respuesta_vacia_no_se_evalua(prueba) -> None:
    assert evaluar_correccion(prueba, "   ").evaluado is False


def test_correccion_alternativa_aceptada(prueba) -> None:
    """El gerundio de posterioridad admite «y concluyó» sin coma, además de «, y concluyó»."""
    texto = prueba.texto_referencia.replace("invierno, y concluyó", "invierno y concluyó")
    res = evaluar_correccion(prueba, texto)
    assert res.errores[6].estado == "corregido"
    assert res.sobrecorrecciones == []


def test_correccion_equivocada_cuenta_como_mal_corregida(prueba) -> None:
    texto = prueba.texto_referencia.replace("a través", "através")
    res = evaluar_correccion(prueba, texto)
    assert res.errores[2].estado == "mal_corregido"
    assert res.falsos_positivos == 1
    assert res.falsos_negativos == 1


def test_trampa_y_sobrecorreccion(prueba) -> None:
    """Cambiar «solo» por «sólo» cae en la trampa; reformular «pocos minutos» es sobrecorrección."""
    texto = prueba.texto_referencia.replace("no solo en", "no sólo en").replace(
        "en pocos minutos", "en apenas unos minutos"
    )
    res = evaluar_correccion(prueba, texto)
    assert _estados(res) == ["corregido"] * 8
    assert [t.respetada for t in res.trampas] == [False, True, True, True]
    assert [(e.antes, e.despues) for e in res.sobrecorrecciones] == [("solo", "sólo"), ("pocos", "apenas unos")]
    assert res.precision == pytest.approx(80.0)
    assert res.f05 == pytest.approx(83.33, abs=0.01)


def test_ignora_encabezado_bloque_de_codigo_notas_y_comillas(prueba) -> None:
    texto = prueba.texto_referencia.replace("Había pocos", "Había pocos")
    respuesta = (
        "Aquí tienes el texto corregido:\n\n```\n" + texto + "\n```\n\n---\n**Cambios realizados:**\n- Había\n- a través"
    )
    res = evaluar_correccion(prueba, respuesta)
    assert res.f05 == 100.0
    assert res.aviso_reescritura is False


def test_reescritura_completa_activa_aviso(prueba) -> None:
    res = evaluar_correccion(prueba, "La biblioteca digitalizó sus libros y los estudiantes están contentos.")
    assert res.aviso_reescritura is True


def test_demos(prueba) -> None:
    res_a = evaluar_correccion(prueba, DEMO_CORRECCION_MODELO_A)
    assert res_a.verdaderos_positivos == 7
    assert res_a.errores[6].estado == "omitido"
    assert not res_a.trampas[0].respetada

    res_b = evaluar_correccion(prueba, DEMO_CORRECCION_MODELO_B)
    assert _estados(res_b) == [
        "corregido", "omitido", "mal_corregido", "omitido", "omitido", "corregido", "omitido", "omitido"
    ]
    assert "MAL CORREGIDO" in formatear_informe_correccion(prueba, "B", res_b)


def test_texto_2_referencia_y_original() -> None:
    from correccion_texto import TEXTO_BREVE_2

    prueba2 = construir_prueba(TEXTO_BREVE_2)
    assert len(prueba2.errores) == 18
    assert len(prueba2.trampas) == 10
    assert evaluar_correccion(prueba2, prueba2.texto_referencia).f05 == 100.0
    assert evaluar_correccion(prueba2, prueba2.texto_original).f05 == 0.0


def test_texto_2_alternativas_y_trampas() -> None:
    """«Recuerdo que» y «todavía» son correcciones válidas; «aún así» y «provisional» caen en trampas."""
    from correccion_texto import TEXTO_BREVE_2, version_parcial

    prueba2 = construir_prueba(TEXTO_BREVE_2)
    texto = version_parcial(
        TEXTO_BREVE_2,
        set(range(1, 19)),
        (
            ("Me acuerdo de que", "Recuerdo que"),
            ("y aún no", "y todavía no"),
            ("aun así", "aún así"),
            ("digital provisoria", "digital provisional"),
        ),
    )
    res = evaluar_correccion(prueba2, texto)
    assert all(r.estado == "corregido" for r in res.errores)
    trampas = {t.zona.texto: t.respetada for t in res.trampas}
    assert trampas["aun así"] is False
    assert trampas["provisoria"] is False
    assert len(res.sobrecorrecciones) == 2


def test_demos_de_todos_los_textos_son_evaluables() -> None:
    from correccion_texto import DEMOS_CORRECCION

    for texto_id, demos in DEMOS_CORRECCION.items():
        prueba_x = construir_prueba(BANCO_TEXTOS_BREVES[texto_id])
        for demo in demos:
            res = evaluar_correccion(prueba_x, demo)
            assert res.evaluado and not res.aviso_reescritura
            assert 0 < res.f05 < 100
