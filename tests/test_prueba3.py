"""
tests/test_prueba3.py - Tests de aceptación para la Prueba 3 (detección de errores).
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from auditor import evaluate_error_detection
from textos_errores import (
    BANCO_TEXTOS_ERRORES,
    DEMO_RESPUESTA_ERRORES_MODELO_A,
    DEMO_RESPUESTA_ERRORES_MODELO_B,
    ERRORES_TEXTO_1,
    ERRORES_TEXTO_2,
    ERRORES_TEXTO_3,
    TEXTO_1_CONTENIDO,
)


def test_estructura_anclas_y_correcciones_34_errores() -> None:
    """Verifica que los 34 errores de los 3 textos tengan anclas y correcciones sin etiquetas prohibidas."""
    etiquetas_prohibidas = {
        "queísmo",
        "queismo",
        "dequeísmo",
        "dequeismo",
        "quesuismo",
        "gerundio",
        "gerundio de posterioridad",
        "gerundio especificativo",
        "sujeto y verbo",
    }
    todos = list(ERRORES_TEXTO_1) + list(ERRORES_TEXTO_2) + list(ERRORES_TEXTO_3)
    assert len(todos) == 34

    for err in todos:
        assert len(err.anclas) >= 1, f"Error {err.orden} ({err.fenomeno}) sin anclas"
        assert len(err.correcciones) >= 1, f"Error {err.orden} ({err.fenomeno}) sin correcciones"
        for ancla in err.anclas:
            assert ancla.strip().lower() not in etiquetas_prohibidas, (
                f"Etiqueta prohibida '{ancla}' usada como ancla en error {err.orden}"
            )


def test_demo_a_texto_1() -> None:
    """Demo A, texto 1: 11/11, 0 ítems sin correspondencia."""
    res = evaluate_error_detection(DEMO_RESPUESTA_ERRORES_MODELO_A, ERRORES_TEXTO_1)
    assert res.detectados_total == 11
    assert res.total_plantados == 11
    assert res.claros_detectados == 5
    assert res.sutiles_detectados == 6
    assert res.cobertura_porcentaje == 100.0
    assert len(res.items_sin_correspondencia) == 0
    assert res.precision_aprox_porcentaje == 100.0


def test_demo_b_texto_1() -> None:
    """Demo B, texto 1: 6/11 (5 claros, 1 sutil); el error 10 (queísmo) queda omitido."""
    res = evaluate_error_detection(DEMO_RESPUESTA_ERRORES_MODELO_B, ERRORES_TEXTO_1)
    assert res.detectados_total == 6
    assert res.total_plantados == 11
    assert res.claros_detectados == 5
    assert res.sutiles_detectados == 1
    assert len(res.items_sin_correspondencia) == 0

    # El error 5 (dequeísmo) sí se detecta, pero el error 10 (queísmo) debe quedar omitido
    det_por_orden = {d.error_plantado.orden: d for d in res.detalles_plantados}
    assert det_por_orden[5].detectado is True
    assert det_por_orden[10].detectado is False


def test_texto_1_completo_copiado_sin_lista() -> None:
    """Respuesta = texto 1 completo copiado, sin lista: 0/11."""
    res = evaluate_error_detection(TEXTO_1_CONTENIDO, ERRORES_TEXTO_1)
    assert res.detectados_total == 0
    assert res.claros_detectados == 0
    assert res.sutiles_detectados == 0
    assert res.cobertura_porcentaje == 0.0
    assert res.aviso_formato is True


def test_explicacion_sujeto_y_verbo_no_acredita_coma_criminal() -> None:
    """
    Ítem `1. "habían muchos operarios" -> "había muchos operarios" | Falta concordancia entre sujeto y verbo`:
    acredita solo el error 2, no el 3.
    """
    respuesta = (
        '1. "habían muchos operarios" -> "había muchos operarios" | '
        "Falta concordancia entre sujeto y verbo"
    )
    res = evaluate_error_detection(respuesta, ERRORES_TEXTO_1)
    det_por_orden = {d.error_plantado.orden: d for d in res.detalles_plantados}
    assert res.detectados_total == 1
    assert det_por_orden[2].detectado is True
    assert det_por_orden[3].detectado is False


def test_texto_2_llendo_no_acredita_resolución_regulando() -> None:
    """
    Texto 2, ítem `"llendo a ciegas" -> "yendo a ciegas" | Gerundio mal escrito`:
    acredita solo "llendo" (error 6), no el error de "resolución regulando" (error 9).
    """
    respuesta = '1. "llendo a ciegas" -> "yendo a ciegas" | Gerundio mal escrito'
    res = evaluate_error_detection(respuesta, ERRORES_TEXTO_2)
    det_por_orden = {d.error_plantado.orden: d for d in res.detalles_plantados}
    assert res.detectados_total == 1
    assert det_por_orden[6].detectado is True
    assert det_por_orden[9].detectado is False


def test_texto_3_sobre_todo_en_explicacion_no_acredita_sobretodo() -> None:
    """Texto 3: un ítem cuya explicación diga 'sobre todo' no acredita el error 'sobretodo'."""
    respuesta = (
        '1. "intentado hayar evidencias" -> "intentado hallar evidencias" | '
        "Confusión ortográfica que ocurre sobre todo en zonas yeístas."
    )
    res = evaluate_error_detection(respuesta, ERRORES_TEXTO_3)
    det_por_orden = {d.error_plantado.orden: d for d in res.detalles_plantados}
    assert res.detectados_total == 1
    assert det_por_orden[1].detectado is True
    assert det_por_orden[2].detectado is False


def test_item_inventado_cuenta_como_sin_correspondencia() -> None:
    """Ítem inventado `"aposentos privados" -> "aposentos particulares" | Impropiedad`: sin correspondencia."""
    respuesta = (
        '1. "Atravez de una red" -> "A través de una red" | Ortografía.\n'
        '2. "aposentos privados" -> "aposentos particulares" | Impropiedad.'
    )
    res = evaluate_error_detection(respuesta, ERRORES_TEXTO_1)
    assert res.detectados_total == 1
    assert len(res.items_extraidos) == 2
    assert len(res.items_sin_correspondencia) == 1
    assert res.items_sin_correspondencia[0].error_identificado == "aposentos privados"
    assert res.precision_aprox_porcentaje == 50.0


def test_normalizacion_punto_y_espacio_antes_de_coma() -> None:
    """Verifica que el punto se conserve ('indexación.aquellos') y se quite el espacio antes de coma."""
    resp_t2 = '1. "motores de indexación.aquellos fondos" -> "motores de indexación. Aquellos fondos" | Falta espacio.'
    res2 = evaluate_error_detection(resp_t2, ERRORES_TEXTO_2)
    det2 = {d.error_plantado.orden: d for d in res2.detalles_plantados}
    assert det2[8].detectado is True

    resp_t1 = '1. "talleres madrileños , organizaban" -> "talleres madrileños organizaban" | Coma.'
    res1 = evaluate_error_detection(resp_t1, ERRORES_TEXTO_1)
    det1 = {d.error_plantado.orden: d for d in res1.detalles_plantados}
    assert det1[3].detectado is True
