"""
tests/test_prueba3.py - Tests de aceptación para la Prueba 3 (detección de errores).
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

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


def test_correccion_equivocada_detecta_pero_no_corrige() -> None:
    """`"Atravez de una red" -> "Atravéz de una red"`: detectado, mal corregido."""
    res = evaluate_error_detection('1. "Atravez de una red" -> "Atravéz de una red" | Tilde', ERRORES_TEXTO_1)
    det = {d.error_plantado.orden: d for d in res.detalles_plantados}
    assert det[1].detectado is True
    assert det[1].corregido is False
    assert res.corregidos_total == 0


@pytest.mark.parametrize(
    "correccion",
    ["habían muchos operarios", "hubieron muchos operarios", ""],
)
def test_correccion_que_conserva_o_empeora_el_error_no_cuenta(correccion: str) -> None:
    res = evaluate_error_detection(f'1. "habían muchos operarios" -> "{correccion}" | Haber', ERRORES_TEXTO_1)
    det = {d.error_plantado.orden: d for d in res.detalles_plantados}
    assert det[2].detectado is True
    assert det[2].corregido is False


def test_demo_a_todos_bien_corregidos() -> None:
    res = evaluate_error_detection(DEMO_RESPUESTA_ERRORES_MODELO_A, ERRORES_TEXTO_1)
    assert res.corregidos_total == 11
    assert res.correccion_porcentaje == 100.0


def test_correcciones_esperadas_de_la_clave_son_validas() -> None:
    """La corrección esperada de cada uno de los 34 errores debe contar como bien corregida."""
    for texto in BANCO_TEXTOS_ERRORES.values():
        respuesta = "\n".join(
            f'{i}. "{e.fragmento_erroneo}" -> "{e.correccion_esperada}" | x' for i, e in enumerate(texto.errores, 1)
        )
        res = evaluate_error_detection(respuesta, texto.errores)
        assert res.corregidos_total == len(texto.errores), texto.id


@pytest.mark.parametrize(
    ("respuesta", "orden"),
    [
        ('1. "habían" -> "había" | Haber impersonal', 2),
        ('1. "infringir" -> "infligir" | Paronimia', 11),
        ('1. "echo" -> "hecho" | Homófonos', 4),
    ],
)
def test_cita_corta_no_ambigua_acredita_y_corrige(respuesta: str, orden: int) -> None:
    res = evaluate_error_detection(respuesta, ERRORES_TEXTO_1)
    det = {d.error_plantado.orden: d for d in res.detalles_plantados}
    assert res.detectados_total == 1
    assert det[orden].detectado is True
    assert det[orden].corregido is True


def test_cita_corta_sin_la_parte_erronea_no_acredita() -> None:
    """«En cada taller» está en el fragmento del error 2, pero no en sus anclas: no acredita."""
    res = evaluate_error_detection('1. "En cada taller" -> "En cada taller" | ?', ERRORES_TEXTO_1)
    assert res.detectados_total == 0


def test_formato_tabla_markdown() -> None:
    respuesta = (
        "| # | Error | Corrección | Explicación |\n"
        "|---|---|---|---|\n"
        "| 1 | habían muchos operarios | había muchos operarios | Haber impersonal |\n"
        "| 2 | Atravez | A través | Ortografía |"
    )
    res = evaluate_error_detection(respuesta, ERRORES_TEXTO_1)
    assert len(res.items_extraidos) == 2
    assert res.detectados_total == 2
    assert res.corregidos_total == 2
    assert res.precision_aprox_porcentaje == 100.0


def test_formato_en_varias_lineas_con_rotulos() -> None:
    respuesta = (
        "**Error 1**\n"
        'Original: "Atravez de una red"\n'
        'Corrección: "A través de una red"\n'
        "Explicación: ortografía\n"
        "**Error 2**\n"
        'Original: "habían muchos"\n'
        'Corrección: "había muchos"'
    )
    res = evaluate_error_detection(respuesta, ERRORES_TEXTO_1)
    assert len(res.items_extraidos) == 2
    assert res.detectados_total == 2
    assert res.corregidos_total == 2


def test_texto_corregido_completo_no_acredita_errores_que_siguen_ahi() -> None:
    """
    Si el modelo devuelve el texto entero con solo los 6 primeros errores corregidos, se acreditan
    esos 6 y ninguno de los que siguen en el texto (antes, el error 3 se acreditaba igual).
    """
    texto = TEXTO_1_CONTENIDO
    for e in ERRORES_TEXTO_1[:6]:
        texto = texto.replace(e.fragmento_erroneo, e.correccion_esperada)
    res = evaluate_error_detection(texto, ERRORES_TEXTO_1)
    det = {d.error_plantado.orden: d for d in res.detalles_plantados}
    assert res.texto_reescrito is True
    assert [o for o, d in det.items() if d.detectado] == [1, 2, 3, 4, 5, 6]
    assert res.corregidos_total == 6
