from __future__ import annotations

import re

from app_models import SimulationBreakdownItem, SimulationRequest, SimulationResponse


ATECO_GROUPS = [
    {"ranges": [(10, 11)], "coeff": 0.40},
    {"ranges": [(45, 45), (46, 46), (47, 47)], "coeff": 0.40},
    {"ranges": [(41, 43), (68, 68)], "coeff": 0.86},
    {"ranges": [(55, 56)], "coeff": 0.40},
    {"ranges": [(64, 66), (69, 75), (85, 85), (86, 88)], "coeff": 0.78},
    {
        "ranges": [
            (1, 3),
            (5, 9),
            (12, 33),
            (35, 39),
            (49, 53),
            (58, 63),
            (77, 82),
            (84, 84),
            (90, 99),
        ],
        "coeff": 0.67,
    },
]

DEFAULT_CONTRIBUTION_RATES = {
    "nessuna": 0.0,
    "gestione_separata": 0.2607,
    "artigiani_commercianti": 0.24,
}


def _parse_ateco_prefix(ateco_code: str | None) -> tuple[int, int | None] | None:
    if not ateco_code:
        return None
    match = re.search(r"([0-9]{2})(?:[.\s-]?([0-9]{1,2}))?", ateco_code)
    if not match:
        return None
    prefix = int(match.group(1))
    subcode = int(match.group(2)) if match.group(2) else None
    return prefix, subcode


def lookup_coefficiente(ateco_code: str | None) -> float | None:
    parsed = _parse_ateco_prefix(ateco_code)
    if parsed is None:
        return None
    prefix, subcode = parsed
    if prefix == 46:
        if subcode == 1 or (subcode is not None and 10 <= subcode <= 19):
            return 0.62
        return 0.40
    if prefix == 47:
        if subcode == 81:
            return 0.40
        if subcode is not None and 82 <= subcode <= 89:
            return 0.54
        return 0.40
    for group in ATECO_GROUPS:
        for start, end in group["ranges"]:
            if start <= prefix <= end:
                return group["coeff"]
    return None


def simulate_forfettario(payload: SimulationRequest) -> SimulationResponse:
    coeff = payload.coefficiente_redditivita
    if coeff is None:
        coeff = lookup_coefficiente(payload.ateco_code)
    if coeff is None:
        raise ValueError(
            "Serve un coefficiente di redditivita o un codice ATECO riconoscibile."
        )
    if coeff > 1:
        coeff = coeff / 100

    contribution_rate = payload.aliquota_contributiva
    if contribution_rate is None:
        contribution_rate = DEFAULT_CONTRIBUTION_RATES.get(
            payload.gestione_previdenziale,
            0.0,
        )
    if contribution_rate > 1:
        contribution_rate = contribution_rate / 100

    if payload.riduzione_inps_35 and payload.gestione_previdenziale == "artigiani_commercianti":
        contribution_rate *= 0.65

    imponibile = round(payload.ricavi * coeff, 2)
    contributi = round(imponibile * contribution_rate, 2)
    base_imposta = max(imponibile - contributi, 0)
    imposta = round(base_imposta * payload.aliquota_imposta, 2)
    netto = round(payload.ricavi - contributi - imposta, 2)

    notes = [
        "Simulazione orientativa: non sostituisce il calcolo fiscale o previdenziale definitivo.",
        "Per artigiani e commercianti il calcolo e' semplificato e non include minimi o conguagli.",
    ]
    if payload.riduzione_inps_35 and payload.gestione_previdenziale != "artigiani_commercianti":
        notes.append("La riduzione INPS del 35% e' stata ignorata perche' non applicabile alla gestione selezionata.")

    return SimulationResponse(
        regime_id=payload.regime_id,
        coefficiente_redditivita=round(coeff, 4),
        aliquota_imposta=payload.aliquota_imposta,
        aliquota_contributiva=round(contribution_rate, 4),
        imponibile_stimato=imponibile,
        contributi_stimati=contributi,
        imposta_sostitutiva_stimata=imposta,
        netto_stimato=netto,
        breakdown=[
            SimulationBreakdownItem(label="Ricavi", amount=round(payload.ricavi, 2)),
            SimulationBreakdownItem(label="Imponibile stimato", amount=imponibile),
            SimulationBreakdownItem(label="Contributi stimati", amount=contributi),
            SimulationBreakdownItem(
                label="Imposta sostitutiva stimata",
                amount=imposta,
            ),
            SimulationBreakdownItem(label="Netto stimato", amount=netto),
        ],
        notes=notes,
    )
