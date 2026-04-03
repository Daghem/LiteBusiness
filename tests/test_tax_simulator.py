import unittest

from app_models import SimulationRequest
from tax_simulator import lookup_coefficiente, simulate_forfettario


class TaxSimulatorTests(unittest.TestCase):
    def test_lookup_coefficiente_from_ateco(self):
        self.assertEqual(lookup_coefficiente("62.01"), 0.67)
        self.assertEqual(lookup_coefficiente("46.17"), 0.62)
        self.assertEqual(lookup_coefficiente("47.85"), 0.54)

    def test_simulate_forfettario_with_manual_coeff(self):
        result = simulate_forfettario(
            SimulationRequest(
                ricavi=50000,
                coefficiente_redditivita=78,
                aliquota_imposta=0.15,
                gestione_previdenziale="nessuna",
            )
        )
        self.assertAlmostEqual(result.imponibile_stimato, 39000.0)
        self.assertAlmostEqual(result.imposta_sostitutiva_stimata, 5850.0)
        self.assertAlmostEqual(result.netto_stimato, 44150.0)

    def test_simulate_forfettario_with_artigiani_reduction(self):
        result = simulate_forfettario(
            SimulationRequest(
                ricavi=40000,
                ateco_code="47.81",
                aliquota_imposta=0.05,
                gestione_previdenziale="artigiani_commercianti",
                riduzione_inps_35=True,
            )
        )
        self.assertGreater(result.contributi_stimati, 0)
        self.assertLess(result.aliquota_contributiva, 0.24)


if __name__ == "__main__":
    unittest.main()
