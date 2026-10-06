import math
import unittest

from careport.ct_update_precision import precision_scenarios


class PrecisionPlanningTests(unittest.TestCase):
    def test_seed_average_does_not_triple_patient_count(self):
        result = precision_scenarios([
            {'a': 1, 'b': 2, 'c': 3},
            {'a': 3, 'b': 4, 'c': 5},
            {'a': 5, 'b': 6, 'c': 7},
        ], [30, 45], [1, 2, 5])
        self.assertEqual(result['patients'], 3)
        self.assertEqual(result['seeds'], 3)
        self.assertAlmostEqual(result['patient_sd_after_seed_average_ml'], 1)
        self.assertAlmostEqual(result['planning_sd_ml'], 1)

    def test_uncorrelated_seed_errors_do_not_select_optimistic_dispersion(self):
        result = precision_scenarios([{'a': -1, 'b': 1}, {'a': 1, 'b': -1}], [45], [2])
        self.assertEqual(result['patient_sd_after_seed_average_ml'], 0)
        self.assertAlmostEqual(result['planning_sd_ml'], math.sqrt(2))
        rows = result['halfwidth_scenarios']
        marginal = next(r for r in rows if r['family_size'] == 1 and r['sd_multiplier'] == 1)
        family = next(r for r in rows if r['family_size'] == 6 and r['sd_multiplier'] == 1)
        self.assertGreater(family['halfwidth_ml'], marginal['halfwidth_ml'])

    def test_invalid_or_unpaired_inputs_fail(self):
        for seeds in [[], [{'a': 1}], [{'a': 1, 'b': 2}, {'a': 1, 'c': 2}],
                      [{'a': 1, 'b': float('nan')}]]:
            with self.assertRaises(ValueError):
                precision_scenarios(seeds, [45], [2])

    def test_required_size_scales_quadratically_with_resolution(self):
        result = precision_scenarios([{'a': 0, 'b': 10, 'c': 20}], [45], [1, 2])
        rows = [r for r in result['required_count_scenarios']
                if r['family_size'] == 1 and r['sd_multiplier'] == 1]
        first, second = rows
        self.assertLessEqual(abs(first['required_patients'] - 4*second['required_patients']), 3)
