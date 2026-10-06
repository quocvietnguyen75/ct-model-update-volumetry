"""Development-only precision scenarios; no external outcome or power claims."""

from __future__ import annotations

import math
from statistics import NormalDist, mean, stdev
from typing import Mapping, Sequence


def precision_scenarios(seeds: Sequence[Mapping[str, float]], sample_sizes: Sequence[int],
                        halfwidth_targets: Sequence[float]) -> dict:
    if not seeds or len(seeds[0]) < 2:
        raise ValueError('At least two paired patients are required')
    patients = sorted(seeds[0])
    if any(set(seed) != set(patients) or any(type(v) not in (int, float) or not math.isfinite(v)
                                           for v in seed.values()) for seed in seeds):
        raise ValueError('Unpaired patients or invalid contrasts')
    if (not sample_sizes or any(type(n) is not int or n < 2 for n in sample_sizes)
            or not halfwidth_targets or any(type(h) not in (int, float) or not math.isfinite(h)
                                             or h <= 0 for h in halfwidth_targets)):
        raise ValueError('Invalid precision grid')
    averaged = [mean(seed[patient] for seed in seeds) for patient in patients]
    individual_sd = [stdev(seed[patient] for patient in patients) for seed in seeds]
    averaged_sd = stdev(averaged)
    planning_sd = max(averaged_sd, *individual_sd)
    widths, counts = [], []
    for family in (1, 6):
        z = NormalDist().inv_cdf(1 - 0.05/(2*family))
        for multiplier in (1.0, 1.5, 2.0):
            settings = {'family_size': family, 'sd_multiplier': multiplier}
            for n in sample_sizes:
                widths.append({**settings, 'patients': n,
                               'halfwidth_ml': z * planning_sd * multiplier / math.sqrt(n)})
            for h in halfwidth_targets:
                counts.append({**settings, 'target_halfwidth_ml': h,
                               'required_patients': max(2, math.ceil((z*planning_sd*multiplier/h)**2))})
    return {'patients': len(patients), 'seeds': len(seeds),
            'patient_sd_after_seed_average_ml': averaged_sd,
            'individual_seed_patient_sd_ml': individual_sd, 'planning_sd_ml': planning_sd,
            'halfwidth_scenarios': widths, 'required_count_scenarios': counts,
            'method': 'normal_approximation_max_observed_patient_SD_with_inflation_grid',
            'external_variance_assumed_known': False, 'power_analysis': False,
            'clinical_preservation_margin': None}
