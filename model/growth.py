"""Readiness classes, scenarios and the logistic drilling schedule (Section 2.3 of the paper)."""
import math

import numpy as np

# ── Readiness classes ────────────────────────────────────────────────────────
# Each country is placed in a class by fixed rules applied to observed injection
# (London Register of Subsurface CO2 Storage), the project pipeline (IEA CCUS
# Projects Database) and the CCS Storage Indicator. The rules and the evidence
# for every country are in Supplementary Data 1 of the paper.
#
#   1  Injecting    injection at scale and/or operational projects
#   2  Pipeline     committed or announced project
#   3  Appraised    no projects, but resource assessed and Storage Readiness defined
#   4  Conceptual   none of the above, but geological data available
#
# Keys are the shapefile names in inputs/, so multi-word names use underscores.
COUNTRY_TIER = {
    # 1 - Injecting (9 countries)
    'Australia': 1, 'Brazil': 1, 'Canada': 1, 'China': 1, 'Croatia': 1,
    'Hungary': 1, 'Norway': 1, 'Saudi_Arabia': 1, 'USA': 1,
    # 2 - Pipeline (17 countries)
    'France': 2, 'Greece': 2, 'India': 2, 'Indonesia': 2, 'Italy': 2,
    'Japan': 2, 'Libya': 2, 'Malaysia': 2, 'New_Zealand': 2, 'Oman': 2,
    'Papua_New_Guinea': 2, 'Poland': 2, 'Romania': 2, 'Russia': 2,
    'Spain': 2, 'Thailand': 2, 'UK': 2,
    # 3 - Appraised (19 countries)
    'Algeria': 3, 'Angola': 3, 'Austria': 3, 'Egypt': 3, 'Germany': 3,
    'Iran': 3, 'Israel': 3, 'Kazakhstan': 3, 'Latvia': 3, 'Mexico': 3,
    'Morocco': 3, 'Myanmar': 3, 'Nigeria': 3, 'Pakistan': 3,
    'Philippines': 3, 'South_Africa': 3, 'Tunisia': 3, 'Turkey': 3,
    'Vietnam': 3,
    # 4 - Conceptual (26 countries)
    'Argentina': 4, 'Azerbaijan': 4, 'Belize': 4, 'Bolivia': 4,
    'Cameroon': 4, 'Chad': 4, 'Colombia': 4, 'Cote_dIvoire': 4, 'Cuba': 4,
    'Falkland_Islands': 4, 'Gabon': 4, 'Ghana': 4, 'Iraq': 4, 'Peru': 4,
    'Senegal': 4, 'Sudan': 4, 'Syria': 4, 'Tajikistan': 4, 'Tanzania': 4,
    'Trinidad_and_Tobago': 4, 'Turkmenistan': 4, 'Uganda': 4, 'Ukraine': 4,
    'Uzbekistan': 4, 'Venezuela': 4, 'Yemen': 4,
}

# A country missing from the table takes a fifth, lowest class. None of the 71
# modelled countries falls in it.
DEFAULT_TIER = 5

# Start delay (years after 2023) and logistic growth rate k (1/yr) by class (Table 2)
TIER_DELAY_YEARS = {1: 0, 2: 10, 3: 25, 4: 50, 5: 75}
TIER_GROWTH_RATE = {1: 0.15, 2: 0.10, 3: 0.075, 4: 0.05, 5: 0.025}


def country_tier(country):
    """Readiness class of a country (1-4), or DEFAULT_TIER if it is not listed."""
    return COUNTRY_TIER.get(country, DEFAULT_TIER)


# ── Scenarios (Table 3) ──────────────────────────────────────────────────────
# A scenario name is '<deployment>' or '<deployment>_<scale-up>', for example
# 'reference' (Reference (Central)) or 'maximum_fast' (Maximum (Fast)).
#   deployment  which classes deploy at all
#   scale-up    multipliers on the class growth rate and start delay
SCENARIO_TIERS = {
    'limited':   (1, 2),
    'reference': (1, 2, 3),
    'maximum':   (1, 2, 3, 4, 5),
}
RATE_K_SCALE = {'slow': 0.50, 'central': 1.00, 'fast': 1.50}
RATE_DELAY_SCALE = {'slow': 1.50, 'central': 1.00, 'fast': 0.50}

ALL_SCENARIOS = ('reference', 'limited', 'maximum',
                 'reference_fast', 'limited_fast', 'maximum_fast',
                 'reference_slow', 'limited_slow', 'maximum_slow')


def split_scenario(scenario):
    """'limited_slow' -> ('limited', 'slow'); 'limited' -> ('limited', 'central')."""
    coverage, _, rate = scenario.partition('_')
    if coverage not in SCENARIO_TIERS:
        raise KeyError(f"unknown scenario {scenario!r}; deployment must be one of {sorted(SCENARIO_TIERS)}")
    if (rate or 'central') not in RATE_K_SCALE:
        raise KeyError(f"unknown scenario {scenario!r}; scale-up must be one of {sorted(RATE_K_SCALE)}")
    return coverage, rate or 'central'


def scenario_countries(countries, scenario):
    """The subset of `countries` that deploy under `scenario`."""
    tiers = SCENARIO_TIERS[split_scenario(scenario)[0]]
    return [c for c in countries if country_tier(c) in tiers]


def scenario_growth_rate(country, scenario):
    """Logistic growth rate k (1/yr) for a country under a scenario."""
    rate = split_scenario(scenario)[1]
    return TIER_GROWTH_RATE[country_tier(country)] * RATE_K_SCALE[rate]


def injection_delay_years(country, scenario):
    """Years after 2023 before a country's first well (whole years)."""
    rate = split_scenario(scenario)[1]
    return int(RATE_DELAY_SCALE[rate] * TIER_DELAY_YEARS[country_tier(country)])


# ── Logistic curve ───────────────────────────────────────────────────────────

def logistic_capacity_and_midpoint(S_gdf, k_growth, seed_mt, inj_years):
    """Carrying capacity L (wells) and midpoint t0 (model years) for a country.

    L is the number of wells needed to use the country's pressure-limited
    capacity over `inj_years` per well. t0 is set so that the cumulative wells
    in the first model year match `seed_mt`, the country's 2023 injection from
    the London Register, converted to wells with the area-weighted mean
    injection rate. Countries not in the register start from a single well.

    Returns (L, t0, mean_rate).
    """
    S_gdf['L_aquifer'] = (S_gdf['MCO2'] * 1000 / S_gdf['inj_rate'] / inj_years).clip(lower=1)
    L = max(1, int(S_gdf['L_aquifer'].sum()))

    mean_rate = (S_gdf['inj_rate'] * S_gdf['well_fraction']).sum()  # MtCO2/yr per well
    y1 = max(1.0, seed_mt / mean_rate)
    y1 = min(y1, 0.5 * L)  # keep the anchor below the midpoint
    t0 = solve_logistic_t0(L, k_growth, 1, y1)

    return L, t0, mean_rate


# Drilling continues until the logistic curve reaches 99% of L, for at least
# 100 model years (Section 2.1.1), so large or slow-growing countries are not
# cut off part-way up their curve.
LOGISTIC_COMPLETE = 0.99
MIN_MODEL_YEARS = 100


def drilling_window(t0, k):
    """Model years of drilling needed to reach LOGISTIC_COMPLETE of L."""
    to_complete = math.log(LOGISTIC_COMPLETE / (1 - LOGISTIC_COMPLETE)) / k
    return max(MIN_MODEL_YEARS, int(math.ceil(t0 + to_complete)))


def logistic_function(L, t, k, t0):
    return L / (1 + np.exp(-k * (t - t0)))


def solve_logistic_t0(L, k, t, y):
    """Midpoint t0 at which the logistic curve passes through (t, y)."""
    return t + math.log((L / y) - 1) / k


def generate_well_schedule_logistic(years, L, growth_rate, peak_year, injection_duration,
                                    candidate_locations, used_index):
    """New wells for each model year, keeping the cumulative count on the logistic curve.

    Wells take the next unused locations from `candidate_locations` (already in
    random order). Returns the wells and the number drilled each year.
    """
    all_wells = []
    num_wells_drilled_annually = []
    drilled = 0

    for year in range(1, years + 1):
        cum_target = logistic_function(L, year, growth_rate, peak_year)
        num_to_drill = max(0, int(round(cum_target)) - drilled)
        drilled += num_to_drill
        num_wells_drilled_annually.append(num_to_drill)

        for loc in pooled_well_generator(num_to_drill, candidate_locations, used_index):
            all_wells.append({'x': loc[0], 'y': loc[1], 'start_year': year,
                              'end_year': year + injection_duration - 1})

    return all_wells, num_wells_drilled_annually


def pooled_well_generator(n, candidate_locations, used_index):
    """Take the next `n` unused locations from the candidate pool."""
    wells = candidate_locations[used_index[0]: used_index[0] + n]
    if len(wells) < n:
        raise ValueError("Not enough unused well locations in the candidate pool.")
    used_index[0] += n
    return wells
