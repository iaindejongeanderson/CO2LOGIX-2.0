import numpy as np
import math


# ── Readiness tiers ──────────────────────────────────────────────────────────
# Each country is graded on how far it has actually got towards injecting, from
# three sources: observed injection (London Register), the project pipeline
# (IEA CCUS Projects Database) and the CCS Storage Indicator. The tiers are
# derived in Readiness/buildInjectionDelays.py; rerun that script to regenerate
# this table when any source is updated.
#
#   1  injecting    active geological storage - reported injection volumes, an
#                   operating permitted project
#   2  pipeline     projects in development, announced or permitted but not yet
#                   injecting at scale - storage licence, injection permit
#                   application, committed project, FEED complete, government
#                   cluster or hub programme
#   3  appraised    storage appraisal conducted but no projects - national
#                   storage atlas, published basin assessment, CCS strategy,
#                   identified hubs. Assessed at all by the CSI counts here
#   4  conceptual   a storage resource understood only at a high level: in the
#                   basin dataset, but no national appraisal and no projects
#   5  no evidence  not even that. Empty for the current country set, since the
#                   model only runs countries that are in the basin dataset
#
# Tier 3 turns on CSI *membership*, not CSI score. A score threshold put a
# cliff between two countries either side of an arbitrary cut; the score
# separates degree, not kind, so it is supporting evidence rather than the
# discriminator.
#
# The table is hard-coded rather than read from the workbook so that a model
# run never depends on the Readiness folder being present or current.
COUNTRY_TIER = {
    # tier 1 - injecting (9 countries)
    'Australia': 1,
    'Brazil': 1,
    'Canada': 1,
    'China': 1,
    'Croatia': 1,
    'Hungary': 1,
    'Norway': 1,
    'Saudi_Arabia': 1,
    'USA': 1,
    # tier 2 - pipeline (17 countries)
    'France': 2,
    'Greece': 2,
    'India': 2,
    'Indonesia': 2,
    'Italy': 2,
    'Japan': 2,
    'Libya': 2,
    'Malaysia': 2,
    'New_Zealand': 2,
    'Oman': 2,
    'Papua_New_Guinea': 2,
    'Poland': 2,
    'Romania': 2,
    'Russia': 2,
    'Spain': 2,
    'Thailand': 2,
    'UK': 2,
    # tier 3 - appraised (19 countries)
    'Algeria': 3,
    'Angola': 3,
    'Austria': 3,
    'Egypt': 3,
    'Germany': 3,
    'Iran': 3,
    'Israel': 3,
    'Kazakhstan': 3,
    'Latvia': 3,
    'Mexico': 3,
    'Morocco': 3,
    'Myanmar': 3,
    'Nigeria': 3,
    'Pakistan': 3,
    'Philippines': 3,
    'South_Africa': 3,
    'Tunisia': 3,
    'Turkey': 3,
    'Vietnam': 3,
    # tier 4 - conceptual (26 countries)
    'Argentina': 4,
    'Azerbaijan': 4,
    'Belize': 4,
    'Bolivia': 4,
    'Cameroon': 4,
    'Chad': 4,
    'Colombia': 4,
    'Cote_dIvoire': 4,
    'Cuba': 4,
    'Falkland_Islands': 4,
    'Gabon': 4,
    'Ghana': 4,
    'Iraq': 4,
    'Peru': 4,
    'Senegal': 4,
    'Sudan': 4,
    'Syria': 4,
    'Tajikistan': 4,
    'Tanzania': 4,
    'Trinidad_and_Tobago': 4,
    'Turkmenistan': 4,
    'Uganda': 4,
    'Ukraine': 4,
    'Uzbekistan': 4,
    'Venezuela': 4,
    'Yemen': 4,
}

# A country absent from the table is one the evidence has nothing to say about,
# so it takes the bottom tier.
DEFAULT_TIER = 5

# Country keys come from the shapefile stems, so anything multi-word carries an
# underscore. The supplementary workbooks are read by people rather than by the
# model, so they get the spelling the paper uses.
COUNTRY_DISPLAY_NAMES = {
    'Cote_dIvoire': "Cote d'Ivoire",
}


def country_display_name(country):
    """Printed form of a country key: 'Saudi_Arabia' -> 'Saudi Arabia'."""
    return COUNTRY_DISPLAY_NAMES.get(country, country.replace('_', ' '))

# Published names for the tiers, as Table 2 of the paper writes them. Tier 5
# is empty for the current country set, so nothing downstream should offer it
# to a reader as a category they could look for.
CLASS_NAMES = {1: 'Injecting', 2: 'Pipeline', 3: 'Appraised', 4: 'Conceptual',
               5: 'No evidence'}

# Years after START_YEAR (2023) before the curve begins. Tiers 1 and 2 match
# the median start date of their members' own projects; 3 to 5 describe the
# absence of a project, so their spacing is a judgement, not a measurement.
TIER_DELAY_YEARS = {1: 0, 2: 10, 3: 25, 4: 50, 5: 75}

# Logistic growth rate k (1/yr) by tier: a country that is already injecting
# scales fastest, one with nothing behind it barely moves.
TIER_GROWTH_RATE = {1: 0.15, 2: 0.10, 3: 0.075, 4: 0.05, 5: 0.025}


def country_tier(country):
    """Readiness tier for a country, defaulting to the bottom tier."""
    return COUNTRY_TIER.get(country, DEFAULT_TIER)


# ── Scenarios ────────────────────────────────────────────────────────────────

SCENARIO_TIERS = {
    'limited':   (1, 2),
    'reference': (1, 2, 3),
    'maximum':   (1, 2, 3, 4, 5),
}

RATE_K_SCALE = {
    'slow':    0.50,
    'central': 1.00,
    'fast':    1.50,
}

RATE_DELAY_SCALE = {
    'slow':    1.50,
    'central': 1.00,
    'fast':    0.50,
}

# The scenario sets runCO2LOGIX writes, and the published names everything
# downstream labels them with. Defined here so the model and every reporting
# script work from one list.
SCENARIOS_THREE = ('limited', 'reference', 'maximum')

SCENARIOS_FULL = ('limited_slow', 'limited', 'limited_fast',
                  'reference_slow', 'reference', 'reference_fast',
                  'maximum_slow', 'maximum', 'maximum_fast')

# Five scenarios ordered from most to least constrained, for the plots that
# show a spread. Strictly monotonic: each step either widens the tier coverage
# or raises the rate, never trading one against the other - which is what the
# ambiguous pairs (limited_fast against reference_slow) would do.
SCENARIO_LADDER = ('limited_slow', 'limited', 'reference', 'maximum',
                   'maximum_fast')

SCENARIO_LABELS = {
    'limited_slow':   'Limited (slow)',
    'limited':        'Limited',
    'limited_fast':   'Limited (fast)',
    'reference_slow': 'Reference (slow)',
    'reference':      'Reference',
    'reference_fast': 'Reference (fast)',
    'maximum_slow':   'Maximum (slow)',
    'maximum':        'Maximum',
    'maximum_fast':   'Maximum (fast)',
}


# The same nine scenarios written the way the paper writes them, with both
# halves named. SCENARIO_LABELS leaves "(Central)" off because a console table
# reads better without it, but the supplementary workbooks sit next to the
# paper, so their sheet names have to match its wording exactly.
SCENARIO_LABELS_FULL = {
    'limited_slow':   'Limited (Slow)',
    'limited':        'Limited (Central)',
    'limited_fast':   'Limited (Fast)',
    'reference_slow': 'Reference (Slow)',
    'reference':      'Reference (Central)',
    'reference_fast': 'Reference (Fast)',
    'maximum_slow':   'Maximum (Slow)',
    'maximum':        'Maximum (Central)',
    'maximum_fast':   'Maximum (Fast)',
}


def split_scenario(scenario):
    """'limited_slow' -> ('limited', 'slow'); 'limited' -> ('limited', 'central')."""
    coverage, _, rate = scenario.partition('_')
    if coverage not in SCENARIO_TIERS:
        raise KeyError(f"unknown scenario coverage {coverage!r} in {scenario!r}; "
                       f"expected one of {sorted(SCENARIO_TIERS)}")
    return coverage, rate or 'central'


def scenario_countries(countries, scenario):
    """The subset of `countries` whose tier is modelled under `scenario`."""
    tiers = SCENARIO_TIERS[split_scenario(scenario)[0]]
    return [c for c in countries if country_tier(c) in tiers]


def in_scenario(country, scenario):
    """Whether a single country is modelled under `scenario`."""
    return country_tier(country) in SCENARIO_TIERS[split_scenario(scenario)[0]]


def scenario_growth_rate(country, scenario):
    """Logistic growth rate k for a country under a given scenario."""
    rate = split_scenario(scenario)[1]
    return TIER_GROWTH_RATE[country_tier(country)] * RATE_K_SCALE[rate]


def injection_delay_years(country, scenario):
    """Years before a country's curve starts, from its readiness tier.

    The tier delays are already measured from START_YEAR, so nothing is
    subtracted here. Countries in tier 1 are injecting today and start
    immediately, whether or not the London Register gives them a seed volume.
    """
    rate = split_scenario(scenario)[1]
    return int(RATE_DELAY_SCALE[rate] * TIER_DELAY_YEARS[country_tier(country)])


def logistic_capacity_and_midpoint(S_gdf, k_growth, seed_mt, inj_years):
    """Carrying capacity L (wells) and midpoint t0 (model years) for a country.

    L is the number of wells needed to produce the country's pressure-limited
    capacity over `inj_years`. t0 anchors the curve so that cumulative wells
    in the first model year reproduce `seed_mt`, the country's observed
    injection from the London Register, converted to well equivalents using
    the area-weighted mean injection rate (wells are placed area-weighted, so
    that is the expected rate of an early well). Countries not in the
    register start from a single well.

    The per-basin well counts are attached as `L_aquifer` so they carry
    through to the exported basin shapefile.

    Returns (L, t0, mean_rate).
    """
    S_gdf['L_aquifer'] = (S_gdf['MCO2'] * 1000 / S_gdf['inj_rate'] / inj_years).clip(lower=1)
    L = max(1, int(S_gdf['L_aquifer'].sum()))

    mean_rate = (S_gdf['inj_rate'] * S_gdf['well_fraction']).sum()  # Mt/yr/well
    y1 = max(1.0, seed_mt / mean_rate)
    y1 = min(y1, 0.5 * L)  # keep the anchor below the midpoint
    t0 = solve_logistic_t0(L, k_growth, 1, y1)

    return L, t0, mean_rate


def logistic_function(L, t, k, t0):

    return L / (1 + np.exp(-k *(t - t0)))


def generate_well_schedule_logistic(years, L,
                                    growth_rate,
                                    peak_year,
                                    injection_duration,
                                    candidate_locations,
                                    used_index):

    all_wells = []
    num_wells_drilled_annually = []
    drilled = 0

    for year in range(1, years + 1):

        # Drill only enough wells to keep the cumulative count on the
        # logistic curve, so the total converges to L rather than
        # overshooting (ceil of the derivative added up to 1 well/year).
        cum_target = logistic_function(L, year, growth_rate, peak_year)

        num_to_drill = max(0, int(round(cum_target)) - drilled)
        drilled += num_to_drill
        num_wells_drilled_annually.append(num_to_drill)

        new_wells = pooled_well_generator(num_to_drill,
                                   candidate_locations,
                                   used_index)

        for loc in new_wells:
            well = {
                'x': loc[0],
                'y': loc[1],
                'start_year': year,
                'end_year': year + injection_duration - 1
            }
            all_wells.append(well)

    return all_wells,num_wells_drilled_annually

def solve_logistic_t0(L, k, t, y):

    return t + math.log((L / y) - 1) / k


def pooled_well_generator(n,candidate_locations,used_index):
    # selects well locations from a predefined pool then updates used_index
    # to ensure those wells are not selected next time its called
    wells = candidate_locations[used_index[0] : used_index[0] + n]
    if len(wells) < n:
        raise ValueError("Not enough unused well locations in the candidate pool.")
    used_index[0] += n
    return wells
