"""
Default simulation parameters used when a recipe does not provide overrides.
"""

import copy


DEFAULT_CIRCUIT_PARAMS = {
    "ccasr": {
        "delta": 0.01,
        "alpha": 1,
        "k": 0.4851,
        "h1": 0.07100805,
        "h2": 0.0303,
        "tau_delay": 12,
        "n": 3.6,
        "c2": 0.0631,
        "t_max": 960,
        "sampling": 10,
    },
    "inverter": {
        "delta": 0.01,
        "alpha": 0.1,
        "beta": 1,
        "k": 0.4851,
        "k_tet": 5,
        "h1": 0.07100805,
        "h2": 0.0303,
        "tau_delay": 12,
        "n": 3.6,
        "n_tet": 2,
        "c2": 0.0631,
        "t_max": 960,
        "sampling": 10,
    },
}

_ALIASES = {
    "ccasr": "ccasr",
    "ccasr_inverter": "inverter",
    "ccasr-inverter": "inverter",
    "inverter": "inverter",
}


def normalize_circuit_name(circuit):
    return _ALIASES.get(str(circuit).lower(), str(circuit).lower())


def default_circuit_params(circuit):
    circuit_key = normalize_circuit_name(circuit)
    if circuit_key not in DEFAULT_CIRCUIT_PARAMS:
        raise ValueError(
            f"No default parameters are defined for circuit {circuit!r}. "
            f"Available defaults: {sorted(DEFAULT_CIRCUIT_PARAMS)}"
        )
    return copy.deepcopy(DEFAULT_CIRCUIT_PARAMS[circuit_key])
