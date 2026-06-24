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
    "ccasr_noe": {
        "delta": 0.01,
        "alpha": 1,
        "k": 0.4851,
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
    "inverter_noe": {
        "delta": 0.01,
        "alpha": 0.1,
        "beta": 1,
        "k": 0.4851,
        "k_tet": 5,
        "tau_delay": 12,
        "n": 3.6,
        "n_tet": 2,
        "c2": 0.0631,
        "t_max": 960,
        "sampling": 10,
    },
    "double_inverter": {
        "delta": 0.01,
        "alpha": 0.1,
        "beta": 0.1,
        "gamma": 1,
        "k": 0.4851,
        "k_tet": 5,
        "k_lac": 2,
        "h1": 0.07100805,
        "h2": 0.0303,
        "tau_delay": 12,
        "n": 3.6,
        "n_tet": 2,
        "n_lac": 1,
        "c2": 0.0631,
        "t_max": 960,
        "sampling": 10,
    },
    "double_inverter_noe": {
        "delta": 0.01,
        "alpha": 0.1,
        "beta": 0.1,
        "gamma": 1,
        "k": 0.4851,
        "k_tet": 5,
        "k_lac": 2,
        "tau_delay": 12,
        "n": 3.6,
        "n_tet": 2,
        "n_lac": 1,
        "c2": 0.0631,
        "t_max": 960,
        "sampling": 10,
    },
    "ccasr_ode": {
        "delta": 0.01,
        "alpha": 1,
        "k": 0.4851,
        "tau_delay": 12,
        "n": 3.6,
        "c2": 0.0631,
        "t_max": 960,
        "sampling": 10,
        "x0": [0, 0],
    },
    "ode_inverter": {
        "delta": 0.01,
        "alpha": 0.1,
        "beta": 1,
        "k": 0.4851,
        "k_tet": 5,
        "tau_delay": 12,
        "n": 3.6,
        "n_tet": 2,
        "c2": 0.0631,
        "t_max": 960,
        "sampling": 10,
        "x0": [0, 0, 0],
    },
    "ode_double_inverter": {
        "delta": 0.01,
        "alpha": 0.1,
        "beta": 0.1,
        "gamma": 1,
        "k": 0.4851,
        "k_tet": 5,
        "k_lac": 2,
        "tau_delay": 12,
        "n": 3.6,
        "n_tet": 2,
        "n_lac": 1,
        "c2": 0.0631,
        "t_max": 960,
        "sampling": 10,
        "x0": [0, 0, 0, 0],
    },
}

CIRCUIT_PARAMETER_SCHEMAS = {
    "ccasr": ["delta", "alpha", "k", "h1", "h2", "tau_delay", "n", "c2"],
    "ccasr_noe": ["delta", "alpha", "k", "tau_delay", "n", "c2"],
    "ccasr_ode": ["delta", "alpha", "k", "tau_delay", "n", "c2"],
    "inverter": [
        "delta",
        "alpha",
        "beta",
        "k",
        "k_tet",
        "h1",
        "h2",
        "tau_delay",
        "n",
        "n_tet",
        "c2",
    ],
    "inverter_noe": [
        "delta",
        "alpha",
        "beta",
        "k",
        "k_tet",
        "tau_delay",
        "n",
        "n_tet",
        "c2",
    ],
    "ode_inverter": [
        "delta",
        "alpha",
        "beta",
        "k",
        "k_tet",
        "tau_delay",
        "n",
        "n_tet",
        "c2",
    ],
    "double_inverter": [
        "delta",
        "alpha",
        "beta",
        "gamma",
        "k",
        "k_tet",
        "k_lac",
        "h1",
        "h2",
        "tau_delay",
        "n",
        "n_tet",
        "n_lac",
        "c2",
    ],
    "double_inverter_noe": [
        "delta",
        "alpha",
        "beta",
        "gamma",
        "k",
        "k_tet",
        "k_lac",
        "tau_delay",
        "n",
        "n_tet",
        "n_lac",
        "c2",
    ],
    "ode_double_inverter": [
        "delta",
        "alpha",
        "beta",
        "gamma",
        "k",
        "k_tet",
        "k_lac",
        "tau_delay",
        "n",
        "n_tet",
        "n_lac",
        "c2",
    ],
}

TIMING_PARAMETERS = ["t_max", "sampling"]
ODE_OPTIONAL_PARAMETERS = ["x0", "measurement_noise", "std"]
MEASUREMENT_NOISE_ALIASES = {
    "ccasr_ode_measurement_noise",
    "ccasr_ode_with_measurement_noise",
    "ode_ccasr_measurement_noise",
    "ode_ccasr_with_measurement_noise",
    "inverter_ode_measurement_noise",
    "inverter_ode_with_measurement_noise",
    "ode_inverter_measurement_noise",
    "ode_inverter_with_measurement_noise",
    "double_inverter_ode_measurement_noise",
    "double_inverter_ode_with_measurement_noise",
    "ode_double_inverter_measurement_noise",
    "ode_double_inverter_with_measurement_noise",
}

_ALIASES = {
    "ccasr": "ccasr",
    "ccasr_inverter": "inverter",
    "inverter": "inverter",
    "double_inverter": "double_inverter",
    "ccasr_double_inverter": "double_inverter",
    "ccasr_noe": "ccasr_noe",
    "ccasr_no_e": "ccasr_noe",
    "ccasrnoe": "ccasr_noe",
    "noe_ccasr": "ccasr_noe",
    "ccasr_ode": "ccasr_ode",
    "ode_ccasr": "ccasr_ode",
    "ode_ccasr_noe": "ccasr_ode",
    "ccasr_noe_ode": "ccasr_ode",
    "ccasr_ode_measurement_noise": "ccasr_ode",
    "ccasr_ode_with_measurement_noise": "ccasr_ode",
    "ode_ccasr_measurement_noise": "ccasr_ode",
    "ode_ccasr_with_measurement_noise": "ccasr_ode",
    "inverter_noe": "inverter_noe",
    "inverter_no_e": "inverter_noe",
    "inverternoe": "inverter_noe",
    "ccasr_inverter_noe": "inverter_noe",
    "ccasr_inverter_no_e": "inverter_noe",
    "double_inverter_noe": "double_inverter_noe",
    "double_inverter_no_e": "double_inverter_noe",
    "doubleinverternoe": "double_inverter_noe",
    "ccasr_double_inverter_noe": "double_inverter_noe",
    "ccasr_double_inverter_no_e": "double_inverter_noe",
    "ode_inverter": "ode_inverter",
    "inverter_ode": "ode_inverter",
    "ode_inverter_ccasr": "ode_inverter",
    "ode_ccasr_inverter": "ode_inverter",
    "ccasr_inverter_ode": "ode_inverter",
    "ode_inverter_noe": "ode_inverter",
    "inverter_ode_measurement_noise": "ode_inverter",
    "inverter_ode_with_measurement_noise": "ode_inverter",
    "ode_inverter_measurement_noise": "ode_inverter",
    "ode_inverter_with_measurement_noise": "ode_inverter",
    "ode_double_inverter": "ode_double_inverter",
    "double_inverter_ode": "ode_double_inverter",
    "ode_ccasr_double_inverter": "ode_double_inverter",
    "ccasr_double_inverter_ode": "ode_double_inverter",
    "double_inverter_ode_measurement_noise": "ode_double_inverter",
    "double_inverter_ode_with_measurement_noise": "ode_double_inverter",
    "ode_double_inverter_measurement_noise": "ode_double_inverter",
    "ode_double_inverter_with_measurement_noise": "ode_double_inverter",
}

ODE_CIRCUITS = {"ccasr_ode", "ode_inverter", "ode_double_inverter"}

_SOLVER_ALIASES = {
    "deterministic": "deterministic",
    "ode": "deterministic",
    "gillespy": "gillespy_tau_hybrid",
    "gillespy_tau_hybrid": "gillespy_tau_hybrid",
    "tau_hybrid": "gillespy_tau_hybrid",
    "tauhybrid": "gillespy_tau_hybrid",
    "stochastic": "gillespy_tau_hybrid",
    "sde": "gillespy_tau_hybrid",
    "sde+noise": "gillespy_tau_hybrid",
    "sde+noise+heterogeneity": "gillespy_tau_hybrid",
}


def normalize_circuit_name(circuit):
    circuit_key = _normalize_key(circuit)
    return _ALIASES.get(circuit_key, circuit_key)


def is_supported_circuit(circuit):
    return normalize_circuit_name(circuit) in DEFAULT_CIRCUIT_PARAMS


def is_ode_circuit(circuit):
    return normalize_circuit_name(circuit) in ODE_CIRCUITS


def default_solver_for_circuit(circuit):
    return "deterministic" if is_ode_circuit(circuit) else "gillespy_tau_hybrid"


def normalize_solver_name(solver):
    solver_key = _normalize_key(solver)
    return _SOLVER_ALIASES.get(solver_key, solver_key)


def validate_solver_for_circuit(solver, circuit):
    solver_key = normalize_solver_name(solver)
    expected = default_solver_for_circuit(circuit)
    if solver_key != expected:
        raise ValueError(
            f"Circuit {circuit!r} expects solver {expected!r}, got {solver!r}. "
            "Use deterministic for ODE circuits and gillespy_tau_hybrid for stochastic circuits."
        )
    return solver_key


def default_circuit_params(circuit):
    requested_key = _normalize_key(circuit)
    circuit_key = normalize_circuit_name(circuit)
    if circuit_key not in DEFAULT_CIRCUIT_PARAMS:
        raise ValueError(
            f"No default parameters are defined for circuit {circuit!r}. "
            f"Available defaults: {sorted(DEFAULT_CIRCUIT_PARAMS)}"
        )
    params = copy.deepcopy(DEFAULT_CIRCUIT_PARAMS[circuit_key])
    if requested_key in MEASUREMENT_NOISE_ALIASES and is_ode_circuit(circuit_key):
        params["measurement_noise"] = True
    return clean_circuit_params(circuit_key, params)


def parameter_schema_for_circuit(circuit, include_timing=True, include_optional=True):
    circuit_key = normalize_circuit_name(circuit)
    if circuit_key not in CIRCUIT_PARAMETER_SCHEMAS:
        raise ValueError(
            f"No parameter schema is defined for circuit {circuit!r}. "
            f"Available schemas: {sorted(CIRCUIT_PARAMETER_SCHEMAS)}"
        )
    schema = list(CIRCUIT_PARAMETER_SCHEMAS[circuit_key])
    if include_timing:
        schema.extend(TIMING_PARAMETERS)
    if include_optional and is_ode_circuit(circuit_key):
        schema.extend(ODE_OPTIONAL_PARAMETERS)
    return schema


def clean_circuit_params(circuit, params, keep_unknown=False):
    if not isinstance(params, dict):
        raise ValueError("Circuit parameters must be provided as a dictionary.")

    allowed = parameter_schema_for_circuit(circuit, include_timing=True, include_optional=True)
    cleaned = {}
    for key in allowed:
        if key in params:
            cleaned[key] = copy.deepcopy(params[key])

    if keep_unknown:
        for key, value in params.items():
            if key not in cleaned:
                cleaned[key] = copy.deepcopy(value)

    return cleaned


def all_circuit_parameter_names(include_timing=True, include_optional=True):
    names = set()
    for circuit in CIRCUIT_PARAMETER_SCHEMAS:
        names.update(parameter_schema_for_circuit(circuit, include_timing, include_optional))
    return names


def is_measurement_noise_variant(circuit):
    return _normalize_key(circuit) in MEASUREMENT_NOISE_ALIASES


def _normalize_key(value):
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")
