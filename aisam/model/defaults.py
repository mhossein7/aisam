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
}

_ALIASES = {
    "ccasr": "ccasr",
    "ccasr_inverter": "inverter",
    "inverter": "inverter",
    "ccasr_noe": "ccasr_noe",
    "ccasr_no_e": "ccasr_noe",
    "ccasrnoe": "ccasr_noe",
    "noe_ccasr": "ccasr_noe",
    "ccasr_ode": "ccasr_ode",
    "ode_ccasr": "ccasr_ode",
    "ode_ccasr_noe": "ccasr_ode",
    "ccasr_noe_ode": "ccasr_ode",
    "inverter_noe": "inverter_noe",
    "inverter_no_e": "inverter_noe",
    "inverternoe": "inverter_noe",
    "ccasr_inverter_noe": "inverter_noe",
    "ccasr_inverter_no_e": "inverter_noe",
    "ode_inverter": "ode_inverter",
    "inverter_ode": "ode_inverter",
    "ode_inverter_ccasr": "ode_inverter",
    "ode_ccasr_inverter": "ode_inverter",
    "ccasr_inverter_ode": "ode_inverter",
    "ode_inverter_noe": "ode_inverter",
}

ODE_CIRCUITS = {"ccasr_ode", "ode_inverter"}

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
    circuit_key = str(circuit).strip().lower().replace("-", "_").replace(" ", "_")
    return _ALIASES.get(circuit_key, circuit_key)


def is_supported_circuit(circuit):
    return normalize_circuit_name(circuit) in DEFAULT_CIRCUIT_PARAMS


def is_ode_circuit(circuit):
    return normalize_circuit_name(circuit) in ODE_CIRCUITS


def default_solver_for_circuit(circuit):
    return "deterministic" if is_ode_circuit(circuit) else "gillespy_tau_hybrid"


def normalize_solver_name(solver):
    solver_key = str(solver).strip().lower().replace("-", "_").replace(" ", "_")
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
    circuit_key = normalize_circuit_name(circuit)
    if circuit_key not in DEFAULT_CIRCUIT_PARAMS:
        raise ValueError(
            f"No default parameters are defined for circuit {circuit!r}. "
            f"Available defaults: {sorted(DEFAULT_CIRCUIT_PARAMS)}"
        )
    return copy.deepcopy(DEFAULT_CIRCUIT_PARAMS[circuit_key])
