"""Model-free Utility built-in model specification."""

from ._model_free_list import _model_free_builtin


def Utility(params=None, priors=None, lower=None, upper=None, settings=None):
    return _model_free_builtin(
        name="Utility",
        free={
            "alpha": 0.3,
            "beta": 0.5,
            "gamma": 0.5,
        },
        params=params,
        priors=priors,
        lower=lower,
        upper=upper,
        settings=settings,
    )
