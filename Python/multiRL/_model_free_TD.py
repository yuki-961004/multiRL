"""Model-free TD built-in model specification."""

from ._model_free_base import _model_free_builtin


def TD(params=None, priors=None, lower=None, upper=None, settings=None):
    return _model_free_builtin(
        name="TD",
        free={
            "alpha": 0.3,
            "beta": 0.5,
        },
        params=params,
        priors=priors,
        lower=lower,
        upper=upper,
        settings=settings,
    )
