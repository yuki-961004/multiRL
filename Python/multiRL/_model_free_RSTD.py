"""Model-free RSTD built-in model specification."""

from ._model_free_list import _model_free_builtin


def RSTD(params=None, priors=None, lower=None, upper=None, settings=None):
    return _model_free_builtin(
        name="RSTD",
        free={
            "alphaN": 0.3,
            "alphaP": 0.3,
            "beta": 0.5,
        },
        params=params,
        priors=priors,
        lower=lower,
        upper=upper,
        settings=settings,
    )
