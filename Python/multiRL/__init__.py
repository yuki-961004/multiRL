"""Python frontend for the multiRL C++ remake."""


def shell_run_m(*args, **kwargs):
    from ._shell import shell_run_m as shell_run_m_impl

    return shell_run_m_impl(*args, **kwargs)


def run_m(*args, **kwargs):
    return shell_run_m(*args, **kwargs)

__all__ = [
    "run_m",
    "shell_run_m",
]
