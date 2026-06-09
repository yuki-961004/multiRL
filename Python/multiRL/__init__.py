"""Python frontend for the multiRL C++ backend."""

import os


def _add_torch_dll_directories():
    candidates = []
    package_dir = os.path.dirname(os.path.abspath(__file__))
    candidates.append(package_dir)

    torch_root = os.environ.get("MULTIRL_LIBTORCH_DIR", "")
    if torch_root:
        torch_root = os.path.abspath(torch_root)
        candidates.append(torch_root)
        candidates.append(os.path.join(torch_root, "lib"))

    for candidate in candidates:
        if not os.path.isdir(candidate):
            continue
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(candidate)
        os.environ["PATH"] = candidate + os.pathsep + os.environ.get("PATH", "")


_add_torch_dll_directories()

from ._estimate_abc import estimate_abc
from ._estimate_mcmc import estimate_mcmc
from ._estimate_mle import estimate_mle
from ._estimate_map import estimate_map
from ._shell_fit_p import fit_p
from ._shell_rcv_d import rcv_d
from ._shell_rpl_e import rpl_e
from ._estimate_rnn import estimate_rnn
from ._model_free_RSTD import RSTD
from ._model_free_TD import TD
from ._model_free_Utility import Utility
from ._shell_run_m import run_m

__all__ = [
    "estimate_map",
    "estimate_abc",
    "estimate_mcmc",
    "estimate_mle",
    "estimate_rnn",
    "fit_p",
    "rcv_d",
    "rpl_e",
    "run_m",
    "RSTD",
    "TD",
    "Utility",
]
