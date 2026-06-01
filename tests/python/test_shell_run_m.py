import importlib
import math
import pathlib
import sys

import pytest


def import_shell_module():
    root = pathlib.Path(__file__).resolve().parents[2]
    candidates = []

    for directory in (root / "build").rglob("_shell_run_m*"):
        if directory.suffix in {".pyd", ".so", ".dll"}:
            candidates.append(directory.parent)

    for directory in (root / "build-python").rglob("_shell_run_m*"):
        if directory.suffix in {".pyd", ".so", ".dll"}:
            candidates.append(directory.parent)

    for directory in candidates:
        sys.path.insert(0, str(directory))

    try:
        return importlib.import_module("_shell_run_m")
    except ImportError as error:
        pytest.skip(
            "Build Python wrapper first with "
            "`cmake -S . -B build-python -DMULTIRLCPP_BUILD_PYTHON=ON` "
            "and `cmake --build build-python --config Release`: "
            f"{error}"
        )


def test_python_shell_run_m_td_policy_off_smoke():
    shell = import_shell_module()

    params = {
        "alpha": 0.3,
        "beta": 0.5,
        "gamma": 1.0,
        "delta": 0.1,
        "epsilon": math.nan,
        "zeta": 0.0,
        "seed": 123.0,
        "L": math.nan,
        "penalty": 1.0,
        "Q0": math.nan,
        "reset": math.nan,
        "lapse": 0.01,
        "threshold": 20.0,
        "bonus": 0.0,
        "weight": 1.0,
        "capacity": 0.0,
        "sticky": 0.0,
    }

    result = shell.shell_run_m(
        object=[["A", "B"], ["A", "B"], ["C", "D"]],
        reward=[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        action=["A", "B", "D"],
        block=[1, 1, 1],
        trial=[1, 2, 3],
        cue=["A", "B", "C", "D"],
        rsp=["A", "B", "C", "D"],
        params=params,
        free_names=["alpha", "beta"],
        system=["RL"],
        policy="off",
        name="TD",
        mode="fitting",
        estimate="MLE",
    )

    assert result["metric"]["LogL"] == pytest.approx(3.0 * math.log(0.5))
    assert result["metric"]["AIC"] == pytest.approx(
        2.0 * 2.0 - 2.0 * result["metric"]["LogL"]
    )
    assert "behave" in result["result"]
