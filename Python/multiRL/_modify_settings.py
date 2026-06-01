"""Internal settings conversion helpers."""


def _modify_settings(
    settings,
    system=None,
    policy="off",
    name="TD",
    mode="fitting",
    estimate="MLE",
):
    if settings is None:
        settings = {}

    out = {
        "system": system,
        "policy": policy,
        "name": name,
        "mode": mode,
        "estimate": estimate,
    }
    out.update(settings)

    if out["system"] is None:
        out["system"] = ["RL"]

    return out
