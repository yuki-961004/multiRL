"""Internal settings conversion helpers."""


def _modify_settings(
    settings,
    system=None,
    generate=False,
    name="TD",
    mode="fitting",
    estimate="MLE",
):
    if settings is None:
        settings = {}

    out = {
        "system": system,
        "generate": bool(generate),
        "name": name,
        "mode": mode,
        "estimate": estimate,
    }
    out.update(settings)

    if out["system"] is None:
        out["system"] = ["RL"]

    return out
