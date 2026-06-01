"""Internal column-name conversion helpers."""


def _modify_colnames(data, colnames):
    if colnames is None:
        colnames = {}

    out = {
        "subid": "Subject",
        "block": "Block",
        "trial": "Trial",
        "object": None,
        "reward": None,
        "action": "Action",
        "exinfo": None,
    }
    out.update(colnames)

    if out["object"] is None:
        out["object"] = [
            name for name in list(data.columns)
            if str(name).startswith("Object_")
        ]

    if out["reward"] is None:
        out["reward"] = [
            name for name in list(data.columns)
            if str(name).startswith("Reward_")
        ]

    return out
