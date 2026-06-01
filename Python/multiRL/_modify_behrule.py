"""Internal behrule conversion helpers."""


def _modify_behrule(behrule, cue=None, rsp=None):
    if behrule is None:
        behrule = {}

    cue = behrule.get("cue", cue)
    rsp = behrule.get("rsp", rsp)

    if cue is None or rsp is None:
        raise ValueError("behrule must contain cue and rsp.")

    return {
        "cue": _vector_to_list(cue),
        "rsp": _vector_to_list(rsp),
    }


def _vector_to_list(value):
    if hasattr(value, "tolist"):
        value = value.tolist()

    return list(value)
