"""Internal data feature conversion helpers."""


def _modify_features(data, id, colnames):
    if id is not None:
        ids = id if isinstance(id, (list, tuple, set)) else [id]
        data = data[data[colnames["subid"]].isin(ids)]

    return {
        "object": _matrix_to_list(data[colnames["object"]]),
        "reward": _numeric_matrix_to_list(data[colnames["reward"]]),
        "action": _vector_to_list(data[colnames["action"]]),
        "block": [
            int(value)
            for value in _vector_to_list(data[colnames["block"]])
        ],
        "trial": [
            int(value)
            for value in _vector_to_list(data[colnames["trial"]])
        ],
    }


def _vector_to_list(value):
    if value is None:
        raise ValueError("run_m received a missing vector input.")

    if hasattr(value, "tolist"):
        value = value.tolist()

    return list(value)


def _matrix_to_list(value):
    if value is None:
        raise ValueError("run_m received a missing matrix input.")

    if hasattr(value, "values"):
        value = value.values

    if hasattr(value, "tolist"):
        value = value.tolist()

    return [
        [str(cell) for cell in row]
        for row in value
    ]


def _numeric_matrix_to_list(value):
    if value is None:
        raise ValueError("run_m received a missing numeric matrix input.")

    if hasattr(value, "values"):
        value = value.values

    if hasattr(value, "tolist"):
        value = value.tolist()

    return [
        [float(cell) for cell in row]
        for row in value
    ]
