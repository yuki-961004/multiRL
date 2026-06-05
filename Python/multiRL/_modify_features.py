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
            _integer_value(value, colnames["block"])
            for value in _vector_to_list(data[colnames["block"]])
        ],
        "trial": [
            _integer_value(value, colnames["trial"])
            for value in _vector_to_list(data[colnames["trial"]])
        ],
    }


def _vector_to_list(value):
    if value is None:
        raise ValueError("run_m received a missing vector input.")

    if hasattr(value, "tolist"):
        value = value.tolist()

    return list(value)


def _integer_value(value, name):
    if value is None:
        raise ValueError(f"Column '{name}' contains a missing value.")

    try:
        numeric = float(str(value))
    except ValueError as error:
        raise ValueError(
            f"Column '{name}' must be coercible to integer."
        ) from error

    integer = int(numeric)
    if numeric != integer:
        raise ValueError(
            f"Column '{name}' must contain integer-valued data."
        )

    return integer


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
