import numpy as np

from src.homeworks.errors import InvalidInput


def numpy_isclose(true_values, student_values):
    true = np.array(true_values)

    try:
        student = np.array(student_values)
    except BaseException as e:
        raise InvalidInput(e.args)

    if true.shape != student.shape:
        raise InvalidInput("Dimensions does not match")

    result = np.isclose(true, student)
    status = int(np.all(result))
    comments = f"{result.sum() / result.size}% correct"

    # correct: correct or fail
    # comments: where it failed or text comments
    return status, result.tolist(), comments
