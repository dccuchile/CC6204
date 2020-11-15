import numpy as np

from src.exceptions import InvalidInput


def numpy_isclose(true_values, student_values, msg, equal_nan=True):
    try:
        student = np.array(student_values)
    except BaseException as e:
        raise InvalidInput(e.args)

    if true_values.shape != student.shape:
        raise InvalidInput(
            f'[{msg}] Dimensions does not match. '
            f'Expected: {true_values.shape}, Given: {student.shape}')

    result = np.isclose(student, true_values, equal_nan=equal_nan)
    status = int(np.all(result))
    comments = f'[{msg}] {result.sum() / result.size}% correct'

    # correct: correct or fail
    # comments: where it failed or text comments
    return status, result.tolist(), comments


def simple_max_value(true_value, student_value, msg):
    if not isinstance(student_value, (int, float)):
        raise InvalidInput(
            f'[{msg}] Type Error. Expected `float` or `int` but given: '
            f'{type(student_value)}')

    val = student_value <= true_value
    status = int(val)
    if status == 0:
        comments = f'[{msg}] Given value is greater that allowed maximum value'
    else:
        comments = f'[{msg}] OK'

    return status, val, comments
