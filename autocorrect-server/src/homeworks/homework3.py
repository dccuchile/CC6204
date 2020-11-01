import numpy as np

from src.errors import Container, InvalidInput
from src.handler import error, send_results, send_test
from src.homeworks.tester import numpy_isclose, simple_max_value
from src.homeworks.utils import extract_argument, load_tests_path, test_getter

__tests_data = {}
__tests_expected = {}


def check(question_number, data):
    if not __tests_expected:
        load_tests()

    try:
        test = extract_argument(data, "test")
        student_answer = extract_argument(data, "student_answer")

        expected = test_getter(
            question_number,
            test,
            __tests_expected,
            "test_output")

        try:
            if question_number == "1b":
                status, result, comments = simple_max_value(
                    expected,
                    student_answer,
                    msg="dropout")
            else:
                status, result, comments = numpy_isclose(
                    expected,
                    student_answer,
                    msg="value",
                    equal_nan=False)
        except InvalidInput as e:
            return error(message=e.msg, code="invalid_input")
        else:
            return send_results(
                status=status,
                mask=result,
                comments=comments)

    except Container as e:
        return e.data


def get_test(question_number, data):
    if not __tests_data:
        load_tests()

    try:
        test = extract_argument(data, "test")

        value = test_getter(
            question_number,
            test,
            __tests_data,
            "test_data")
    except Container as e:
        return e.data
    else:
        return send_test(value)


def load_tests():
    load_tests_path(
        "./tests/hw3.json",
        test_data=__tests_data,
        test_expected=__tests_expected)

    for question_id, tests in __tests_expected.items():
        for test_id, values in tests.items():
            if not isinstance(values, list):
                continue
            __tests_expected[question_id][test_id] = np.array(values)


def reload_tests():
    __tests_data.clear()
    __tests_expected.clear()
    load_tests()
