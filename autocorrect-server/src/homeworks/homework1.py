from src.errors import Container, InvalidInput
from src.handler import error, send_results, send_test
from src.homeworks.tester import numpy_isclose
from src.homeworks.utils import load_tests_path, test_getter

__tests_data = {}
__tests_expected = {}


def check(question_number, test_number, student_answer):
    if not __tests_expected:
        load_tests()

    try:
        value = test_getter(
            question_number,
            test_number,
            __tests_expected,
            "test_output")
    except Container as e:
        return e.data

    try:
        status, result, comments = numpy_isclose(value, student_answer)
    except InvalidInput as e:
        return error(message=e.msg, code="invalid_input")
    else:
        return send_results(status=status, mask=result, comments=comments)


def get_test(question_number, test_number):
    if not __tests_data:
        load_tests()
    try:
        value = test_getter(
            question_number,
            test_number,
            __tests_data,
            "test_data")
    except Container as e:
        return e.data
    else:
        return send_test(value)


def load_tests():
    load_tests_path(
        "./tests/hw1.json",
        test_data=__tests_data,
        test_expected=__tests_expected)


def reload_tests():
    __tests_data.clear()
    __tests_expected.clear()
    load_tests()
