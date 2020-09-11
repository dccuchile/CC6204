import json

import numpy as np

from src.handler import error, send_results, send_test
from src.homeworks.errors import Container, InvalidInput
from src.homeworks.tester import numpy_isclose

__tests_data = {}
__tests_expected = {}


def getter(question_number, test_number, test_set, code):
    if question_number not in test_set:
        raise Container(error(
            f"Question {question_number} not found in homework", code))
    if test_number not in test_set[question_number]:
        raise Container(
            error(
                f"Test {test_number} not found in Question {question_number}",
                code))
    return test_set[question_number][test_number]


def check(question_number, test_number, student_answer):
    try:
        value = getter(question_number, test_number,
                       __tests_expected, "test_output")
    except Container as e:
        return e.data

    try:
        status, result, comments = numpy_isclose(value, student_answer)
    except InvalidInput as e:
        return error(message=e.msg, code="invalid_input")
    else:
        return send_results(status=status, mask=result, comments=comments)


def get_test(question_number, test_number):
    try:
        value = getter(question_number, test_number, __tests_data, "test_data")
    except Container as e:
        return e.data
    else:
        return send_test(value)


def load_tests():
    with open("./data/homeworks/1.json", "r") as f:
        data = json.load(f)

    __tests_data.update(data["input"])
    expected = data["expected"]

    for question, tests in expected.items():
        for test, values in tests.items():
            __tests_expected[question][test] = np.array(values)
