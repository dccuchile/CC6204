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
        student_time = extract_argument(data, "time")

        expected = test_getter(
            question_number,
            test,
            __tests_expected,
            "test_output")

        expected_value = expected["value"]
        expected_time = expected["time"]

        try:
            value_status, value_result, value_comments = numpy_isclose(
                expected_value,
                student_answer,
                msg="value",
                equal_nan=True)
            time_status, time_result, time_comments = simple_max_value(
                expected_time,
                student_time,
                msg="time")
        except InvalidInput as e:
            return error(message=e.msg, code="invalid_input")
        else:
            if value_status == 0:
                return send_results(
                    status=value_status,
                    mask=value_result,
                    comments=value_comments)
            elif time_status == 0:
                return send_results(
                    status=time_status,
                    mask=time_result,
                    comments=time_comments)
            else:
                return send_results(
                    status=1,
                    mask=[],
                    comments="Correct Test.")

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
        "./tests/hw2.json",
        test_data=__tests_data,
        test_expected=__tests_expected)

    for question_id, tests in __tests_expected.items():
        for test_id, values in tests.items():
            # leave time as a float
            __tests_expected[question_id][test_id]["value"] = np.array(
                values["value"])


def reload_tests():
    __tests_data.clear()
    __tests_expected.clear()
    load_tests()
