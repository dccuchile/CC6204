import json

from src.errors import Container
from src.handler import error


def test_getter(question_number, test_number, test_set, code):
    if question_number not in test_set:
        raise Container(error(
            f"Question {question_number} not found in homework", code))
    if test_number not in test_set[question_number]:
        raise Container(
            error(
                f"Test {test_number} not found in Question {question_number}",
                code))
    return test_set[question_number][test_number]


def load_tests_path(tests_path, test_data, test_expected):
    with open(tests_path, "r") as f:
        data = json.load(f)

    test_data.update(data["input"])
    test_expected.update(data["expected"])


def extract_argument(data, arg_name):
    try:
        return data[arg_name]
    except KeyError:
        raise Container(error(f"Missing argument `{arg_name}`", "missing_arg"))
