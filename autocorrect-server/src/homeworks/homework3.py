import numpy as np

from src.exceptions import Container, InvalidInput
from src.homeworks.base import Homework
from src.homeworks.methods import numpy_isclose, simple_max_value
from src.messages import error, send_results


class __homework3(Homework):
    def __init__(self):
        super().__init__(test_filename='hw3.json')

    def convert_test(self):
        for question_id, tests in self._test_expected.items():
            for test_id, values in tests.items():
                if not isinstance(values, list):
                    continue
                self._test_expected[question_id][test_id] = np.array(values)

    def check(self, question_number, data):
        try:
            test_number = self.extract_argument(data, 'test')
            student_answer = self.extract_argument(data, 'student_answer')

            expected = self.test_getter(
                question_number,
                test_number,
                self._test_expected,
                'test_output')

            try:
                if question_number == '1b':
                    status, result, comments = simple_max_value(
                        expected,
                        student_answer,
                        msg='dropout')
                else:
                    status, result, comments = numpy_isclose(
                        expected,
                        student_answer,
                        msg='value',
                        equal_nan=False)

            except InvalidInput as e:
                return error(message=e.msg, code='invalid_input')

            else:
                return send_results(
                    status=status,
                    mask=result,
                    comments=comments)

        except Container as e:
            return e.data


homework3 = __homework3()
