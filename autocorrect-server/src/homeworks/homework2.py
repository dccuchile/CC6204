import numpy as np

from src.exceptions import Container, InvalidInput
from src.homeworks.base import Homework
from src.homeworks.methods import numpy_isclose, simple_max_value
from src.messages import error, send_results


class __homework2(Homework):
    def __init__(self):
        super().__init__(test_filename='hw2.json')

    def convert_test(self):
        for question_id, tests in self._test_expected.items():
            for test_id, values in tests.items():
                # leave time as a float, only convert value -> list
                self._test_expected[question_id][test_id]['value'] = np.array(
                    values['value'])

    def check(self, question_number, data):
        try:
            test_number = self.extract_argument(data, 'test')
            student_answer = self.extract_argument(data, 'student_answer')
            student_time = self.extract_argument(data, 'time')

            expected = self.test_getter(
                question_number,
                test_number,
                self._test_expected,
                'test_output')

            expected_value = expected['value']
            expected_time = expected['time']

            try:
                value_status, value_result, value_comments = numpy_isclose(
                    expected_value,
                    student_answer,
                    msg='value',
                    equal_nan=True)
                time_status, time_result, time_comments = simple_max_value(
                    expected_time,
                    student_time,
                    msg='time')

            except InvalidInput as e:
                return error(message=e.msg, code='invalid_input')

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
                        comments='Correct Test.')

        except Container as e:
            return e.data


homework2 = __homework2()
