from functools import partial

import numpy as np

from src.exceptions import Container, InvalidInput
from src.homeworks.base import Homework
from src.homeworks.methods import numpy_isclose, simple_max_value
from src.messages import error, send_results


class __homework4(Homework):
    def __init__(self):
        super().__init__(test_filename='hw4.json')
        self.special_questions = ('1a', '1b')

    def convert_test(self):
        for question_id, tests in self._test_expected.items():
            for test_id, values in tests.items():
                # leave time as a float, only convert value -> list
                self._test_expected[question_id][test_id]['value'] = np.array(
                    values['value'])

    def _check(self, func, student, expected):
        try:
            status, result, comments = func(expected, student)
        except InvalidInput as e:
            return error(message=e.msg, code='invalid_input')

        else:
            return send_results(
                status=status,
                mask=result,
                comments=comments)

    def check(self, question_number, data):
        try:
            test_number = self.extract_argument(data, 'test')
            student_answer = self.extract_argument(data, 'student_answer')

            student_time = None
            if question_number in self.special_questions:
                student_time = self.extract_argument(data, 'time')

            expected = self.test_getter(
                question_number,
                test_number,
                self._test_expected,
                'test_output')
            expected_value = expected['value']
            expected_time = expected.get('time', None)

            numpy_isclose_func = partial(
                numpy_isclose, msg='value', equal_nan=True)
            max_value_func = partial(simple_max_value, msg='time')

            # check the value first
            res_val = self._check(
                numpy_isclose_func,
                expected_value,
                student_answer)
            if res_val['status'] == 'error' or res_val['result_status'] == 0:
                # if error or value is wrong, then return
                return res_val

            # if one of the special questions
            if question_number in self.special_questions:
                # check time
                res_time = self._check(
                    max_value_func,
                    expected_time,
                    student_time)
                # if error or wrong time, then return
                if res_time['status'] == 'error' or \
                        res_time['result_status'] == 0:
                    return res_time

            # if everything is correct, return correct test
            return send_results(
                status=1,
                mask=[],
                comments='Correct Test.')

        except Container as e:
            return e.data


homework4 = __homework4()
