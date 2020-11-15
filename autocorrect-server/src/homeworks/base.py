import json
import os
from abc import ABC, abstractmethod

import numpy as np

from src.exceptions import Container
from src.messages import error, send_test


class Homework(ABC):
    _test_data = {}
    _test_expected = {}

    def __init__(self, test_filename, test_filepath='./tests'):
        self.test_file = os.path.join(test_filepath, test_filename)
        self.load_tests()

    @abstractmethod
    def check(self, question_number, data):
        raise NotImplementedError

    def get_test(self, question_number, data):
        try:
            test_number = self.extract_argument(data, 'test')

            value = self.test_getter(
                question_number,
                test_number,
                self._test_data,
                'test_data')
        except Container as e:
            return e.data
        else:
            return send_test(value)

    def load_tests(self):
        with open(self.test_file, 'r') as f:
            data = json.load(f)

        self._test_data.update(data['input'])
        self._test_expected.update(data['expected'])

        self.convert_test()

    @abstractmethod
    def convert_test(self):
        raise NotImplementedError

    def reload_tests(self):
        self._test_data.clear()
        self._test_expected.clear()
        self.load_tests()

    @staticmethod
    def extract_argument(data, arg_name):
        try:
            return data[arg_name]
        except KeyError:
            raise Container(
                error(
                    f'Missing argument `{arg_name}`',
                    'missing_arg'))

    @staticmethod
    def test_getter(question_number, test_number, test_set, code):
        if question_number not in test_set:
            raise Container(error(
                f'Question `{question_number}` not found in homework', code))
        if test_number not in test_set[question_number]:
            raise Container(
                error(
                    f'Test `{test_number}` not found in Question {question_number}',
                    code))
        return test_set[question_number][test_number]
