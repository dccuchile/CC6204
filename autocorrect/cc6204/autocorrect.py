import warnings
from collections import defaultdict
from typing import Any, Dict, List, Union

import numpy as np
import requests
import torch

from cc6204.utils import check_list_types

from .exceptions import (
    FailedTest,
    LibraryError,
    MessageFromServer,
    NestedFailedTest,
    UserError
)

_nested = lambda: defaultdict(_nested)


class AutoCorrect:
    _all_test_data = _nested()
    protocol = 'https'

    def __init__(self, host: str, port: Union[int, str]):
        self.host = host
        self.port = port

        try:
            val = requests.get(
                f'{self.protocol}://{self.host}:{self.port}/ping')
        except requests.exceptions.ConnectionError:
            raise LibraryError(
                'Connection could not be stablished. Contact JP') from None
        except BaseException:
            raise LibraryError(
                'Unknown Error occurred. Contact JP') from None
        else:
            try:
                val = val.json()
            except BaseException:
                raise MessageFromServer(val)
            else:
                if val['status'] == 'OK':
                    print('Connection stablished')
                else:
                    raise MessageFromServer(val['message'])

    def submit(
            self,
            homework: int,
            question: str,
            test: str,
            token: str,
            answer,
            verbose: bool = True,
            **kwargs):
        answer = check_list_types(answer)

        try:
            response = requests.post(
                f'{self.protocol}://{self.host}:{self.port}/api/autocheck/{homework}/{question}',
                json={
                    **kwargs,
                    'token': token,
                    'test': str(test),
                    'student_answer': answer})

            if response.status_code != 200:
                raise MessageFromServer(
                    f'Status code is not 200: {response.status_code}. Contact JP')

            response = response.json()

        except MessageFromServer as e:
            raise e
        except requests.exceptions.ConnectionError:
            raise LibraryError(
                'Connection could not be stablished. Contact JP') from None
        except requests.exceptions.RequestException:
            raise LibraryError(
                'Request Error. Contact JP or try again later') from None
        except BaseException:
            raise LibraryError(
                'Unknown Error occurred. Contact JP, save the rest of the error message.')
        else:
            if response['status'] == 'error':
                raise UserError(**response)

            if response['status'] == 'OK':
                status = response['result_status']
                mask = response['mask']
                comments = response['comments']
                if status == 1:
                    if verbose:
                        print('Correct Test!')
                elif status == 0:
                    raise FailedTest(mask, comments)
                else:
                    raise ValueError(
                        'Something went wrong and I don\'t know what to do. '
                        'Contact JP.')

    def sumbit(self, *args, **kwargs):
        warnings.warn(
            'Old method `sumbit` has been renamed to `submit`, please use that instead.',
            category=FutureWarning,
            stacklevel=2)
        return self.submit(*args, **kwargs)

    def submit_check_some(
            self,
            homework: int,
            question: str,
            tests: List[str],
            token: str,
            answer_dict: Dict[str, Any],
            required_number: int):

        current_correct = 0
        acc_results = {}

        for test in tests:
            try:
                answer = answer_dict[test]
                self.submit(
                    homework=homework,
                    question=question,
                    test=test,
                    token=token,
                    answer=answer,
                    verbose=False)

            except KeyError:
                raise UserError(
                    'key_mismatch',
                    '`answer` keys must match with the tests given')
            except FailedTest as e:
                acc_results[test] = e
            else:
                current_correct += 1

        if current_correct >= required_number:
            print('Correct Test!')
        else:
            raise NestedFailedTest(
                failed_tests=acc_results,
                message=f'You got {current_correct} of the {required_number} required tests to pass the test')

    def get_test_data(
            self,
            homework: int,
            question: str,
            test: str,
            token: str):

        data = self._all_test_data[homework][question][test]
        if not data:
            try:
                response = requests.post(
                    f'{self.protocol}://{self.host}:{self.port}/api/tests/{homework}/{question}',
                    json={
                        'token': token,
                        'test': str(test)})

                if response.status_code != 200:
                    raise MessageFromServer(
                        f'Status code is not 200: {response.status_code}. Contact JP')

                response = response.json()

            except requests.exceptions.ConnectionError:
                raise LibraryError(
                    'Connection could not be stablished. Contact JP') from None
            except requests.exceptions.RequestException:
                raise LibraryError(
                    'Request Error. Contact JP or try again later') from None
            except BaseException:
                raise LibraryError(
                    'Unknown Error occurred. Contact JP, save the rest of the error message.')
            else:
                if response['status'] == 'error':
                    raise UserError(**response)

                if response['status'] == 'OK':
                    data = response['data']
                    self._all_test_data[homework][question][test] = data
        else:
            print('Using cached test data')

        return data
