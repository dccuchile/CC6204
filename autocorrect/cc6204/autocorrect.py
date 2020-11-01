from collections import defaultdict
from typing import Union

import numpy as np
import requests
import torch

from .exceptions import FailedTest, LibraryError, MessageFromServer, UserError

_nested = lambda: defaultdict(_nested)


class AutoCorrect:
    _all_test_data = _nested()

    def __init__(self, host: str, port: Union[int, str]):
        self.host = host
        self.port = port

        try:
            val = requests.get(f"https://{self.host}:{self.port}/ping")
        except requests.exceptions.ConnectionError:
            raise LibraryError(
                "Connection could not be stablished. Contact JP") from None
        except BaseException:
            raise LibraryError(
                "Unknown Error occurred. Contact JP") from None
        else:
            try:
                val = val.json()
            except BaseException:
                print("Connection stablished")
            else:
                if "message" in val:
                    raise MessageFromServer(val["message"])
                else:
                    raise MessageFromServer(val)

    def sumbit(
            self,
            homework: int,
            question: str,
            test: str,
            token: str,
            answer,
            **kwargs):
        if isinstance(answer, (np.ndarray, torch.Tensor)):
            answer = answer.tolist()
        elif not isinstance(answer, (list, int, float)):
            raise ValueError(
                "Supported submit values are numpy array, "
                "torch tensors, python lists and int/floats. "
                f"Answer type is: {type(answer)}")

        try:
            response = requests.post(
                f"https://{self.host}:{self.port}/api/autocheck/{homework}/{question}",
                json={
                    **kwargs,
                    "token": token,
                    "test": str(test),
                    "student_answer": answer}).json()
        except requests.exceptions.ConnectionError:
            raise LibraryError(
                "Connection could not be stablished. Contact JP") from None
        except requests.exceptions.RequestException:
            raise LibraryError(
                "Request Error. Contact JP or try again later") from None
        except BaseException:
            raise LibraryError("Unknown Error occurred. Contact JP")
        else:
            if "error" in response:
                raise UserError(**response["error"])

            status = response["status"]
            mask = response["mask"]
            comments = response["comments"]
            if status == 1:
                print("Correct Test!")
            elif status == 0:
                raise FailedTest(mask, comments)
            else:
                raise ValueError(
                    "Something went wrong and I don't know what to do. "
                    "Contact JP.")

    def get_test_data(
            self,
            homework: int,
            question: str,
            test: str,
            token: str):

        data = self._all_test_data[homework][question][test]
        if not data:
            try:
                response = requests.get(
                    f'https://{self.host}:{self.port}/api/tests/{homework}/{question}',
                    params={
                        "token": token,
                        "test": str(test)}).json()
            except requests.exceptions.ConnectionError:
                raise LibraryError(
                    "Connection could not be stablished. Contact JP") from None
            except requests.exceptions.RequestException:
                raise LibraryError(
                    "Request Error. Contact JP or try again later") from None
            except BaseException:
                raise LibraryError(
                    "Unknown Error occurred. Contact JP") from None
            else:
                if "error" in response:
                    raise UserError(**response["error"])

                data = response["data"]
                self._all_test_data[homework][question][test] = data
        else:
            print("Using cached test data")

        return data
