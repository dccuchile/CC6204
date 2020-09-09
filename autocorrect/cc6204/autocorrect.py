from typing import Union

import numpy as np
import requests
import torch

from .exceptions import FailedTest, LibraryError, UserError


class AutoCorrect:
    all_test_data = {}

    def __init__(self, host: str, port: Union[int, str]):
        self.host = host
        self.port = port

        try:
            requests.get(f"http://{self.host}:{self.port}/ping")
        except requests.exceptions.ConnectionError:
            raise LibraryError(
                "Connection could not be stablished. Contact JP") from None
        except BaseException:
            raise LibraryError(
                "Unknown Error occurred. Contact JP") from None

    def sumbit(self, task: int, test: int, token: str, answer):
        if isinstance(answer, (np.ndarray, torch.Tensor)):
            answer = answer.tolist()
        elif not isinstance(answer, list):
            raise ValueError(
                "Supported submit values are numpy array, "
                "torch tensors and python lists. "
                f"Answer type is: {type(answer)}")

        try:
            response = requests.post(
                f"http://{self.host}:{self.port}/api/autocheck/{task}/{test}",
                json={
                    "token": token,
                    "answer": answer
                }).json()
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

            result = response["result"]
            msg = response["message"]
            if result == "correct":
                print("Correct Test!")
            elif result == "fail":
                raise FailedTest(msg)
            else:
                raise ValueError(
                    "Something went wrong and I don't know what to do. "
                    "Contact JP.")

    def get_test_data(self, task: int, test: int, token: str):
        try:
            data = self.all_test_data[task][test]
        except KeyError:

            try:
                response = requests.get(
                    f'http://{self.host}:{self.port}/api/tests/{task}/{test}',
                    params={
                        "token": token}).json()
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
                self.all_test_data[task][test] = data
        else:
            print("Using cached test data")

        return data
