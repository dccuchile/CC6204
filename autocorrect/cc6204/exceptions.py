class UserError(Exception):
    def __init__(self, code, message, **kwargs):
        self.code = code
        self.message = message

    def __str__(self):
        return f'UserError with code `{self.code}`: {self.message}'


class FailedTest(Exception):
    def __init__(self, mask, comments):
        self.mask = mask
        self.comments = comments

    def __str__(self):
        return (
            'Test Failed.\n'
            f'{self.comments}\n'
            'Here is a mask of the correct and '
            'incorrect samples you provided:\n'
            f'{self.mask}')


class NestedFailedTest(Exception):
    def __init__(self, failed_tests, message):
        self.failed_tests = failed_tests
        self.message = message

    def __str__(self):
        rec_tests = []
        for key, test in self.failed_tests.items():
            rec_tests.append(f'{key}: {str(test)}')

        tests_string = '\n'.join(rec_tests)

        return f'{self.message}: {tests_string}'


class LibraryError(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return f'LibraryError with message: {self.msg}'


class MessageFromServer(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return f'Message: {self.msg}'
