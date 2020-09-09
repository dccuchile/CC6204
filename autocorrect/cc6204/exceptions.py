class UserError(Exception):
    def __init__(self, code, text):
        self.code = code
        self.text = text

    def __str__(self):
        return f"UserError with code {self.code}: {self.text}"


class FailedTest(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return (
            "Test Failed.\n"
            "Here is a mask of the correct and "
            "incorrect samples you provided:\n"
            f"{self.msg}")


class LibraryError(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return f"LibraryError with message: {self.msg}"
