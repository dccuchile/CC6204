class UserError(Exception):
    def __init__(self, code, text):
        self.code = code
        self.text = text

    def __str__(self):
        return f"UserError with code {self.code}: {self.text}"


class FailedTest(Exception):
    def __init__(self, mask, comments):
        self.mask = mask
        self.comments = comments

    def __str__(self):
        return (
            "Test Failed.\n"
            f"{self.comments}\n"
            "Here is a mask of the correct and "
            "incorrect samples you provided:\n"
            f"{self.mask}")


class LibraryError(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return f"LibraryError with message: {self.msg}"


class MessageFromServer(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return f"Message: {self.msg}"
