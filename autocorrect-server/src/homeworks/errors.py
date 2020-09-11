
class InvalidInput(Exception):
    def __init__(self, msg):
        self.msg = msg


class Container(Exception):
    def __init__(self, data):
        self.data = data
