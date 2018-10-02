from interface import Interface


class ParserInterface(Interface):
    def __init__(self, filename, **kwargs):
        pass

    def open(self):
        pass

    def get_data(self):
        pass

    def close(self):
        pass
