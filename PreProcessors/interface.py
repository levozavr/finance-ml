from interface import Interface


class PreProcessorInterface(Interface):
    def __init__(self, filename, **kwargs):
        pass

    def start(self, ws_pred=20, ws_future=7, grade=20):
        pass

    def get_train(self):
        pass

    def get_test(self):
        pass

    def get_all_data(self):
        pass
