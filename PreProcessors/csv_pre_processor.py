from PreProcessors.interface import PreProcessorInterface
from Parsers.csv_parser import Parser
from interface import implements
import matplotlib.pyplot as plt
import numpy as np
import math


class PreProcessor(implements(PreProcessorInterface)):
    def __init__(self, filename):
        self.parser = Parser(filename)
        self.parser.open()
        self.__pre_data = self.parser.get_data()
        self.parser.close()
        self.__train_data_x = None
        self.__train_data_y = None
        self.__test_data_x = None
        self.__test_data_y = None
        self.__all_data_x = []

        self.__all_data_y = []

    def start(self, ws_pred=20, ws_future=7):
        size = len(self.__pre_data)
        for i in range(size - ws_pred - ws_future):
            matr = np.zeros((ws_pred, ws_pred))
            tmp = np.array(self.__pre_data[i:i+ws_pred])/np.linalg.norm(np.array(self.__pre_data[i:i+ws_pred]))
            for ix, x in enumerate(tmp):
                for iy, y in enumerate(tmp):
                    matr[ix][iy] = x*y - math.sqrt(1-x*x)*math.sqrt(1-y*y)
            self.__all_data_x.append(matr)
            """______________________________________________________________________________________________"""
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.set_aspect('equal')
            plt.imshow(matr, interpolation='nearest', cmap=plt.cm.ocean)
            plt.colorbar()
            plt.show()
            """______________________________________________________________________________________________"""
            self.__all_data_y.append(self.__pre_data[i+ws_pred-1] - self.__pre_data[i+ws_pred+ws_future-1])
        for i in zip(self.__all_data_x,self.__all_data_y):
            print(i)

    def __process_train(self):
        pass

    def __process_test(self):
        pass

    def get_train(self):
        return self.__train_data_x, self.__train_data_y

    def get_test(self):
        return self.__test_data_x, self.__test_data_y

    def get_all_data(self):
        pass
