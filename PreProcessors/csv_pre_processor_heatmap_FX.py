from PreProcessors.interface import PreProcessorInterface
from Parsers.csv_parser_FX import Parser
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import np_utils
import math
import datetime


class PreProcessor(PreProcessorInterface):
    def __init__(self, filename, classes=3, date1=datetime.datetime(2015,9,4,0,0), date2=datetime.datetime(2017,1,1,0,0)):
        self.classes=classes
        self.parser = Parser(filename)
        self.parser.open()
        self.__pre_data = self.parser.get_data()
        i1, i2 = self.parser.get_indexes(date1, date2)
        self.i1 = i1
        self.i2 = i2
        self.parser.close()
        self.__train_data_x = None
        self.__train_data_y = None
        self.__test_data_x = None
        self.__test_data_y = None
        self.__all_data_x = []
        self.__all_data_y = []
        self.__train_len = 0
        self.__len = 0
        self.__ws = 0

    def start(self, ws_pred=20, ws_future=7, grade=20):
        self.__ws = ws_pred
        size = len(self.__pre_data)
        for i in range(size - ws_pred - ws_future):
            matr = self.__matrix_compute(i, ws_pred)
            self.__all_data_x.append(matr)
            self.__all_data_y.append(self.__trend_compute(i, ws_pred, ws_future, grade))
        self.__process_train()
        self.__process_valid()
        self.__process_test()

    def __matrix_compute(self, i, j):
        matr = np.zeros((j, j))
        data = self.__pre_data[i:i + j]
        tmp = np.array(data) / np.linalg.norm(np.array(data))
        for ix, x in enumerate(tmp):
            for iy, y in enumerate(tmp):
                matr[ix][iy] = x * y - math.sqrt(1 - x * x) * math.sqrt(1 - y * y)
        return matr

    def __trend_compute(self, i, j, k, g=20):
        delta =  self.__pre_data[i+j+k-1] - self.__pre_data[i+j-1]
        if self.classes ==3 :
            if delta < -g:
                return 0
            elif -g <= delta <= g:
                return 1
            elif g < delta:
                return 2
        elif self.classes == 2:
            if delta <= g:
                return 0
            elif g < delta:
                return 1


    @staticmethod
    def plt_show(matr):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_aspect('equal')
        plt.imshow(matr, interpolation='nearest', cmap=plt.cm.ocean)
        plt.colorbar()
        plt.show()

    def __process_train(self):
        self.__train_data_x = np.array(self.__all_data_x[self.i1[0]:self.i1[1]])\
            .reshape(self.i1[1], self.__ws, self.__ws, 1)
        self.__train_data_y = np.array(self.__all_data_y[self.i1[0]:self.i1[1]])
        self.__train_data_y = np_utils.to_categorical(self.__train_data_y, self.classes)

    def __process_valid(self):
        self.__val_data_x = np.array(self.__all_data_x[self.i2[0]:self.i2[1]]) \
            .reshape(self.i2[1]-self.i2[0], self.__ws, self.__ws, 1)
        print(self.__val_data_x.shape)
        self.__val_data_y = self.__all_data_y[self.i2[0]:self.i2[1]]
        self.__val_data_y = np_utils.to_categorical(self.__val_data_y, self.classes)
        print(self.__val_data_y.shape)

    def __process_test(self):
        self.__test_data_x = np.array(self.__all_data_x[self.i2[1]+7:])\
            .reshape(len(self.__all_data_x)-self.i2[1]-7, self.__ws, self.__ws, 1)
        self.__test_data_y = np.array(self.__all_data_y[self.i2[1]+7:])
        self.__test_data_y = np_utils.to_categorical(self.__test_data_y, self.classes)

    def get_train(self):
        return self.__train_data_x, self.__train_data_y

    def get_test(self):
        return self.__test_data_x, self.__test_data_y

    def get_val(self):
        return self.__val_data_x, self.__val_data_y

    def get_all_data(self):
        return self.__all_data_x, self.__all_data_y
