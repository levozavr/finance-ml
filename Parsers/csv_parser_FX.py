from Parsers.interface import ParserInterface
import csv
import logging
import datetime


class Parser(ParserInterface):
    def __init__(self, filename, **kwargs):
        self.filename = filename
        self.reader = None
        self.file = None
        self.__date = []
        self._dates = []

    def open(self):
        try:
            self.file = open(self.filename, mode='r')
            logging.info(f"open {self.filename} file")
            self.reader = csv.DictReader(self.file)
            self.__read()
            logging.info(f"read all information from {self.filename}")
        except Exception as e:
            logging.fatal(e)
            raise Exception(e)

    def __read(self):
        line_count = 0
        for row in self.reader:
            if line_count == 0:
                line_count = 1
            avg_row = row['money']
            date = row['date']
            try:
                self.__date.append(float(avg_row))
                self._dates.append(datetime.datetime.strptime(date, '%Y-%m-%d'))
                #self._dates.append(datetime.datetime.strptime(date, '%d.%m.%Y'))
            except Exception as e:
                print(e)

    def get_data(self):
        return self.__date

    def get_dates(self):
        return self._dates

    def get_indexes(self, date1, date2):
        index1, index2 = 0, 0
        for i, date in enumerate(self._dates):
            if date >= date1 and index1 == 0:
                index1 = (0, i)
            if date >= date2:
                index2 = (index1[1]+7, i)
                break
        return index1, index2

    def close(self):
        if self.file:
            self.file.close()
        else:
            logging.fatal("No file to close")
            raise Exception('No file to close')
