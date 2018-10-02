from Parsers.interface import ParserInterface
from interface import implements
import csv


class CsvParser(implements(ParserInterface)):
    def __init__(self, filename, **kwargs):
        self.filename = filename
        self.reader = None
        self.file = None
        self.__date = []

    def open(self):
        self.file = open(self.filename, mode='r')
        self.reader = csv.DictReader(self.file)
        self.__read()

    def __read(self):
        line_count = 0
        for row in self.reader:
            if line_count == 0:
                line_count = 1
            avg_row = row['avg']
            try:
                self.__date.append(float(avg_row))
            except Exception as e:
                print(e)

    def get_data(self):
        return self.__date

    def close(self):
        if self.file:
            self.file.close()
        else:
            raise Exception('No file')