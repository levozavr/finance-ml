from Parsers.interface import ParserInterface
import csv
import logging


class Parser(ParserInterface):
    def __init__(self, filename, **kwargs):
        self.filename = filename
        self.reader = None
        self.file = None
        self.__date = []

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
            logging.fatal("No file to close")
            raise Exception('No file to close')
