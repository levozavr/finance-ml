from Parsers.csv_parser import CsvParser
import logging
logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                    level=logging.DEBUG, filename='test.log')
logging.info('program start')
a = CsvParser(filename="./index/avg_index.csv")
a.open()
print(a.get_data())
a.close()
logging.info('program stopped')
