from Parsers.csv_parser import Parser
from PreProcessors.csv_pre_processor import PreProcessor
import logging
logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                    level=logging.DEBUG, filename='test.log')
logging.info('program start')
a = PreProcessor(filename="./index/avg_index.csv")
a.start()
logging.info('program stopped')
