from PreProcessors.csv_pre_processor_assim_heatmap import PreProcessor
from Parsers.csv_parser_FX import Parser
import numpy as np

import logging
logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                    level=logging.DEBUG, filename='test.log')
logging.info('program start')
a = PreProcessor("./index/FX_EURKRW.csv")
a.start()
print(a.get_train())
print(a.get_val())
print(a.get_test())
logging.info('program stopped')
