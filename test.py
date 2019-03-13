from PreProcessors.csv_pre_processor_assim_heatmap import PreProcessor
from Parsers.csv_parser_FX import Parser
import numpy as np

import logging
logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                    level=logging.DEBUG, filename='test.log')
logging.info('program start')
a = PreProcessor("./index/FX_EURKRW.csv")
a.start()
a = PreProcessor("./index/FX_EURKRW.csv")
a.start(grade=8)
x, y = a.get_all_data()
a0, a1, a2 = 0, 0, 0
for i in y:
    if i == 0:
        a0 += 1
    if i == 1:
        a1 += 1
    if i == 2:
        a2 += 1
print(a0, a1, a2)
x, y = a.get_val()
print(y)
logging.info('program stopped')
