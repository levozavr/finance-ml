from PreProcessors.csv_pre_processor_inctiments import PreProcessor
from Parsers.csv_parser_FX import Parser
import logging
logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                    level=logging.DEBUG, filename='test.log')
logging.info('program start')
a = PreProcessor("./index/FX_EURKRW.csv")
a.start()
q = [abs(i) for i in a.delta]
z = sum(q)/len(q)
v = 0
for i in q:
    v+=(i-z)**2
v/=(len(q))
import math
print(math.sqrt(v))
x, y = a.get_test()
print(len(y))
a0, a1, a2 = 0, 0, 0
for i in y:
    if i[0] >i[1]:
        a1 += 1
    if i[0] < i[1]:
        a0 += 1
print(a0, a1, a2)
logging.info('program stopped')
