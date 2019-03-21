from Parsers.csv_parser_FX import Parser
import matplotlib.pyplot as plt
import datetime

def f(delta):
    g=8
    if delta < -g:
        return 0
    elif -g <= delta <= g:
        return 1
    elif g < delta:
        return 2

a = Parser("./index/FX_EURKRW.csv", )
a.open()
q = a.get_data()
x = []
plt.plot(a.get_dates(), q)
plt.plot([datetime.datetime(2015,9,4,0,0),datetime.datetime(2015,9,4,0,0)],[1100,2000])
plt.plot([datetime.datetime(2017,1,1,0,0),datetime.datetime(2017,1,1,0,0)],[1100,2000])
plt.show()
