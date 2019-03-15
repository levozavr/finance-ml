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

a = Parser("./index/FX_USDKRW.csv", )
a.open()
q = a.get_data()
x = []
for i in range(len(q)-7):
    x.append(f(q[i+7]-q[i]))
for j, t in enumerate(x):
    plt.plot(a.get_dates()[j], t, color='green', marker='.')
plt.plot([datetime.datetime(2015,9,4,0,0),datetime.datetime(2015,9,4,0,0)],[0,2])
plt.plot([datetime.datetime(2016,1,29,0,0),datetime.datetime(2016,1,29,0,0)],[0,2])
plt.show()
