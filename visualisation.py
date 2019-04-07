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
fig, ax = plt.subplots()
ax.set_xlabel('Временные отчеты')
ax.set_ylabel('Значение индекса')
ax.plot(q, label="index2")
ax.plot([2835,2835],[800,1600])
ax.plot([3163,3163],[800,1600])
ax.legend(loc='upper left')
plt.show()

