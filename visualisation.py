from Parsers.csv_parser_FX import Parser
import matplotlib.pyplot as plt
import datetime

a = Parser("./index/FX_USDKRW.csv", )
a.open()
q = a.get_data()
x = []
for i in range(len(q)-7):
    x.append(q[i+7]-q[i])

plt.plot(a.get_dates()[0:-7], x)
plt.plot([datetime.datetime(2015,9,4,0,0),datetime.datetime(2015,9,4,0,0)],[-200,200])
plt.plot([datetime.datetime(2016,1,29,0,0),datetime.datetime(2016,1,29,0,0)],[-200,200])
plt.show()
