from Parsers.csv_parser import CsvParser

a = CsvParser(filename="/index/avg_index.csv")
a.open()
print(a.get())
a.close()
