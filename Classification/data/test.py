import csv
from decimal import Decimal

# types = [float, float, float, float, float, float, float, float, float, float, float]
types = []
for _ in range(11):
    types.append(Decimal)
    
with open('/home/zjy/Workspace/lax/VGG/data/datasets/data.csv', 'r') as f:
    f_csv = csv.reader(f)
    tmp = []
    for row in f_csv:
        tmp += [row]
    f.close()
        
    tmp = tmp[1:len(tmp)]
    data = []
    for row in tmp:
        row = list(float(Decimal(value)*Decimal(50.0)) if Decimal(value) < Decimal(10.0) else float(value) for value in row)
        data += [row]
        
    x = 0