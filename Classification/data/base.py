from array import array
from torch.utils.data import Dataset
import csv
from decimal import Decimal
import numpy as np


class CSVPaths(Dataset):
    def __init__(self, paths, copy=True):
        self.paths = paths
        self.e = dict()
        self.copy = copy
        self.slice_start = 0
        self.slice_end = 70
        data, result = self.preprocess_data()
        self.e["data"] = data
        self.e["result"] = result
        self._length = len(data)
    def __len__(self):
        return self._length

    def preprocess_data(self):
        with open(self.paths, 'r') as f:
            f_csv = csv.reader(f)
            data = []
            pred_tmp = []
            for idx, row in enumerate(f_csv):
                if idx > 0:
                    row = list(float(value) for value in row)
                    result = row[-1]
                    data_tmp = row[0:-1]
                    row = []
                    if self.copy:
                        for _ in range(24):
                            row += data_tmp 
                        row += data_tmp[0:8]
                    else:
                        row += data_tmp
                    data += [row]
                    pred_tmp += [int(result)]
            f.close()
        data = np.array(data)
        pred_tmp = np.array(pred_tmp)
        avg_data = np.average(data, axis=0)
        std_data = np.std(data, axis=0)
        data = (data - avg_data) / std_data
        pred = np.zeros([len(pred_tmp), 2])
        for i in range(len(pred_tmp)):
            pred[i, pred_tmp[i]] = 1.0
        input_data = data[self.slice_start:self.slice_end, :]
        input_pred = pred[self.slice_start:self.slice_end, :]
            
        return input_data, input_pred
            
    def __getitem__(self, i):
        item = dict()
        item["data"] = np.array(self.e["data"][i]).astype(np.float32)
        item["result"] = np.array(self.e["result"][i]).astype(np.float32)
        return item
    
class CSVPathsT(CSVPaths):
    def __init__(self, paths, copy):
        super().__init__(paths, copy)
        self.slice_start = 70
        self.slice_end = 100