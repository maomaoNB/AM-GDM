from torch.utils.data import Dataset
from data.base import CSVPaths, CSVPathsT

class CustomBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None

    def __len__(self):
        return self.data._length

    def __getitem__(self, i):
        example = self.data[i]
        return example


class CustomTrain(CustomBase):
    def __init__(self, training_file):
        super().__init__()
        self.data = CSVPaths(paths=training_file, copy=True)


class CustomTest(CustomBase):
    def __init__(self, test_file):
        super().__init__()
        self.data = CSVPathsT(paths=test_file, copy=True)
        
class TransformerTrain(CustomBase):
    def __init__(self, training_file):
        super().__init__(training_file)
        self.data = CSVPaths(paths=training_file, copy=False)
        
class TransformerTest(CustomBase):
    def __init__(self, test_file):
        super().__init__()
        self.data = CSVPathsT(paths=test_file, copy=False)
