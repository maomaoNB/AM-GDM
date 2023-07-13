class Father():
    def __init__(self):
        self.copy = 23
    
    def __print__(self):
        print(self.copy)
        
class Son(Father):
    def __init__(self):
        super().__init__()
        self.copy = 1

if __name__ == "__main__":
    s = Son()
    s.__print__()
