class Oya():
    def __init__(self):
        self.a = None

    def read(self):
        self.a = [2]

class Ko(Oya):
    def __init__(self):
        self.a = []

class koko(Ko):
    def __init__(self):
        None
    
    def pri(self):
        print(self.a)


kobun = koko()
kobun.read()

kobun.pri()
