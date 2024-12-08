from everyai.classfier.classfier import TextClassifer

class Explanation:
    def __init__(self,classfier:TextClassifer):
        self.classfier = TextClassifer()

    def exlain(self,text:str):
        return self.classfier.predict(text)