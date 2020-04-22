import re
import nltk
from collections import defaultdict

class Decoder():
    def __init__(self, hmm_model):
        self.tags = []
        self.transition = defaultdict(dict)
        self.emission = defaultdict(dict)
        self.load_model(hmm_model)    
    
    def load_model(self, hmm_model):
        f = open(hmm_model, "r")
        while True:
            line = f.readline()
            if not line: break
            
            l = line.rstrip().split()
            if len(l) == 0: continue
            if l[0] == "Tags:":
                self.tags = ["Begin"] + l[1:]
            
            if l[0] == "Transition":
                while True:
                    t = f.readline()
                    if t == '\n': break
                    t = t.rstrip().split()
                    self.transition[t[0]][t[2]] = float(t[4])
            
            if l[0] == "Emission":
                while True:
                    e = f.readline()
                    if e == '\n' or e == '': break
                    e = e.rstrip().split()
                    self.emission[e[1][1:-1].split('|')[0].lower()][e[1][1:-1].split('|')[1]] = float(e[3])
            
    def tranp(self, tag1, tag2):
        try: return self.transition[tag1][tag2]/sum(self.transition[tag1].values())
        except: return 0

    def emisp(self, word, tag):
        try: return self.emission[word][tag]/sum(self.emission[word].values())
        except: return 0

    def decode_viterbi(self, sentence):
        words = nltk.tokenize.word_tokenize(sentence)
        for word in words: word = word.lower()
        m = defaultdict(dict)
        
        for i in range(len(words)):
            word = words[i].lower()
            if i == 0:
                for tag in self.tags:
                    m[i][tag] = ['Begin', self.tranp('Begin', tag)]
                    if word in self.emission.keys():
                        m[i][tag][1] *= self.emisp(word, tag)
            else:
                for tag in self.tags:
                    vals = []
                    for prev_tag in m[i-1].keys():
                        vals.append(m[i-1][prev_tag][1] * self.tranp(prev_tag, tag))
                        if word in self.emission.keys():
                            vals[-1] *= self.emisp(word, tag)
                    m[i][tag] = [list(m[i-1].keys())[vals.index(max(vals))], max(vals)]

        prediction = [max(m[len(words)-1], key=m[len(words)-1].get)]
        for i in range(len(words)-1):
            prediction.append(m[len(words)-i-1][prediction[-1]][0])
        
        return list(reversed(prediction))

if __name__ == "__main__":
    sentence = input()
    
    d = Decoder("./hmmmodel.txt")
    print(*d.decode_viterbi(sentence))
