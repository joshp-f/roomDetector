from collections import defaultdict
import json, math

class memoryCache:
    def __init__(self):
        self.fn = 'mcache.JSON'
        init = {}
        try:
            file = open(self.fn, 'r')
            text = file.read()
            if text:
                init = json.loads(text)
            else: init = {}
            file.close()
        except IOError:
            pass
        self.d = defaultdict(lambda : defaultdict(int))
        self.state = defaultdict(float)
        for k in init:
            for k2 in init[k]:
                self.d[k][k2] = init[k][k2]
    def propogate(self,features): #gets concept activations
        self.state = defaultdict(float)
        for f in features:
          items = self.d[f]
          total = sum(items.values())
          normalized = dict([(k, math.pow(v/total,5)) for (k,v) in items.items()]) # gives very weak impact to mixed prob sights
          for k in normalized: self.state[k] += normalized[k]

    def reinforce(self, features):
        for f1 in features:
            for f2 in features:
                self.d[f1][f2] += 1.0 
    def getHighest(self, items): #pulls the highest activation from items in the last propagation
        activations = [(name,self.state[name]) for name in items]
        total = max(0.0000000000001,sum([v for (k,v) in activations]))
        top = max([i for i in activations],key= lambda p1:p1[1]  )[0]
        score = self.state[top]
        # only if reinforcing
        return top, score/total
    def saveCache(self):
        d = self.d

        file = open(self.fn, 'w')
        finald = {}
        for k in d:
            finald[k] = dict([(j, v) for (j,v) in d[k].items()])
        file.write(json.dumps(finald))
