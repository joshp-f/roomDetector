from collections import defaultdict
import json

class memoryCache:
    def __init__(self):
        self.fn = 'mcache.JSON'
        init = {}
        try:
            file = open(self.fn, 'r')
            text = file.read()
            if text:
                init = json.loads(text)
            if not init: init = {}
            file.close()
        except IOError:
            pass
        self.d = defaultdict(lambda : defaultdict(int))
        for k in init:
            for k2 in init[k]:
                self.d[k][k2] = init[k][k2]
    def runPredictor(self,o,label=None):
        d = self.d
        top = None
        if d and d[o]:
            top = max([i for i in d[o].items()],key= lambda p1:p1[1]  )[0]
        # only if reinforcing
        if label and top != o:
            pass
        if label: d[o][label] += 1
        return top

    def saveCache(self):
        d = self.d

        file = open(self.fn, 'w')
        finald = {}
        for k in d:
            finald[k] = dict([(j, v) for (j,v) in d[k].items()])
        file.write(json.dumps(finald))