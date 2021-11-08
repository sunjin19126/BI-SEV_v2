import numpy as np
from collections import Counter

def EAAC(frags):

    eaacs = []
    for frag in frags:
        eaac = []
        for i in range(24):
            count = Counter(frag[i:i+8])
            if 20 in count:
                count.pop(20)
            sums = sum(count.values()) + 1e-6
            aac = [count[i]/sums for i in range(20)]
            eaac += aac
        eaacs.append(eaac)

    return np.array(eaacs)

def EAAC2d(frags):

    eaacs = []
    for frag in frags:
        eaac = []
        for i in range(24):
            count = Counter(frag[i:i+8])
            if 20 in count:
                count.pop(20)
            sums = sum(count.values()) + 1e-6
            aac = [count[i] / sums for i in range(20)]
            eaac.append(aac)
        eaacs.append(eaac)

    return np.array(eaacs)

def Index(frags):

    return np.array(frags)