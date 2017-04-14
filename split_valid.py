import os
from glob import glob
import numpy as np
import sys

if __name__ == "__main__":
    n = int(sys.argv[1])
    os.chdir('data/')
    os.chdir('train')
    os.mkdir('../valid')

    g = glob('*')
    for d in g: os.mkdir('../valid/'+d)

    g = glob('*/*.jpg')
    shuf = np.random.permutation(g)
    for i in range(n): os.rename(shuf[i], '../valid/' + shuf[i])
