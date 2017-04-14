import cv2
import numpy as np
import os
import sys

def process(imgpath, outpath):
    img = cv2.imread(imgpath )
    h,w,c = img.shape # (768, 1024, 3)

    noise = np.random.randint(0,50,(h, w)) # design jitter/noise here
    zitter = np.zeros_like(img)
    zitter[:,:,1] = noise  

    noise_added = cv2.add(img, zitter)
    img1 = np.vstack((img[:h/2,:,:], noise_added[h/2:,:,:]))
    
    img2 = np.dstack( (
        np.roll(img1[:,:,0], 10, axis=0), 
        np.roll(img1[:,:,0], 10, axis=1), 
        np.roll(img1[:,:,0], -10, axis=0)
    ))
    img3 = cv2.flip( img2, np.random.randint(2))

    cv2.imwrite(outpath,img3)

if __name__ == "__main__":
    rootdir = sys.argv[1]
    n = int(sys.argv[2])
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            infile = os.path.join(subdir, file)
            for i in range(n):
                outfile = os.path.join(subdir, "aug_%d_%s"%(i, file))
                process(infile, outfile)
