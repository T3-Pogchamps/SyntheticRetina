import os
from PIL import Image

def getMinute(img):
    firstSlice = img[img.index("_")+1:]
    minute = int(firstSlice[:firstSlice.index("_")])
    secondSlice = firstSlice[firstSlice.index("_")+1:]
    return int(secondSlice[0:2]) + minute * 60

currentPath = os.getcwd()
seqPath = currentPath[:-4] + "data/sample_seq"
outFolPath = currentPath[:-4] + "data/sample_filtered_seq"
seqList = os.listdir(seqPath)
sequenceSize = 6
count = 0

for seq in seqList:
    path = seqPath + "/" + seq
    imgList = sorted(os.listdir(path), key=lambda x: int(x[0:x.index("_")]))
    i = 0
    imgCount = 0
    start = 0
    while i < len(imgList) - 1 and imgCount < sequenceSize:
        if getMinute(imgList[i]) + 3 == getMinute(imgList[i + 1]):
            imgCount += 1
        else:
            imgCount = 0
            start = i
        i += 1
    if imgCount == 6:
        count += 1
        outSubPath = outFolPath + "/" + seq
        os.system("mkdir " + outSubPath)
        for i in range(6):
            os.system("cp " + path + "/" + imgList[start+i] + " " + outSubPath)
