import cv2
import numpy as np

imgSize = open('imgSize.txt', "r")
row = int(imgSize.readline())
col = int(imgSize.readline())
ch = int(imgSize.readline())
flattenSize = row * col * ch

tupleIt = 0
charIt = 0

encodedTuples = np.load('./encodedTuples.npy')
encodedChars = np.load('./encodedChars.npy')

decodedRes = np.array([])

while tupleIt < encodedTuples.size:
    back = encodedTuples[tupleIt]
    current = encodedChars[charIt]
    matchLength = int(encodedTuples[tupleIt + 1])
    charIt += 1
    tupleIt += 2
    if back == 0:  # it's just a pixel
        decodedRes = np.append(decodedRes, current)
    else:  # it's a pattern
        tmpSize = decodedRes.size
        for i in range(matchLength):
            limit = int(tmpSize - back + i)
            if(limit < decodedRes.size):
                decodedRes = np.append(decodedRes, decodedRes[limit])
        if decodedRes.size < flattenSize:
            decodedRes = np.append(decodedRes, current)

for i in range(decodedRes.size, flattenSize):
    decodedRes = np.append(decodedRes, 0)
decodedRes = np.reshape(decodedRes, (row, col, ch))
cv2.imshow('output', decodedRes)
cv2.imwrite('output.jpg', decodedRes)
