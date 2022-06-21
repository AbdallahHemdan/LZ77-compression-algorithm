import numpy as np
import cv2

inputImg = cv2.imread('input.jpg')
flat = np.array(inputImg).flatten()
cv2.imshow('input', inputImg)

# size of the image(row, col)
row = inputImg.shape[0]
col = inputImg.shape[1]
# number of image channels
ch = inputImg.shape[2]
flattenSize = row * col * ch

# get sliding window and look ahead window sizes
slidingWindow = int(input('1. Enter the sliding window size: '))
lookAhead = int(input('2. Enter the look ahead window size: '))


# if look ahead size is too large (larger than the sliding window size)
# it will be equal to the sliding window size
if lookAhead > slidingWindow:
    lookAhead = slidingWindow


# <go-back, matching-size, nxt-pixel-code>
encodedTuple = np.array([], dtype=np.uint16)
encodedChar = np.array([], dtype=np.uint8)

# get the length of the search buffer from (slidingWindow length and lookAhead length)
sbLength = slidingWindow - lookAhead
for it in range(sbLength):
    encodedTuple = np.append(encodedTuple, (0, 0))
    encodedChar = np.append(encodedChar, flat[it])

# initialize the pointer of the search buffer to 0
sbLeft = 0

# process the encoding
while sbLeft + sbLength < flattenSize:
    mxMatch = 0
    mxBack = 0
    sbRight = sbLeft + sbLength
    current = flat[sbRight]  # current pixel to encode
    frq = np.array([], dtype=np.int16)

    for i in range(sbLeft, sbLeft + sbLength):
        if (flat[i] == current):  # there is a match
            frq = np.append(frq, i)

    # there is no match for the current pixel
    if (frq.size == 0):
        encodedChar = np.append(encodedChar, current)
        encodedTuple = np.append(encodedTuple, (0, 0))
    # there is a match
    else:
        for matchIndex in frq:
            curMatch = 0
            it = 0
            back = sbRight - matchIndex
            while sbRight + it < flattenSize:  # we still in the range
                if flat[it + matchIndex] == flat[sbRight + it]:
                    curMatch += 1
                    it += 1
                else:  # once there is no match => exit
                    break
            # maximize the match of the current pixel
            if curMatch > mxMatch:
                mxMatch = curMatch
                mxBack = back
        encodedTuple = np.append(encodedTuple, (mxBack, mxMatch))
        encodedChar = np.append(encodedChar, flat[sbRight + mxMatch - 1])

    # update the sliding window
    sbLeft += 1 + mxMatch

print('total size', encodedTuple.size)
print('total size', encodedChar.size)

np.save('encodedTuples', encodedTuple)
np.save('encodedChars', encodedChar)

imgSize = open('imgSize.txt', "w")
imgSize.write(str(row) + '\n')  # write row dimension
imgSize.write(str(col) + '\n')  # write col dimension
imgSize.write(str(ch) + '\n')   # write channel dimension

cv2.waitKey(0)
cv2.destroyAllWindows()
