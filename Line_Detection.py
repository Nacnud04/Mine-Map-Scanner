import cv2
import numpy as np
import os
from pylsd import lsd
import random
from math import *
from matplotlib import pyplot as plt

full_name = 'F:/G4313_G5H1_svar_M58_02.jpg'
folder, img_name = os.path.split(full_name)
img = cv2.imread(full_name, cv2.IMREAD_COLOR)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

segments = lsd(img_gray, scale=0.5, ang_th=20)

print(segments.shape)

characteristics = np.zeros(shape=(segments.shape[0], 5))

angles = []

for i in range(segments.shape[0]):
    pt1 = (int(segments[i, 0]), int(segments[i, 1]))
    pt2 = (int(segments[i, 2]), int(segments[i, 3]))

    # calculate length of segment
    length = sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

    # calculate angle of segment
    try:
        angle = atan((pt2[1] - pt1[1]) / (pt2[0] - pt1[0]))
    except ZeroDivisionError:
        angle = pi / 2
        continue

    # this is just here to display the min and max angle
    angles.append(degrees(angle))

    characteristics[i] = [length, angle, 0, 0, 0]


segments = np.append(segments, characteristics, axis = 1)

print('1')
minlength = 40

badsegments = []

for i in range(segments.shape[0]):
    pt1 = (int(segments[i, 0]), int(segments[i, 1]))
    pt2 = (int(segments[i, 2]), int(segments[i, 3]))
    width, length = segments[i, 4], segments[i, 5]

    # filters
    # line length

    if length <= minlength:
        badsegments.append(i)

# segment grouping
def similarAngle(array, angle, bound):
    results = []
    degangle = degrees(angle)
    degbound = bound
    for i in range(array.shape[0]):
        lineangle = segments[i, 6]
        if degrees(lineangle) <= degangle + degbound and degrees(lineangle) >= degangle - degbound:
            results.append(i)
    return results

def angleGrouper(segments, angle, bound):
    segmentscopy = segments
    previous = []
    centerangles = []
    anglegroups = []
    for i in range(segmentscopy.shape[0]):
        anglegroups.append([])
        angle = segmentscopy[i, 6]
        results = similarAngle(segments, angle, bound)
        red, green, blue = random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255)
        j = 0
        for result in results:
            if result not in previous:
                if j == 0:
                    centerangles.append(degrees(angle))
                segments[result, 7], segments[results, 8], segments[results, 9]= blue, green, red
                anglegroups[i].append(result)
                j += 1
        previous.extend(results)
    return segments, centerangles, anglegroups

def showAngles(segments, centerangles):
    x, y = [], []
    for i in range(segments.shape[0]):
        degangle = int(degrees(segments[i, 6]))
        if degangle not in x:
            x.append(degangle)
            y.append(0)
    for i in range(max(x)-min(x)):
        if i not in x:
            x.append(i)
            y.append(0)
    # order data
    x.sort()
    print(f'Angles Min: {min(angles)}  |  Angles Max: {max(angles)}')
    for i in range(segments.shape[0]):
        degangle = int(degrees(segments[i, 6]))
        index = x.index(degangle)
        y[index] += 1

    for angle in centerangles:
        xangle = [angle, angle]
        yangle = [0, max(y)]
        plt.plot(xangle, yangle, linestyle='dashdot', color='lightgray', linewidth='1')
    plt.plot(x, y)
    plt.xlabel("Angle")
    plt.ylabel("Count")
    plt.title("Angle frequency")
    plt.show()

# for every line in a group compare it with every other line in that same group
# if the line is not closer than a set distance then dont include the line in the homogenization
def homogenizeLines(segments, anglegroups):
    lines = []
    distances = []
    for group in anglegroups:
        biggestdistance = 0
        '''
        for lineid in group:
            angle = radians(segments[lineid, 6])
            cX, cY = (int(segments[lineid, 0]), int(segments[lineid, 1]))
            complete = False

            for otherlines in group:
                if otherlines != lineid:
                    x0, y0 = (int(segments[otherlines, 0]), int(segments[otherlines, 1]))
                    x1, y1 = (int(segments[otherlines, 2]), int(segments[otherlines, 3]))
                    x0, y0 = int(x0*cos(angle)-y0*sin(angle) + cX), int(x0*sin(angle)+y0*cos(angle) + cY)
                    x1, y1 = int(x1*cos(angle)-y1*sin(angle) + cX), int(x1*sin(angle)+y1*cos(angle) + cY)
                    if abs(y0) <= 30 or abs(y1) <= 30:
                        complete = True
                    if abs(y0) >= biggestdistance:
                        biggestdistance = abs(y0)
                    elif abs(y1) >= biggestdistance:
                        biggestdistance = abs(y1)

                if complete == True:
                    break

            if complete == False:
                index = group.index(lineid)
                group.pop(index)

        distances.append(biggestdistance)
        '''

        # now blend the lines

        # look for the closest and furthest point from the origin
        # this needs to be improved
        i = 0
        totm = 0
        totb = 0
        for lineid in group:
            pt1 = (int(segments[lineid, 0]), int(segments[lineid, 1]))
            pt2 = (int(segments[lineid, 2]), int(segments[lineid, 3]))
            #cv2.line(img, pt1, pt2, (0,0,0), int(np.ceil(width / 4)))
            m = (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])
            b = pt2[1] - m * pt2[0]
            totm += m
            totb += b
            i += 1
            print(i)
        if i == 0:
            lines.append([(0, 0), (0, 0)])
            break
        finm = totm / i
        finb = totb / i
        # y = mx + b
        # calculate intersections with borders of image to make a line
        y = int(finb)
        x = int((-1 * finb) / finm)
        lines.append([(0, y), (x, 0)])

    return lines, distances

def showOldLines():
    for i in range(segments.shape[0]):
        if i not in badsegments:
            pt1 = (int(segments[i, 0]), int(segments[i, 1]))
            pt2 = (int(segments[i, 2]), int(segments[i, 3]))
            width, length, angle = segments[i, 4], segments[i, 5], segments[i, 6]
            cv2.line(img, pt1, pt2, (segments[i, 7], segments[i, 8], segments[i, 9]), int(np.ceil(width / 4)))
            #cv2.line(img, pt1, pt2, (122, 146, 164), int(np.ceil(width / 1)))

def showNewLines(lines, distances):
    for i in range(len(lines)):
        closest, furthest = lines[i][0], lines[i][1]
        cv2.line(img, closest, furthest, (random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255)), 5)

def hideLines():
    imgr = img[:, :, :1]
    imgg = img[:, :, 1:2]
    imgb = img[:, :, 2:]
    red, green, blue = np.average(imgr), np.average(imgg), np.average(imgb)
    for i in range(segments.shape[0]):
        if i not in badsegments:
            pt1 = (int(segments[i, 0]), int(segments[i, 1]))
            pt2 = (int(segments[i, 2]), int(segments[i, 3]))
            width, length, angle = segments[i, 4], segments[i, 5], segments[i, 6]
            if length >= 80:
                # prevents lines in text from being removed.
                cv2.line(img, pt1, pt2, (red, green, blue), int(np.ceil(width)))

segments, centerangles, anglegroups = angleGrouper(segments, 0, 2.5)
showAngles(segments, centerangles)
#lines, distances = homogenizeLines(segments, anglegroups)
#showOldLines()
#showNewLines(lines, distances)
print('2')
hideLines()

# rescale image to fit screen
def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

imagescaled = ResizeWithAspectRatio(img, height=500)

cv2.imshow("Output!", imagescaled)
cv2.waitKey(0)