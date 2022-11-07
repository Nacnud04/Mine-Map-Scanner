from multiprocessing import Value
from weakref import ref
import cv2
from math import *
from PreProcessing import *
import numpy as np
import os

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

def grabSegment(folder, segment):
    fullname = f'{folder}{segment[0]}-{segment[1]}.jpg'
    image = cv2.imread(fullname)
    return image

def grabSegmentContour(folder, num):
    fullname = f'{folder}{num}.jpg'
    image = cv2.imread(fullname)
    return image

# combine and interpret the data
def transformDataGrid(data, folder, ppdh, ppdw):
    print(f'\nDetected {len(data)} words.')
    transformed = []
    for datapoint in data:
        try :
            text, confidence, location, theta, x, y, w, h = datapoint
            
            # import the image dimensions for each text localization
            img = grabSegment(folder, location)

            # calculate offset for angle

            (imageheight, imagewidth) = img.shape[:2]
            (cX, cY) = (imagewidth // 2, imageheight // 2)

            x0, y0 = x - cX, y - cY
            x1, y1 = x + w - cX, y - cY
            x2, y2 = x - cX, y + h - cY
            x3, y3 = x + w - cX, y + h - cY

            revtheta = radians(theta)

            x0, y0 = int(x0*cos(revtheta)-y0*sin(revtheta) + cX), int(x0*sin(revtheta)+y0*cos(revtheta) + cY)
            x1, y1 = int(x1*cos(revtheta)-y1*sin(revtheta) + cX), int(x1*sin(revtheta)+y1*cos(revtheta) + cY)
            x2, y2 = int(x2*cos(revtheta)-y2*sin(revtheta) + cX), int(x2*sin(revtheta)+y2*cos(revtheta) + cY)
            x3, y3 = int(x3*cos(revtheta)-y3*sin(revtheta) + cX), int(x3*sin(revtheta)+y3*cos(revtheta) + cY)
        
            # calculate offset for the image segment
            h, w = location
            xoffset, yoffset = int(w * ppdw), int(h * ppdh)
            x0, x1, x2, x3 = xoffset+x0, xoffset+x1, xoffset+x2, xoffset+x3
            y0, y1, y2, y3 = yoffset+y0, yoffset+y1, yoffset+y2, yoffset+y3

            transformed.append((text, (x0, y0), (x1, y1), (x2, y2), (x3, y3), confidence))

        except ValueError:
            break

    return transformed

# combine and interpret the data
def transformDataContoured(data, folder, bordersize = 0.25):
    print(f'\nDetected {len(data)} words.')
    transformed = []
    for datapoint in data:
        try :
            text, confidence, location, theta, x, y, w, h, xoffset, yoffset = datapoint

            # import the image dimensions for each text localization
            img = grabSegmentContour(folder, location)

            # calculate offset for angle

            (imageheight, imagewidth) = img.shape[:2]
            (cX, cY) = (imagewidth // 2, imageheight // 2)

            x0, y0 = x - cX, y - cY
            x1, y1 = x + w - cX, y - cY
            x2, y2 = x - cX, y + h - cY
            x3, y3 = x + w - cX, y + h - cY

            revtheta = radians(theta)

            x0, y0 = int(x0*cos(revtheta)-y0*sin(revtheta) + cX), int(x0*sin(revtheta)+y0*cos(revtheta) + cY)
            x1, y1 = int(x1*cos(revtheta)-y1*sin(revtheta) + cX), int(x1*sin(revtheta)+y1*cos(revtheta) + cY)
            x2, y2 = int(x2*cos(revtheta)-y2*sin(revtheta) + cX), int(x2*sin(revtheta)+y2*cos(revtheta) + cY)
            x3, y3 = int(x3*cos(revtheta)-y3*sin(revtheta) + cX), int(x3*sin(revtheta)+y3*cos(revtheta) + cY)
        
            # calculate offset from borders of segments
            dx = int((imagewidth / 2) - (imagewidth / (2*(bordersize+1))))
            dy = int((imageheight / 2) - (imageheight / (2*(bordersize+1))))
            
            # calculate offset for the image segment
            x0, x1, x2, x3 = int(xoffset+x0 - dx), int(xoffset+x1 - dx), int(xoffset+x2 - dx), int(xoffset+x3 - dx)
            y0, y1, y2, y3 = int(yoffset+y0 - dy), int(yoffset+y1 - dy), int(yoffset+y2 - dy), int(yoffset+y3 - dy)

            transformed.append((text, (x0, y0), (x1, y1), (x2, y2), (x3, y3), confidence))

        except ValueError:
            break

    return transformed

def orderpoints(pointlist):
    endpointlist = [(0, 0), (0, 0), (0, 0), (0, 0)]
    prepoint0, prepoint1, prepoint2, prepoint3 = pointlist
    xlist = np.array([prepoint0[0], prepoint1[0], prepoint2[0], prepoint3[0]])
    ylist = np.array([prepoint0[1], prepoint1[1], prepoint2[1], prepoint3[1]])
    xavg, yavg = np.average(xlist), np.average(ylist)
    for i in range(len(pointlist)):
        x, y = pointlist[i]
        newxlist = xlist[xlist != x]
        newylist = ylist[ylist != y]
        newxavg, newyavg = np.average(newxlist), np.average(newylist)
        # if the new average went down, then the value was in the greater half.
        if newxavg >= xavg and newyavg >= yavg:
            endpointlist[0] = pointlist[i]
        elif newxavg >= xavg and newyavg <= yavg:
            endpointlist[1] = pointlist[i]
        elif newxavg <= xavg and newyavg >= yavg:
            endpointlist[2] = pointlist[i]
        elif newxavg <= xavg and newyavg <= yavg:
            endpointlist[3] = pointlist[i]
    return endpointlist

# merge datapoints that have the same full name
def mergeFullName(data):
    ## DOES GOOFY THINGS
    namescovered = []
    reformated = []
    maximumdist = 60 #maximum distance apart where two boxes can be merged
    for datapoint in data:
        numsame = 1
        text, point0, point1, point2, point3, confidence = datapoint
        x0, y0 = point0
        x1, y1 = point1
        x2, y2 = point2
        x3, y3 = point3
        for i in range(len(data)):
            comptext, comppoint0, comppoint1, comppoint2, comppoint3, compconfidence = data[i]
            # get distance between boxes:
            dist = sqrt((comppoint0[0]-x0)**2+(comppoint0[1]-y0)**2)
            if text == comptext and dist <= maximumdist:
                numsame += 1
                cx0, cy0 = comppoint0
                cx1, cy1 = comppoint1
                cx2, cy2 = comppoint2
                cx3, cy3 = comppoint3
                x0, x1, x2, x3 = x0+cx0, x1+cx1, x2+cx2, x3+cx3
                y0, y1, y2, y3 = y0+cy0, y1+cy1, y2+cy2, y3+cy3
        x0, x1, x2, x3 = int(x0/numsame), int(x1/numsame), int(x2/numsame), int(x3/numsame)
        y0, y1, y2, y3 = int(y0/numsame), int(y1/numsame), int(y2/numsame), int(y3/numsame)
        if text not in namescovered:
            reformated.append((text, (x0, y0), (x1, y1), (x2, y2), (x3, y3)))
        namescovered.append(text)
    return reformated

# merge datapoints who's bounding boxes intersect
from shapely import geometry, ops
def mergeIntersecting(data, requirement):
    toberemoved = []
    intsecrequirement = float(requirement)
    for datapoint in data:
        text, point0, point1, point2, point3, confidence = datapoint
        polygon1 = geometry.Polygon([point0, point1, point2, point3, point1])
        for i in range(len(data)):
            comptext, comppoint0, comppoint1, comppoint2, comppoint3, compconfidence = data[i]
            polygon2 = geometry.Polygon([comppoint0, comppoint1, comppoint2, comppoint3, comppoint1])
            # find intersecting area
            areaofint = polygon1.intersection(polygon2).area
            # find % of intersecting area
            percentintersect = areaofint / (polygon1.area+polygon2.area-areaofint)
            if intsecrequirement <= percentintersect and polygon1.area != polygon2.area:
                if polygon2.area <= polygon1.area:
                    toberemoved.append(i)
                else:
                    res_list = [i for i, value in enumerate(data) if value == datapoint]
                    toberemoved.append(res_list[0])
                print(f'{text}|{comptext}|{percentintersect}')
    # purge items to be removed
    reformated = [v for i, v in enumerate(data) if i not in toberemoved]
    return reformated

def cleanupGrid(folder, maxi, maxj):
    for i in range(maxi+1):
        for j in range(maxj+1):
            filename = f'{folder}{i}-{j}.jpg'
            if os.path.exists(filename):
                os.remove(filename)
            else:
                print(f"File {filename} has gone missing and therfore cannot be deleted.") 

def cleanupContour(folder, maxi):
    for i in range(maxi+1):
        filename = f'{folder}{i}.jpg'
        if os.path.exists(filename):
            os.remove(filename)
        else:
            print(f"File {filename} has gone missing and therfore cannot be deleted.") 


def exportData(folder, filename, data):
    import csv
    with open(f'{folder}{filename}', 'w', newline='') as file:
        writer = csv.writer(file)
        for dp in data:
            writer.writerow([dp[0], dp[1], dp[2], dp[3], dp[4], dp[5]])
    print(f'Exported data to: {folder}{filename}')

def importData(folder, filename):
    import csv
    data = []
    with open(f'{folder}{filename}', 'r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            row[1], row[2], row[3], row[4] = row[1].replace('(', ''), row[2].replace('(', ''), row[3].replace('(', ''), row[4].replace('(', '')
            row[1], row[2], row[3], row[4] = row[1].replace(')', ''), row[2].replace(')', ''), row[3].replace(')', ''), row[4].replace(')', '')
            point0 = tuple(map(int, row[1].split(', ')))
            point1 = tuple(map(int, row[2].split(', ')))
            point2 = tuple(map(int, row[3].split(', ')))
            point3 = tuple(map(int, row[4].split(', ')))
            datapoint = (row[0], point0, point1, point2, point3, row[5])
            data.append(datapoint)
    print(f'Imported data from: {folder}{filename}')
    return data

def showData(data, img, outputheight, thickness):
    for datapoint in data:
        text, point0, point1, point2, point3, confidence = datapoint
        cv2.line(img,
                    point0,
                    point1,
                    (0, 0, 255), thickness)
        cv2.line(img,
                    point1,
                    point3,
                    (0, 0, 255), thickness)
        cv2.line(img,
                    point3,
                    point2,
                    (0, 0, 255), thickness)
        cv2.line(img,
                    point2,
                    point0,
                    (0, 0, 255), thickness)
    imagesscaled = ResizeWithAspectRatio(img, height=outputheight)
    cv2.imshow("Image", imagesscaled)
    cv2.waitKey(0)

def exportImage(data, img, thickness, pathname):
    for datapoint in data:
        text, point0, point1, point2, point3, confidence = datapoint
        cv2.line(img,
                    point0,
                    point1,
                    (0, 0, 255), thickness)
        cv2.line(img,
                    point1,
                    point3,
                    (0, 0, 255), thickness)
        cv2.line(img,
                    point3,
                    point2,
                    (0, 0, 255), thickness)
        cv2.line(img,
                    point2,
                    point0,
                    (0, 0, 255), thickness)
    cv2.imwrite(f"{pathname}", img)

def exportImageWords(data, img, thickness, pathname):
    for datapoint in data:
        text, point0, point1, point2, point3, confidence = datapoint
        cv2.line(img,
                    point0,
                    point1,
                    (0, 0, 255), thickness)
        cv2.line(img,
                    point1,
                    point3,
                    (0, 0, 255), thickness)
        cv2.line(img,
                    point3,
                    point2,
                    (0, 0, 255), thickness)
        cv2.line(img,
                    point2,
                    point0,
                    (0, 0, 255), thickness)
        cv2.putText(img, 
                    text, point0, 3,
                    fontScale=5, color=(255, 0, 0), thickness=thickness)
    cv2.imwrite(f"{pathname}", img)