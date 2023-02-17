from datetime import datetime
import matplotlib.image as pltim
import matplotlib.pyplot as plt
import numpy as np
import copy
import time
import dijkstramin as dj
import showProgress as sp
import os  # For readFromDirectory()
import sys
from datetime import timedelta



def rgb2gray(image):
    print("\n\n rgb2gray process was started...")
    imageHeight = len(image)
    imageWidth = len(image[0])
    grayImage = np.zeros([imageHeight, imageWidth], dtype=np.uint8)

    for i in range(imageHeight):
        for j in range(imageWidth):
            grayImage[i][j] = int(image[i][j][0] * 0.2126 + image[i][j][1] * 0.7152 + image[i][j][2] * 0.0722)

        sp.showProgress(i, imageHeight)  # [Not-Important] This is just for interface
    return grayImage




def gray2rgb(image):
    imageHeight = len(image)
    imageWidth = len(image[0])
    rgbImage = np.zeros([imageHeight, imageWidth, 3], dtype=np.uint8)

    for i in range(imageHeight):
        for j in range(imageWidth):
            color = image[i][j]
            for k in range(3):
                rgbImage[i][j][k] = color

    return rgbImage




def laplacianTransform(image):
    print("\n\n laplacianTransform process was started...")
    imageHeight = len(image)
    imageWidth = len(image[0])
    laplacianImage = np.zeros([imageHeight, imageWidth], dtype=np.uint8)
    laplacianMask = [[1,1,1], [1,-8,1], [1,1,1]]
    min = 0
    max = 0

    for i in range(1, imageHeight-1):
        for j in range(1, imageWidth-1):
            laplacianSum = 0

            for k in range(len(laplacianMask)):
                for m in range(len(laplacianMask[0])):
                    currentLaplacian = laplacianMask[k][m]
                    applied = image[i+m-1][j+k-1] * currentLaplacian
                    laplacianSum += applied

            laplacianSum = abs(laplacianSum)
            if(i == 1 and j == 1):
                min = laplacianSum
                max = laplacianSum
            if(laplacianSum < min):
                min = laplacianSum
            if(laplacianSum > max):
                max = laplacianSum

            laplacianImage[i][j] = laplacianSum

            sp.showProgress(i, imageHeight)  # [Not-Important] This is just for interface

    for i in range(imageHeight):
        for j in range(imageWidth):
            laplacianImage[i][j] = int(((laplacianImage[i][j] - min) / (max - min)) * 255)

    return laplacianImage




def getEnergyOfUniquePixel(laplacianImage2D, iterator, pathFinder, usedPixels, processNumber, neighborType):  # Extra Information: If we give +infinite number as energy point,
    returningEnergyPoint, returningHelper = 0, 1                                                              #   the algorithm will never ever pick that point since it's looking
    usedPixelsTransposed = list(zip(*usedPixels))  # Transpose process - Rows to column, columns to row.
                                                                                                              #   for the lowest energ point. This method will use that trick.
    while True:


        if neighborType == "left":
            if pathFinder - returningHelper < 0:          # This mean that current path on the first pixel of a row.
                returningEnergyPoint = float('inf')       #   So there is no left neighbor since there is no left pixel.
                break
            else:
                if len(usedPixels) != 0:
                    if pathFinder - returningHelper in usedPixelsTransposed[iterator]:  # This pixel is used/marked/deleted/retargeted! Select next left
                        returningHelper += 1
                        continue
                    else:
                        returningEnergyPoint = laplacianImage2D[iterator][pathFinder-returningHelper]
                        break
                else:
                    returningEnergyPoint = laplacianImage2D[iterator][pathFinder - returningHelper]
                    break


        elif neighborType == "center":
            if len(usedPixels) != 0:
                if pathFinder in usedPixelsTransposed[iterator]:  # This pixel is used/marked/deleted/retargeted! So don't
                    returningEnergyPoint = float('inf')                  #   use the center neighbor
                    break
                else:
                    returningEnergyPoint = laplacianImage2D[iterator][pathFinder]
                    break
            else:
                returningEnergyPoint = laplacianImage2D[iterator][pathFinder]
                break


        elif neighborType == "right":
            if pathFinder + returningHelper > len(laplacianImage2D[0])-1:   # imageWidth-1 mean that current path on the last pixel of a row.
                returningEnergyPoint = float('inf')                         #   So there is no right neighbor since there is no right pixel
                break
            else:
                if len(usedPixels) != 0:
                    if pathFinder + returningHelper in usedPixelsTransposed[iterator]:  # This pixel is used/marked/deleted/retargeted! Select next right
                        returningHelper += 1
                        continue
                    else:
                        returningEnergyPoint = laplacianImage2D[iterator][pathFinder+returningHelper]
                        break
                else:
                    returningEnergyPoint = laplacianImage2D[iterator][pathFinder + returningHelper]
                    break


    return returningEnergyPoint, returningHelper




def DetermineRecalculationPixels(recalculationStartPixels, usedPixels, routeAndSumOfEnergyPoints, imageWidth):
    lastIndexOfUsedPixel = len(usedPixels) - 1  # This will be used for getting the last used/marked/deleted route

    if len(usedPixels) == 0:  # If no usedPixels, that means retargeting process just started. So all energy points must be calculated
        recalculationStartPixels = list(range(imageWidth))
    else:
        for j in range(1, len(usedPixels[0])):  # imageHeight
            for k in range(imageWidth):
                if usedPixels[ lastIndexOfUsedPixel ][j] == routeAndSumOfEnergyPoints[j][k]:
                    recalculationStartPixels.append(k)
        recalculationStartPixels = list(set(recalculationStartPixels))
        recalculationStartPixels.remove(usedPixels[lastIndexOfUsedPixel][0])

    return recalculationStartPixels




def EnergyPointsWithGreedySearch(laplacianImage2D, usedPixels, processNumber, routeAndSumOfEnergyPoints, energyPoints, path):
    imageHeight = len(laplacianImage2D)
    imageWidth = len(laplacianImage2D[0])

    currentEnergy1, currentEnergy2, currentEnergy3 = 0, 0, 0
    pathFinder = 0


    recalculationStartPixels = []  # This is for speed up the process: After a lowest energy path removed,
                                   #   algorithm won't calculate all the energy points of the image but
                                   #   will calculate sum of energy points for only if a route includes one of
                                   #   pixel of the removed/marked path
    recalculationStartPixels = DetermineRecalculationPixels(recalculationStartPixels, usedPixels, routeAndSumOfEnergyPoints, imageWidth)
    recalculationStartPixelsCounter = 0  # This is for user interface only
    #optimizedImageWidthList = list(range(imageWidth)) if len(recalculationStartPixels) == 0 else recalculationStartPixels
    # For the start, optimizedImageWidth will be equal to imageWidth since we want algorithm to calculate all the
    #   energy points from starting [0,0] to [0,imageWidth]. After that, we only need to re-calculate the path
    #   if there is a coordinates in the used/marked/deleted (usedPixels).
    #   This is for speed up the targeting process. Significantly decrease the targeting process time.

    for j in recalculationStartPixels:

        pathFinder = j
        path[0][j] = j
        currentEnergyFirst = laplacianImage2D[0][j]
        energyPoints[j] = energyPoints[j] + currentEnergyFirst

        for i in range(imageHeight-1):

            pathFinderHelperLeft, pathFinderHelperRight = 1, 1

            currentEnergyLeft, pathFinderHelperLeft = getEnergyOfUniquePixel(laplacianImage2D, i+1, pathFinder, usedPixels, processNumber, "left")
            currentEnergyCenter = getEnergyOfUniquePixel(laplacianImage2D, i+1, pathFinder, usedPixels, processNumber, "center")[0]
            currentEnergyRight, pathFinderHelperRight = getEnergyOfUniquePixel(laplacianImage2D, i+1, pathFinder, usedPixels, processNumber, "right")

            if ((currentEnergyCenter <= currentEnergyLeft) and
                (currentEnergyCenter <= currentEnergyRight)):            # If you have any possibilty(if all three neighbors are the same) to select
                path[i + 1][j] = pathFinder                              #   retargeting path on the same column (same pixel on next row), go directly
                energyPoints[j] = energyPoints[j] + currentEnergyCenter  #   to down instead of left or right and add that pixel to retargeting path
            elif ((currentEnergyLeft <= currentEnergyCenter) and
                  (currentEnergyLeft <= currentEnergyRight)):            # If left and right pixels' energies are the same and center's energy bigger than them,
                path[i + 1][j] = pathFinder - pathFinderHelperLeft       #   algorithm can select both left or right side (for the lowest energ point) since it's
                energyPoints[j] = energyPoints[j] + currentEnergyLeft    #   using greedy search. So I could use random but I pushed algorithm to select left
                pathFinder -= pathFinderHelperLeft                       #   instead of right; left is always better - saludos al comandante C.G.
            else:
                path[i + 1][j] = pathFinder + pathFinderHelperRight      # Only if right pixel's energy is smaller than left and center pixels energies,
                energyPoints[j] = energyPoints[j] + currentEnergyRight   #   then algorithm select right pixel as next path
                pathFinder += pathFinderHelperRight


        # sp.showProgressWithBars(recalculationStartPixelsCounter, len(recalculationStartPixels))  # [Not-Important] This is just for interface
        recalculationStartPixelsCounter += 1

    for t in range(imageWidth):
        path[imageHeight][t] = energyPoints[t]

    return path




def paintRedForGreedySearch(image, path, startPoint):
    imageHeight = len(image)
    paintingPx = 0

    image.setflags(write=1)

    for x in range(imageHeight):
        paintingPx = path[x][startPoint]
        image[x][paintingPx][0] = 255
        image[x][paintingPx][1] = 0
        image[x][paintingPx][2] = 0




def determineStartPoint(energyPoints, usedPixels):                     # Firstly, algorithm get the last row (the extra created row to write energy sums) of image
    sumOfEnergPoints = (energyPoints[len(energyPoints) - 1]).tolist()  #   and convert to Python array (which is list). Then find minimum value of the list
    for i in range(len(usedPixels)):
        sumOfEnergPoints[usedPixels[i][0]] = float('inf')

    return sumOfEnergPoints.index(min(sumOfEnergPoints))               #   (which is also minimum energy point). Then find the index (the first index from left if multiple same minimum value)
                                                                       #   of that minimum value and return it. "Index" also mean starting pixel for best retargeting process




def convert3Dto2D(image):
    imageHeight = len(image)
    imageWidth = len(image[0])
    image2D = np.zeros([imageHeight, imageWidth], dtype=np.uint8)

    for i in range(imageHeight):
        for j in range(imageWidth):
            image2D[i][j] = int(image[i][j][0])

    return image2D




def saveTheRoute(routes, startPoint, usedPixels, currentProcess, removingPixels):
    usedPixels.append([])
    for i in range(len(routes)-1):
        usedPixels[currentProcess].append(routes[i][startPoint])
        removingPixels[i][currentProcess] = routes[i][startPoint]
    return usedPixels, removingPixels




def removeMarkedPixels(originalImage, removingPixels):  # This method for removing marked points from image to retarget(resize) the image
    originalImageList = originalImage.tolist()
    imageHeight = len(originalImage)
    print("\nCompleting... Please wait...")

    for i in range(imageHeight):
        removingPixels[i] = sorted(removingPixels[i], reverse=True)
        for j in range(len(removingPixels[i])):
            del originalImageList[i][removingPixels[i][j]]

    originalImage = np.array(originalImageList)
    return originalImage




def retargetWithGreedy(packForGreedy):
    return EnergyPointsWithGreedySearch(
        packForGreedy["laplacianImage2D"],
        packForGreedy["usedPixels"],
        packForGreedy["processNumber"],
        packForGreedy["routeAndSumOfEnergyPoints"],
        packForGreedy["energyPoints"],
        packForGreedy["path"]
    )




def prepareForDijkstra(packForDijkstra):
    packForDijkstra["retargetedLaplacianImage2D"] = np.insert(packForDijkstra["retargetedLaplacianImage2D"], 0, [np.zeros([packForDijkstra["imageWidth"]], dtype=int)], axis=0)
    packForDijkstra["retargetedLaplacianImage2D"] = np.insert(packForDijkstra["retargetedLaplacianImage2D"], 0, [np.zeros([packForDijkstra["imageWidth"]], dtype=int)], axis=0)
    packForDijkstra["retargetedLaplacianImage2D"] = np.append(packForDijkstra["retargetedLaplacianImage2D"], [np.zeros([packForDijkstra["imageWidth"]], dtype=int)], axis=0)
    return packForDijkstra["retargetedLaplacianImage2D"]




def retargetWithDijkstra(imageForDijkstra):
    returnPath = []
    g = dj.Graph()

    # create vertices
    for i in range(len(imageForDijkstra)):
        for j in range(len(imageForDijkstra[0])):
            g.add_vertex(f'{i}, {j}')

    # create weighted edges
    for i in range(len(imageForDijkstra) - 1):
        for j in range(len(imageForDijkstra[0])):
            # If it's first row (all-zeros) we make an edge to all nodes in the real first row
            if i == 0 or i == len(imageForDijkstra) - 2:
                for k in range(len(imageForDijkstra[0])):
                    g.add_edge(f'{i}, {j}', f'{i + 1}, {k}')
            # If it's a normal row we make edges to bottom 3 nodes, min and max functions prevent going off-limits on the array (side nodes have only 2 edges)
            else:
                for k in range(max(0, j - 1), min(j + 1, len(imageForDijkstra[0]) - 1) + 1):
                    g.add_edge(f'{i}, {j}', f'{i + 1}, {k}', imageForDijkstra[i + 1][k])



    dj.dijkstra(g, g.get_vertex(f'0, 0'))
    target = g.get_vertex(f'{len(imageForDijkstra) - 1}, 0')
    path = [target.get_id()]
    dj.shortest(target, path)
    path = [str(int(node[0]) - 2) + node[1:] for node in
            path]  # This line just adjusts the row and col numbers to fit the original array indices
    sortedPath = path[::-1][2:-1]

    for step in sortedPath:
        returnPath.append(int(step.split(', ')[1]))
    return returnPath




def saveTheRouteAndPaintRedForDijkstra(retargetedImage, path):
    imageHeight = retargetedImage.imageHeight

    retargetedImage.markedLaplacianImage.setflags(write=1)
    retargetedImage.markedImage.setflags(write=1)
    retargetedImage.laplacianImage2D.setflags(write=1)

    for x in range(imageHeight):
        retargetedImage.markedLaplacianImage[x][ path[x] ][0] = 255
        retargetedImage.markedLaplacianImage[x][ path[x] ][1] = 0
        retargetedImage.markedLaplacianImage[x][ path[x] ][2] = 0

        retargetedImage.markedImage[x][path[x]][0] = 255
        retargetedImage.markedImage[x][path[x]][1] = 0
        retargetedImage.markedImage[x][path[x]][2] = 0




def adjustDijPaths(dijPaths, currentPath, p):
    print("\nAdjusting selected pixels by Dijkstra...")
    currentPathAdjusted = currentPath[:]

    for i in range(len(currentPathAdjusted)):  # We're assuming that all the previous (deleted) pixels affected the current selected pixels
        currentPathAdjusted[i] += p            #   That means all the previous (deleted) pixels are smaller than current selected pixels

    dijPaths = list(zip(*dijPaths))  # Transpose process - Rows to column, columns to row.

    for i in range(len(dijPaths)):  # Each [i] will show us a list that includes which pixels was removed from row [i]
        dijPaths[i] = sorted(list(dijPaths[i]))
        for h in range(len(dijPaths[i])-1, -1, -1):  # Start checking from the biggest (deleted) pixel in the row
            if dijPaths[i][h] >= currentPathAdjusted[i]:  # If deleted pixel bigger or equal, it shouldn't affect current selected pixel's index,
                currentPathAdjusted[i] -= 1               #   so we should minus 1 since we assume that all previous pixels affect the current

    dijPaths = list(zip(*dijPaths))
    dijPaths.append(currentPathAdjusted)
    return dijPaths




def completeRetargetProcessForDijkstra(retargetedImage, pathForPainting, pathForRemoving):
    imageHeight = retargetedImage.imageHeight
    retargetedLaplacianImage2DList = retargetedImage.retargetedLaplacianImage2D.tolist()
    retargetedImageList = retargetedImage.retargetedImage.tolist()

    retargetedImage.markedLaplacianImage.setflags(write=1)
    retargetedImage.markedImage.setflags(write=1)
    retargetedImage.laplacianImage2D.setflags(write=1)
    retargetedImage.retargetedLaplacianImage2D.setflags(write=1)
    retargetedImage.retargetedImage.setflags(write=1)

    for x in range(imageHeight):
        retargetedImage.markedLaplacianImage[x][ pathForPainting[x] ][0] = 255
        retargetedImage.markedLaplacianImage[x][ pathForPainting[x] ][1] = 0
        retargetedImage.markedLaplacianImage[x][ pathForPainting[x] ][2] = 0

        retargetedImage.markedImage[x][pathForPainting[x]][0] = 255
        retargetedImage.markedImage[x][pathForPainting[x]][1] = 0
        retargetedImage.markedImage[x][pathForPainting[x]][2] = 0

        del retargetedLaplacianImage2DList[x][pathForRemoving[x]]
        del retargetedImageList[x][pathForRemoving[x]]

    retargetedImage.retargetedLaplacianImage2D = np.array(retargetedLaplacianImage2DList)
    retargetedImage.retargetedImage = np.array(retargetedImageList)




def ImageRetarget(retargetedImageOriginal, pixelNumber, retargetMethod = "g"):
    if retargetMethod == "d" or retargetMethod == "dijkstra":
        retargetMethodName = "Dijkstra"
    elif retargetMethod == "g" or retargetMethod == "greedy":
        retargetMethodName = "Greedy"
    else:
        print("Wrong retarget method entered.")
        exit()
    print("\n\n ImageRetarget process was started by %s method..." %(retargetMethodName))
    print("Width of the Image will be decreased by {} pixel.".format(pixelNumber))
    startTime = datetime.now()

    retargetedImage = copy.deepcopy(retargetedImageOriginal) # If we don't do that, it uses directly the reference of the original, instead of sending a copy to methods
    usedPixels = []  # This 2D array will be used to save marked/retargeted pixels
                     #   when a route selected for marking/retargeting
    removingPixels = np.zeros([retargetedImage.imageHeight, pixelNumber], dtype=int)

    routeAndSumOfEnergyPoints = []
    energyPoints = np.zeros([retargetedImage.imageWidth], dtype=int)
    path = np.zeros([retargetedImage.imageHeight+1, retargetedImage.imageWidth], dtype=int) # Creating an array that one pixel bigger than original image's height. (add one more row)
                                                                                            #   Because algorithm will use that extra pixels to write total of energy points of each
                                                                                            #   row when it starts from them. For example, if algorithm start to creating retargeting path
                                                                                            #   from 42th pixel ([0][41]), whole path will be calculating using energies with greedy search
                                                                                            #   and energies will be summed for the path and will be written to the last row's 42th element
                                                                                            #   which is [imageHeight][41]. So algorithm can compare all and select which path is better to retarget

    dijPaths = []  # Stores all selected paths by Dijkstra - len will be same as given retarget pixels by end-user

    t0 = time.time() # [Not-Important] This is just for interface
    usedPixelsEstimatedTime = 0
    tFirst = 0
    for p in range(pixelNumber):
        t3 = time.time() # [Not-Important] This is just for interface
        markedLaplacianImage2D = convert3Dto2D(retargetedImage.markedLaplacianImage)

        if retargetMethod == "g" or retargetMethod == "greedy":
            packForGreedy = {
                "laplacianImage2D": markedLaplacianImage2D,
                "usedPixels": usedPixels,
                "processNumber": p,
                "routeAndSumOfEnergyPoints": routeAndSumOfEnergyPoints,
                "energyPoints": energyPoints,
                "path": path
            }

            routeAndSumOfEnergyPoints = retargetWithGreedy(packForGreedy) # This 2D array shows to route for each column and sum of energy points on last row
                                                                          #   For example if algorithm start marking/retargeting on 25.pixel, [0][24] will be also 24
                                                                          #   and next step will be seen on [1][24] - which neighbor will be chosen (left, center, right),
                                                                          #   if it's left, will be writen 23 on there, 24 for center and 25 for right.
                                                                          #   Now next step on [2][24] etc. And on the [imageHeight][24], sum of
                                                                          #   energy points can be seen if select start point as 25.pixel
            energyPoints = routeAndSumOfEnergyPoints[retargetedImage.imageHeight]
            path = routeAndSumOfEnergyPoints

            startPoint = determineStartPoint(routeAndSumOfEnergyPoints, usedPixels)
            usedPixels, removingPixels = saveTheRoute(routeAndSumOfEnergyPoints, startPoint, usedPixels, p,
                                                      removingPixels)

            paintRedForGreedySearch(retargetedImage.markedLaplacianImage, routeAndSumOfEnergyPoints,
                                    startPoint)  # Marking Seam Carving curves on the Laplacian transformed image
            paintRedForGreedySearch(retargetedImage.markedImage, routeAndSumOfEnergyPoints,
                                    startPoint)  # Marking Seam Carving curves on the copy of original image


        elif retargetMethod == "d" or retargetMethod == "dijkstra":
            packForDijkstra = {
                "retargetedLaplacianImage2D": retargetedImage.retargetedLaplacianImage2D,
                "imageWidth": len(retargetedImage.retargetedLaplacianImage2D[0])
            }

            imageForDijkstra = prepareForDijkstra(packForDijkstra)
            currentDijPath = retargetWithDijkstra(imageForDijkstra)
            dijPaths = adjustDijPaths(dijPaths, currentDijPath[:], p)

            completeRetargetProcessForDijkstra(retargetedImage, dijPaths[p], currentDijPath)  # Painting red and deleting selected route from retargetedImage

        sp.overwritePreviousLine()
        sys.stdout.write("Image Retarget process {}/{} COMPLETED    |    ".format(p + 1, pixelNumber))
        t1 = time.time()  # [Not-Important] This is just for interface

        # estimatedRemainingTime = (((t1 - t0) / (p + 1)) * (pixelNumber - p))
        # sp.showRemainingTime(estimatedRemainingTime)
        sp.showRemainingTime(t1 - t3, message=f"The time spent for the {p+1}.process: ")

        # if p > 1:  # [Not-Important] This is just for interface
        #     if p == 1:
        #         tFirst = t1-t3
        #     elif p == 2:
        #         #usedPixelsEstimatedTime = (t1 - t3) * pixelNumber
        #         usedPixelsEstimatedTime = sum( ( (t1-t3) - tFirst ) * i for i in range(1, pixelNumber+1) )
        #         # estimatedRemainingTime = ((t1 - t0) / (p + 1)) * pixelNumber
        #     # print("p:" + str(p) + ", t1:" + str(timedelta(seconds=t1)) + ", t0:" + str(timedelta(seconds=t0)))
        #     lastLoopTime = t1 - t3
        #     # if p == 10:
        #     #     t1 = (t1 - t0) / 10
        #     #     for t in range(pixelNumber, 0, -1):
        #     #         usedPixelsEstimatedTime += t * t1
        #     if retargetMethod == "d" or retargetMethod == "dijkstra":
        #         if p == 0:
        #             x = ( t1 - t3 ) / pixelNumber
        #             totalX = 0
        #             for xin in range(p, pixelNumber):
        #                 totalX += x * xin
        #         # estimatedRemainingTime = (((t1 - t0) / (p + 1)) * (pixelNumber - p))  # It's nothing but user-friendly interface
        #         # usedPixelsEstimatedTime = estimatedRemainingTime
        #         usedPixelsEstimatedTime = totalX
        #     # estimatedRemainingTime = (estimatedRemainingTime * 0.50) + ( (lastLoopTime * (p + 10) ) * 0.50 )
        #     # estimatedRemainingTime = ((t1 - t0) / (p + 1)) * pixelNumber + usedPixelsEstimatedTime
        #     sp.showRemainingTime(usedPixelsEstimatedTime) # [Not-Important] This is just for interface
        #     usedPixelsEstimatedTime -= t1 - t3
        # sys.stdout.write('\x1b[1A')  # Moves up one line on terminal
        # sys.stdout.write('\r')  # Moves the cursor to the beginning of the line
        # sp.overwritePreviousLine()


    if retargetMethod == "g" or retargetMethod == "greedy":
        retargetedImage.retargetedImage = removeMarkedPixels(retargetedImage.retargetedImage, removingPixels)

    retargetedImageOriginal.setMarkedImages(retargetedImage)
    retargetedImageOriginal.setRetargetedImage(retargetedImage)

    endTime = datetime.now()
    processTime = endTime - startTime
    imageName = retargetedImageOriginal.imageDirectory.split('/')[-1]
    statisticsFileSaveName = imageName + "_" + retargetMethodName
    statistics = "Image name: " + imageName
    statistics += "\nRetarget method: " + retargetMethodName
    statistics += "\nRetarget pixels: " + str(pixelNumber)
    statistics += "\n\nStart time: " + startTime.strftime("%H:%M:%S (.%f)")
    statistics += "\nEnd time: " + endTime.strftime("%H:%M:%S (.%f)")
    statistics += "\nProcess time: " + str(processTime)
    print(statistics)
    statistics += "\n\n " + ("usedPixels = " if retargetMethodName == "Greedy" else "dijPaths = ")
    statistics += str(usedPixels) if retargetMethodName == "Greedy" else str(dijPaths)
    retargetedImageOriginal.saveStatistics(statisticsFileSaveName, endTime, statistics)




def readFromDirectory(directory, retargetMethod = 'g', skip = 0):
    if retargetMethod == "d" or retargetMethod == "dijkstra":
        retargetMethodName = "Dijkstra"
    elif retargetMethod == "g" or retargetMethod == "greedy":
        retargetMethodName = "Greedy"
    else:
        print("Wrong retarget method entered.")
        exit()
    # images = os.listdir(directory)
    images = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]  # Read only files but not folders in the directory
    print(f"{len(images)} file(s) found in the \"{directory}\" directory.")
    for i in range(0, len(images)):
        print(f"Retarget process of {i+1}. image ({images[i]}) was started.")
        if i < skip:
            print(f"\n{i+1} skipped...")
            continue
        retargetedImageObject = RetargetedImage(directory + "/" + images[i])
        retargetPixelNumber = retargetedImageObject.imageWidth - retargetedImageObject.imageHeight
        ImageRetarget(retargetedImageObject, retargetPixelNumber, retargetMethod)  # How many pixel you want to be retargeted

        imageExtension = images[i].split(".").pop()
        imageNameAndExtension = []
        imageNameAndExtension.append(images[i][:len(images[i]) - len(imageExtension)])
        imageNameAndExtension.append(imageExtension)

        retargetedImageObject.showMarkedImage("fsave", f"{imageNameAndExtension[0]}_{retargetMethodName}_MarkedImage.{imageNameAndExtension[1]}")
        retargetedImageObject.showLaplacianImage("fsave", f"{imageNameAndExtension[0]}_{retargetMethodName}_LaplacianImage.{imageNameAndExtension[1]}")
        retargetedImageObject.showMarkedLaplacianImage("fsave", f"{imageNameAndExtension[0]}_{retargetMethodName}_MarkedLaplacianImage.{imageNameAndExtension[1]}")
        retargetedImageObject.showRetargetedImage("fsave", f"{imageNameAndExtension[0]}_{retargetMethodName}_RetargetedImage.{imageNameAndExtension[1]}")

        print(f"\n{i+1} of {len(images)} images' retarget process COMPLETED!\n")



class RetargetedImage:
    imageDirectory = ""
    image = None
    grayImage = None
    laplacianImage2D = None
    laplacianImage = None
    markedLaplacianImage = None
    markedImage = None
    retargetedImage = None
    imageHeight = 0
    imageWidth = 0
    statistics = ""

    def __init__(self, imageDirectory):
        self.imageDirectory = imageDirectory
        self.image = pltim.imread(self.imageDirectory)
        self.grayImage = rgb2gray(self.image)
        self.laplacianImage2D = laplacianTransform(self.grayImage).astype(np.float32)
        self.laplacianImage = gray2rgb(self.laplacianImage2D)
        self.markedLaplacianImage = self.laplacianImage
        self.markedImage = self.image
        self.retargetedLaplacianImage2D = self.laplacianImage2D
        self.retargetedImage = self.markedImage
        self.imageHeight = len(self.laplacianImage2D)
        self.imageWidth = len(self.laplacianImage2D[0])

    def setMarkedImages(self, newObject):
        self.markedLaplacianImage = newObject.markedLaplacianImage
        self.markedImage = newObject.markedImage

    def setRetargetedImage(self, newObject):
        self.retargetedImage = newObject.retargetedImage

    def setStatistics(self, statistics):
        self.statistics = statistics

    def showOriginalImage(self, save="", fileName="savedImage"):
        fig, ax = plt.subplots(figsize=(len(self.image[0]),len(self.image)), dpi=1)
        if save == "save" or save == "-s":
            RetargetedImage.setShowAndSave(self.image, fig, ax, fileName)
        elif save == "fSave" or save == "fsave":
            RetargetedImage.setAndSave(self.image, fig, ax, fileName)
        else:
            RetargetedImage.setAndShow(self.image, fig, ax)

    def showGrayImage(self, save="", fileName="savedImage"):
        fig, ax = plt.subplots(figsize=(len(self.grayImage[0]),len(self.grayImage)), dpi=1)
        if save == "save" or save == "-s":
            RetargetedImage.setShowAndSave(self.grayImage, fig, ax, fileName, 'gray')
        elif save == "fSave" or save == "fsave":
            RetargetedImage.setAndSave(self.grayImage, fig, ax, fileName, 'gray')
        else:
            RetargetedImage.setAndShow(self.grayImage, fig, ax, 'gray')

    def showLaplacianImage2D(self, save="", fileName="savedImage"):
        fig, ax = plt.subplots(figsize=(len(self.laplacianImage2D[0]),len(self.laplacianImage2D)), dpi=1)
        if save == "save" or save == "-s":
            RetargetedImage.setShowAndSave(self.laplacianImage2D, fig, ax, fileName, 'gray')
        elif save == "fSave" or save == "fsave":
            RetargetedImage.setAndSave(self.laplacianImage2D, fig, ax, fileName, 'gray')
        else:
            RetargetedImage.setAndShow(self.laplacianImage2D, fig, ax, 'gray')

    def showLaplacianImage(self, save="", fileName="savedImage"):
        fig, ax = plt.subplots(figsize=(len(self.laplacianImage[0]),len(self.laplacianImage)), dpi=1)
        if save == "save" or save == "-s":
            RetargetedImage.setShowAndSave(self.laplacianImage, fig, ax, fileName)
        elif save == "fSave" or save == "fsave":
            RetargetedImage.setAndSave(self.laplacianImage, fig, ax, fileName)
        else:
            RetargetedImage.setAndShow(self.laplacianImage, fig, ax)

    def showMarkedLaplacianImage(self, save="", fileName="savedImage"):
        fig, ax = plt.subplots(figsize=(len(self.markedLaplacianImage[0]),len(self.markedLaplacianImage)), dpi=1)
        if save == "save" or save == "-s":
            RetargetedImage.setShowAndSave(self.markedLaplacianImage, fig, ax, fileName)
        elif save == "fSave" or save == "fsave":
            RetargetedImage.setAndSave(self.markedLaplacianImage, fig, ax, fileName)
        else:
            RetargetedImage.setAndShow(self.markedLaplacianImage, fig, ax)

    def showMarkedImage(self, save="", fileName="savedImage"):
        fig, ax = plt.subplots(figsize=(len(self.markedImage[0]),len(self.markedImage)), dpi=1)
        if save == "save" or save == "-s":
            RetargetedImage.setShowAndSave(self.markedImage, fig, ax, fileName)
        elif save == "fSave" or save == "fsave":
            RetargetedImage.setAndSave(self.markedImage, fig, ax, fileName)
        else:
            RetargetedImage.setAndShow(self.markedImage, fig, ax)

    def showRetargetedImage(self, save="", fileName="savedImage"):
        fig, ax = plt.subplots(figsize=(len(self.retargetedImage[0]),len(self.retargetedImage)), dpi=1)
        if save == "save" or save == "-s":
            RetargetedImage.setShowAndSave(self.retargetedImage, fig, ax, fileName)
        elif save == "fSave" or save == "fsave":
            RetargetedImage.setAndSave(self.retargetedImage, fig, ax, fileName)
        else:
            RetargetedImage.setAndShow(self.retargetedImage, fig, ax)

    def setForShow(self, fig, ax, cmap):
        fig.subplots_adjust(0, 0, 1, 1)
        if cmap != "":
            ax.imshow(self, cmap)
        else:
            ax.imshow(self)
        plt.axis('off')

    def setAndShow(self, fig, ax, cmap=""):
        RetargetedImage.setForShow(self, fig, ax,cmap)
        plt.show()

    def setShowAndSave(self, fig, ax, fileName, cmap=""):
        RetargetedImage.setForShow(self, fig, ax, cmap)
        saveDirectory = "saved/{}.png".format(fileName)
        fig.savefig(saveDirectory)
        plt.show()

    def setAndSave(self, fig, ax, fileName, cmap=""):
        RetargetedImage.setForShow(self, fig, ax, cmap)
        saveDirectory = "saved/{}.png".format(fileName)
        fig.savefig(saveDirectory)

    def saveStatistics(self, fileName, time, statistics):
        f = open("saved/{}_statistics_{}.txt".format(fileName, time.strftime("%H-%M-%S+%d-%m-%Y")), "w")
        f.write(statistics)
        f.close()




# ENCHANTED IMAGE SHRINKER V2.0
#
# Usage of program
#
# - Create an RetargetImage object.
#    ex:
#       exampleObject = RetargetedImage("[IMAGE FULL DIRECTORY]")
# - Determine how many pixels you want to retarget with using ImageRetarget method.
#    ex:
#       ImageRetarget(exampleObject, [PIXEL NUMBER], [RETARGET ALGORITHM]) | Retarget Algorithm:  "g" or "greedy" for Greedy, "d" or "dijkstra" for Dijkstra
# - Show images you wanted. See below for the methods.
#          showRetargetedImage() = It shows the final product which is retargeted image
#            showOriginalImage() = It shows original image
#                showGrayImage() = It shows gray level image ('gray' color map will be applied)
#           showLaplacianImage() = It shows Laplacian transformed image (3D Array with RGB channels)
#         showLaplacianImage2D() = It shows Laplacian transformed image (2D Array with one channel)
#     showMarkedLaplacianImage() = It shows marked curves for retargeting on the Laplacian transformed image (2D)
#              showMarkedImage() = It shows marked curves for retargeting on the original image
# - If you prefer to save the image, give "save" or "-s" parameters to these methods and give a file name as second parameters. It will be saved in "saved" folder.
#    ex:
#       showRetargetedImage("save", "Retargeted Image")
#       showMarkedImage("-s", "Retargeting Curves")
# - If you want to skip showing part and force algorithm to save directly instead, use "fSave"
#    ex:
#       showRetargetedImage("fSave", "Retargeted Image")


# # readFromDirectory("assets/NRID/Optimized", "g")
readFromDirectory("assets/NRID/Optimized", "g")
# readFromDirectory("assets/NRID/move_later", "d")

# retargetedImageObject = RetargetedImage("assets/ours_23_aaa_optimized.jpg")
# ImageRetarget(retargetedImageObject, 29, "g")
# retargetedImageObject.showMarkedImage("fsave", "zxczxc")
# retargetedImageObject.showLaplacianImage("fsave", "zxczxc")
# retargetedImageObject.showMarkedLaplacianImage("fsave", "zxczxc")
# retargetedImageObject.showRetargetedImage("fsave", "zxczxc")



# # Image retargeting with using greedy algorithm
# madMansGreedy = RetargetedImage("assets/mad_mans_and_catsVeryOptimized.JPG")
# ImageRetarget(madMansGreedy, 1, retargetMethod='g')  # How many pixel you want to be retargeted
# #
# madMansGreedy.showLaplacianImage("save", "madManVO-Gre-MarkedLaplacian")
# madMansGreedy.showMarkedImage("fsave", "madManVO-Gre-MarkedOriginal")
# madMansGreedy.showRetargetedImage("fsave", "madManVO-Gre-Retargeted")
#
#
# # Image retargeting with using Dijkstra algorithm
# madMansDijkstra = RetargetedImage("mad_mans_and_catsVeryOptimized.JPG")
# ImageRetarget(madMansDijkstra, 50, retargetMethod='d')
#
# madMansDijkstra.showMarkedLaplacianImage("fsave", "madManVO-Dij-MarkedLaplacian")
# madMansDijkstra.showMarkedImage("fsave", "madManVO-Dij-MarkedOriginal")
# madMansDijkstra.showRetargetedImage("fsave", "madManVO-Dij-Retargeted")


# Image retargeting with using greedy algorithm
# pixilFrameGreedy = RetargetedImage("assets/pixilFrame.png")
# ImageRetarget(pixilFrameGreedy, 25, retargetMethod='g')  # How many pixel you want to be retargeted
#
# pixilFrameGreedy.showOriginalImage("fsave", "dfghxdfg")
# pixilFrameGreedy.showMarkedImage("fsave", "dfghxdfg")
# pixilFrameGreedy.showRetargetedImage("fsave", "dfghxdfg")




# pixilFrameGreedy = RetargetedImage("assets/pixilFrame.png")
# ImageRetarget(pixilFrameGreedy, 25, retargetMethod='d')  # How many pixel you want to be retargeted
#
# pixilFrameGreedy.showMarkedImage("save", "asd")
# pixilFrameGreedy.showLaplacianImage("fsave", "asd")
# pixilFrameGreedy.showRetargetedImage("save", "asd")



# pixilFrameGreedy = RetargetedImage("assets/extreme-small.png")
# ImageRetarget(pixilFrameGreedy, 6, retargetMethod='d')  # How many pixel you want to be retargeted
#
# pixilFrameGreedy.showMarkedImage("pixilFrame-Gre-Laplacian")


# # Image retargeting with using Dijkstra algorithm
# madMansDijkstra = RetargetedImage("assets/pixilFrame.png")
# ImageRetarget(madMansDijkstra, 25, retargetMethod='d')
#
# madMansDijkstra.showMarkedLaplacianImage("fsave", "pixilFrame-Dij-MarkedLaplacian")
# madMansDijkstra.showMarkedImage("fsave", "pixilFrame-Dij-MarkedOriginal")
# madMansDijkstra.showRetargetedImage("fsave", "pixilFrame-Dij-Retargeted")