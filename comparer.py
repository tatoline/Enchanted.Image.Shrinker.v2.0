import matplotlib.image as pltim
import time

class ImageComparer:
    supervisedImage = None
    testImage = None
    name = None
    totalPixels = None  # Total pixel number of the image
    totalSelectedPixels = None  # Correctly selected pixels
    totalUnselectedPixels = None  # Correctly unselected pixels
    perfectTotalSelectedPixels = None  # Number of correctly selected pixels if the success rate 100%
    perfectTotalUnselectedPixels = None  # Number of correctly unselected pixels if the success rate 100%

    def __init__(self, supervisedImageDirectory, testImageDirectory):
        self.supervisedImagePixelPaths = {
            "selected": [],
            "unselected": []
        }
        self.testImagePixelPaths = {
            "selected": [],
            "unselected": []
        }
        self.intersectionPixelPaths = {
            "selected": [],
            "unselected": []
        }
        self.results = {
            "TP": 0,
            "FP": 0,
            "TN": 0,
            "FN": 0,
            "Sensitivity": 0,  # TP / (TP + FN) - True Positive Rate
            "Specificity": 0,  # TN / (TN + FP) - True Negative Rate
            "PPV": 0,  # TP / (TP + FP) - Positive predictive Value
            "NPV": 0,  # TN / (TN + FN) - Negative Predictive Value
            "%": 0
        }

        self.supervisedImage = pltim.imread(supervisedImageDirectory)
        self.testImage = pltim.imread(testImageDirectory)
        self.name = testImageDirectory.split("/")[-1].split(".")[-2]
        self.totalPixels = len(self.testImage) * len(self.testImage[0])
        self.determinePixelPaths()
        self.perfectTotalSelectedPixels = len(self.supervisedImagePixelPaths["selected"])
        self.perfectTotalUnselectedPixels = self.totalPixels - self.perfectTotalSelectedPixels
        self.intersectionPixelPaths["selected"] = self.intersection(self.supervisedImagePixelPaths["selected"],
                                                                    self.testImagePixelPaths["selected"])
        self.intersectionPixelPaths["unselected"] = self.intersection(self.supervisedImagePixelPaths["unselected"],
                                                                    self.testImagePixelPaths["unselected"])
        self.totalSelectedPixels = len(self.intersectionPixelPaths["selected"])
        self.totalUnselectedPixels = len(self.intersectionPixelPaths["unselected"])

        self.results["TP"] = self.totalSelectedPixels
        self.results["FP"] = self.perfectTotalSelectedPixels -self.totalSelectedPixels
        self.results["TN"] = self.totalUnselectedPixels
        self.results["FN"] = self.perfectTotalUnselectedPixels -self.totalUnselectedPixels
        self.results["Sensitivity"] = self.results["TP"] / (self.results["TP"] + self.results["FN"])
        self.results["Specificity"] = self.results["TN"] / (self.results["TN"] + self.results["FP"])
        self.results["PPV"] = self.results["TP"] / (self.results["TP"] + self.results["FP"])
        self.results["NPV"] = self.results["TN"] / (self.results["TN"] + self.results["FN"])
        self.results["%"] = (self.results["TP"] * 100) / self.perfectTotalSelectedPixels

    def determinePixelPaths(self):
        for i in range(len(self.supervisedImage)):
            for j in range(len(self.supervisedImage[i])):  # If the pixel color exactly "red", it's selected, otherwise unselected
                if self.supervisedImage[i][j][0] == 1 and self.supervisedImage[i][j][1] == 0 and self.supervisedImage[i][j][2] == 0:
                    self.supervisedImagePixelPaths["selected"].append([i, j])
                else:
                    self.supervisedImagePixelPaths["unselected"].append([i, j])
                if self.testImage[i][j][0] == 1 and self.testImage[i][j][1] == 0 and self.testImage[i][j][2] == 0:
                    self.testImagePixelPaths["selected"].append([i, j])
                else:
                    self.testImagePixelPaths["unselected"].append([i, j])

    def intersection(self, lst1, lst2):
        map1 = map(tuple, lst1)
        map2 = map(tuple, lst2)
        set1 = set(map1)
        set2 = set(map2)
        return list(set1.intersection(set2))

    def showResults(self):
        resultText = self.name
        resultText += "\n-----------------"
        resultText += "\nImage dimension: " + str(len(self.testImage)) + "x" + str(len(self.testImage[0]))
        resultText += "\nTotal pixel: " + str(self.totalPixels)
        resultText += "\nTP: " + str(self.results["TP"])
        resultText += "\nFP: " + str(self.results["FP"])
        resultText += "\nTN: " + str(self.results["TN"])
        resultText += "\nFN: " + str(self.results["FN"])
        resultText += "\nSensitivity/TPR:" + str(self.results["Sensitivity"])
        resultText += "\nSpecificity/TNR:" + str(self.results["Specificity"])
        resultText += "\nPPV:" + str(self.results["PPV"])
        resultText += "\nNPV:" + str(self.results["NPV"])
        resultText += "\n\nSuccess rate is " + str(self.results["%"]) + "%"
        resultText += "\n-----------------\n\n"
        f = open("saved/{}_results_{}.txt".format(self.name, time.strftime("%H-%M-%S+%d-%m-%Y")), "w")
        f.write(resultText)
        f.close()
        print(resultText)


if __name__ == '__main__':
    supervisedImageDirectory = "assets/mad_mans_and_catsVeryOptimized-Supervised.png"
    testImageGreedyDirectory = "saved/madManVO-Gre-MarkedOriginal.png"
    testImageDijkstraDirectory = "saved/madManVO-Dij-MarkedOriginal.png"

    madMansVOGreedy = ImageComparer(supervisedImageDirectory, testImageGreedyDirectory)
    madMansVODijkstra = ImageComparer(supervisedImageDirectory, testImageDijkstraDirectory)

    madMansVOGreedy.showResults()
    madMansVODijkstra.showResults()


    supervisedImageDirectory = "assets/pixilFrame-Supervised.png"
    testImageGreedyDirectory = "saved/pixilFrame-Gre-MarkedOriginal.png"
    testImageDijkstraDirectory = "saved/pixilFrame-Dij-MarkedOriginal.png"

    pixilFrameGreedy = ImageComparer(supervisedImageDirectory, testImageGreedyDirectory)
    pixilFrameDijkstra = ImageComparer(supervisedImageDirectory, testImageDijkstraDirectory)

    pixilFrameGreedy.showResults()
    pixilFrameDijkstra.showResults()