
import numpy as np
from PIL import Image
import cv2
from tqdm.notebook import tqdm
from tqdm import tqdm
import random

def changing(matrix):
    lst = []
    for i in range(32):
        for j in range(32):
            lst.append(matrix[i][j])
    return lst


class Weight:

    def __init__(self, lines, columns):
        self.lines = lines
        self.columns = columns
        self.Matrix = [[0] * self.columns for i in range(self.lines)]
        for i in range(self.lines):
            for j in range(self.columns):
                self.Matrix[i][j] = random.uniform(-1, 1)

    def less(self, numberLine, arrayPix):
        for i in range(self.columns):
            if (arrayPix[i] == 1): #если пиксель не белый
                self.Matrix[numberLine][i] = self.Matrix[numberLine][i] - 0.1

    def up(self, numberLine, arrayPix):
        for i in range(self.columns):
            if (arrayPix[i] == 1):
                self.Matrix[numberLine][i] = self.Matrix[numberLine][i] + 0.1


def lean():
    Num_era = 0
    weights = Weight(10, 1024)
    while Num_era < 50:
        for variantNumber in range(23):
            for digitNumber in range(10):
                path = "Dataset\i" + str(digitNumber) + str(variantNumber) + ".png"
                src = cv2.imread(path)
                pixels = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY) / 255
                arrayPix = changing(pixels)  # входные нейроны массив
                for i in range(len(arrayPix)):
                    if(arrayPix[i] < 1):
                        arrayPix[i] = 1
                    else:
                        arrayPix[i] = 0
                result = np.dot(weights.Matrix, arrayPix)  # вероятности для каждой цифры
                index, maxP = max(enumerate(result), key=lambda i_v: i_v[1])
                if (index != digitNumber):
                    weights.less(index, arrayPix)
                    weights.up(digitNumber, arrayPix)

        Num_era += 1
    return weights

def get_value(path, weights):
    src = cv2.imread(path)
    pixels = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY) / 255
    arrayPix = changing(pixels)  # входные нейроны
    for i in range(len(arrayPix)):
        if (arrayPix[i] < 1):
            arrayPix[i] = 1
        else:
            arrayPix[i] = 0
    result = np.dot(weights.Matrix, arrayPix)  # вероятности для каждой цифры
    index, maxP = max(enumerate(result), key=lambda i_v: i_v[1])
    return index

def punish(path, weights, index):
    src = cv2.imread(path)
    pixels = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY) / 255
    arrayPix = changing(pixels)  # входные нейроны
    for i in range(len(arrayPix)):
        if (arrayPix[i] < 1):
            arrayPix[i] = 1
        else:
            arrayPix[i] = 0
    weights.less(index, arrayPix)
    result = np.dot(weights.Matrix, arrayPix)  # вероятности для каждой цифры
    index, maxP = max(enumerate(result), key=lambda i_v: i_v[1])
    return index




