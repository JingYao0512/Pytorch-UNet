import numpy as np 
import os 
import cv2 as cv
from shutil import copyfile 

rootFolder = "./kaggle_3m"
imageDataFolder = "./data/imgs"
maskDataFolder = "./data/masks"

dirnames = []
for root, dirs, files in os.walk(rootFolder):
    for file in files:
        if(file.endswith("mask.tif")):
            filePath = os.path.join(root, file)
            image = cv.imread(filePath, -1)
            image = (image / 255).astype('uint8')

            filename = file.replace("_mask.tif", ".tif")
            exportPath = os.path.join(maskDataFolder, filename)
            #copyfile(filePath, exportPath)
            cv.imwrite(exportPath, image)

            imagePath = os.path.join(root, filename)
            exportPath = os.path.join(imageDataFolder, filename)   
            copyfile(imagePath, exportPath)