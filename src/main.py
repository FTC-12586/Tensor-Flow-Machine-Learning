from sys import exit
from src.DatasetImporter import DatasetImporter
import cv2


def main():
    sets, labels = DatasetImporter.load(r"..\datasetExamples\Dataset1")
    breakpoint()
    im = cv2.imread(r"C:\Users\willm\Downloads\Calc 11.png")
    im = DatasetImporter.ResizeFill(im)
    cv2.imshow("Hi",im)
    exit(0)


if __name__ == "__main__":
    main()
