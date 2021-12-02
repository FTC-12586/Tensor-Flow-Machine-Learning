from sys import exit
from src.DatasetImporter import DatasetImporter
import cv2


def main():
    # DatasetImporter.loadFile(r"datasetExamples/Dataset1/train_dataset.record-00000-00001")
    # DatasetImporter.loadFolder(r"datasetExamples/Dataset1")

    im = cv2.imread(r"C:\Users\willm\Downloads\Calc 11.png")
    im = DatasetImporter.ResizeFill(im)
    cv2.imshow("Hi", im)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
