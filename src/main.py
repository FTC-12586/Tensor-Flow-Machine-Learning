import sys

from src.DatasetImporter import DatasetImporter


def main():
    DatasetImporter.load(r"C:\Users\willm\PycharmProjects\Tensor-Flow-Machine-Learning\datasetExamples"
                         r"\exampleDownloadedDataset.zip")
    sys.exit(0)


if __name__ == "__main__":
    main()
