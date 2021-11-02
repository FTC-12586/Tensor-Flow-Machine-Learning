import sys

from src.DatasetImporter import DatasetImporter


def main():
    DatasetImporter.load(r"..\datasetExamples\exampleDownloadedDataset.zip")
    sys.exit(0)


if __name__ == "__main__":
    main()
