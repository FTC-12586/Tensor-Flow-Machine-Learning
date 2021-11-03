import sys

from src.DatasetImporter import DatasetImporter


def main():
    # DatasetImporter.load(r"..\datasetExamples\exampleDownloadedDataset.zip")
    DatasetImporter.load_dataset(r"C:\Users\willm\Downloads\Ex\train_dataset.record-00000-00001")
    sys.exit(0)


if __name__ == "__main__":
    main()
