import sys

from src.DatasetImporter import DatasetImporter


def main():
    sets = DatasetImporter.load(r"..\datasetExamples\Dataset1")
    # print([x for x in DatasetImporter.absoluteFilePaths(r"..\datasetExamples\Dataset1")])
    print(sets)
    sys.exit(0)


if __name__ == "__main__":
    main()
