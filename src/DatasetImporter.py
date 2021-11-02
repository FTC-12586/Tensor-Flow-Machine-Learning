import io
import os
from zipfile import ZipFile


class DatasetImporter:
    @staticmethod
    def _parse_record_file(file_contents: io.BytesIO)-> list:
        pass

    @staticmethod
    def _parse_pbxt_file(file_contents: io.BytesIO) -> dict:
        # reads in the data from the file object as a string
        data = file_contents.read().decode()

        # splits the data into sections
        data = data.split('\n')

        # removes all lines except the id and name lines
        for i in range(0, len(data)):
            data[i] = data[i].strip()
        while 'item {' in data:
            data.remove('item {')
        while '}' in data:
            data.remove('}')
        while '' in data:
            data.remove('')

        ids = []
        names = []

        # fills out a list of names and id's so that id[x] is the id of name[x]
        for i in range(0, len(data)):
            if 'id: ' in data[i]:
                ids.append(data[i])
            elif 'name:' in data[i]:
                names.append(data[i])
            else:
                raise ValueError(
                    "The File is Malformed")  # sanity check on the file as it should never reach this point

        # sanity check on the file
        if len(ids) != len(names):
            raise ValueError("The File is Malformed")

        d = {}
        for i in range(0, len(ids)):
            d[ids[i].split(': ')[1]] = names[i].split("'")[1]  # makes the id the dictionary key and the name the value
        return d

    @staticmethod
    def _is_record_file(file_name: str) -> bool:
        path, ext = os.path.splitext(file_name)
        ext = ext.split('-')[0].strip()
        if ext == '.record':
            return True
        return False

    @staticmethod
    def _is_pbxt(file_name: str)-> bool:
        path, ext = os.path.splitext(file_name)
        if ext == '.pbxt':
            return True
        return False

    @staticmethod
    def load(file_name: str):
        z = ZipFile(file_name, 'r')
        record_files = []
        for file in z.filelist:
            if DatasetImporter._is_pbxt(file.filename):
                label = DatasetImporter._parse_pbxt_file(file)
            elif DatasetImporter._is_record_file(file.filename):
                record_files.append(DatasetImporter._parse_record_file(file))
        z.close()
