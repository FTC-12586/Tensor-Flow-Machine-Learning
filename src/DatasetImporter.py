from zipfile import ZipFile, BadZipFile
# import tensorflow as tf
import io


class DatasetImporter:

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
    def load(file_name: str):
        z = ZipFile(file_name, 'r')
        with z.open("label.pbtxt", 'r') as label_handle:
            label_dictionary = DatasetImporter._parse_pbxt_file(label_handle)
            # print(label_dictionary)
