"""This file containes code for loading data/dataset files"""
import os.path
import pandas as pd


class Dataset:
    """Class representing dataset."""
    def __init__(self, path, sep=',', date_format=''):
        self.path = path
        self._holder = {}
        self._sep = sep
        self.date_format = date_format

    def load(self):
        """Loads dataset into memory"""
        for ts_id, ts in self:
            self._holder[ts_id] = ts

    def _list(self):
        """Lists all files in dataset_path"""
        return os.listdir(self.path)

    def __getitem__(self, item):
        """Returns pandas Series object, containing ts"""
        if item not in self:
            raise Exception(
                'There is not file in dataset {path} with the following'
                'name:{ts_id}'.format(path=self.path, ts_id=item)
            )

        if item in self._holder:
            return self._holder[item]

        ts = pd.read_csv(
            os.path.join(self.path, item),
            index_col=0,
            parse_dates=True,
            squeeze=True,
            sep=self._sep,
            date_parser=self._get_date_parser()
        )

        return ts

    def _get_date_parser(self):
        if self.date_format:
            date_parser = lambda x: pd.datetime.strptime(x, self.date_format)
        else:
            # if date_parser is None, pandas read_csv will use its own parser
            date_parser = None
        return date_parser

    def __contains__(self, item):
        """Checks whether ts_id is present in dataset_path"""
        return item in self._list()

    def __iter__(self):
        """Returns generator yielding (ts_id, ts) tuple"""
        for ts_id in self._list():
            yield ts_id, self[ts_id]

    def __str__(self):
        return '\n'.join(self._list())

    def __repr__(self):
        return '\n'.join(self._list())

    def __len__(self):
        return len(self._list())

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, path):
        if not path:
            raise Exception('Dataset path should not be empty')
        if not os.path.exists(path):
            raise Exception('The path doesn\'t exist')

        self._path = path