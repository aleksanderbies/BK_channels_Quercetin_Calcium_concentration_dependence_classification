import numpy as np
import pandas as pd

class Series:
    """This class represents time series and prepares them for classification"""

    def __init__(self, path, category, sample_length=5000):
        """
        Object initiator
        :param path: path to file with time series
        :param category: name of category of time series
        :param sample_length: length of sample you want to create
        """
        self.path = path
        self.category = category
        self.sample_length = sample_length

    def prepare_TS(self):
        """
        This method prepares time series
        :return: Samples of time series
        """
        data_list = self.read_data()
        time_series = self.make_window(data_list, self.sample_length)
        return time_series

    def make_window(self, data_list, sample_length):
        chunks = [data_list[i:i + sample_length] for i in range(0, len(data_list), sample_length)]
        return chunks
    

    def read_data(self):
        """
        This method reads time series points from file and save it to list
        :return: list that contains points of whole time series
        """
        file = open(self.path, "r")
        data = file.read()
        data_list = data.split("\n")
        file.close()
        data_list.pop()
        return data_list

    def make_df(self):
        """
        This method makes dataframe from preprocessed data
        :return: data to classification
        """
        samples = self.prepare_TS()
        columns = ["f_" + str(i) for i in range(len(samples[0]))]
        df = pd.DataFrame(samples, columns=columns)
        category = [self.category for _ in range(df.shape[0])]
        df.loc[:, 'target'] = category
        return df
