import pandas as pd

class StatsHolder:
    """
    Literally a list holder with some helper functions to make saving important statistics easy
    """
    def __init__(self):
        self.fpss= []
        self.inference_times = []
        self.pics = []
        self.confidences = []

    def save_lists(self, fname):
        print("here")
        stats_dict = {'fps' : self.fpss, 'inf_times' : self.inference_times, 'confidences' : self.confidences,
                      'pics' : self.pics}
        df = pd.DataFrame.from_dict(stats_dict, orient='index')
        df.to_pickle(fname)