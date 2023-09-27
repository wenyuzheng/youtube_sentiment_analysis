import os

import pandas as pd
from torch.utils.data import Dataset

SST5_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, 'data', 'sst5')

class SST5Dataset(Dataset):
  def __init__(self, dataframe, encoder):
    self.dataframe = dataframe
    self.encoder = encoder

  def __getitem__(self, index):
    row = self.dataframe.iloc[index]
    sentence = row['sentence']
    score = row['score'] - 1
    return self.encoder(sentence, score)

  def __len__(self):
    return len(self.dataframe)

def load_dataset(category='train'):
  return pd.read_csv(os.path.join(SST5_PATH, 'sst5_{}.csv'.format(category)))
