# Load data
import pytreebank
import os
import warnings
# Suppress: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
# order of the import matters and needs to be before pandas is loaded
warnings.filterwarnings("ignore")
import pandas as pd

print('Loading data...')
raw_data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'sst5')
dataset = pytreebank.load_sst(raw_data_path)

# Transform and save data
print('Transforming data...')
for corpus_type in ['train', 'test', 'dev']:
    df = pd.DataFrame(columns=['score', 'label', 'sentence'])
    for item in dataset[corpus_type]:
        for label, sentence in item.to_labeled_lines():
          df = df.append({'score': label + 1, 'label': ["very negative", "negative", "neutral", "positive", "very positive"][label], 'sentence': sentence}, ignore_index=True)
        # Uncomment below to get the scored sentences
        # score = item.to_labeled_lines()[0][0]
        # label = ["very negative", "negative", "neutral", "positive", "very positive"][score]
        # df = df.append({'score': score + 1, 'label': label, 'sentence': item.to_labeled_lines()[0][1]}, ignore_index=True)
    df.to_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'sst5_{}.csv'.format(corpus_type)), index=False)
    print('Saved {} data to {} with {} samples'.format(corpus_type, os.path.join(raw_data_path, 'sst5_{}.csv'.format(corpus_type)), len(df)))

print('Done!')
