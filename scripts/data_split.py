import os, sys
import pandas as pd

from allennlp.common.file_utils import cached_path
from sklearn.model_selection import train_test_split

def save(df, fp):
    df.to_csv(fp, sep='\t')
    print(f'{fp}: {df.shape}')
    
    labels = sorted(pd.unique(df.filter(regex=r"option\_.*\_\d", axis=1).values.ravel('K')))
    print(f'Labels: {labels}')

if __name__ == '__main__':
    url = sys.argv[1]
    out_path = os.path.join('/tmp', 'allenrank', 'data', url.split('/')[-1].split('.')[0].lower())
    os.makedirs(out_path, exist_ok=True)
    
    df = pd.read_csv(cached_path(url), sep='\t')
    train, test = train_test_split(df, test_size=0.3, random_state=42)
    
    print(f'Saving to {out_path}.')
    
    save(train, os.path.join(out_path, 'train.tsv'))
    save(test, os.path.join(out_path, 'test.tsv'))