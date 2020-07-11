import os, sys
import pandas as pd

from allennlp.common.file_utils import cached_path
from sklearn.model_selection import train_test_split

def save(df, fp):
    print(f'{fp}: {df.shape}')
    df.to_csv(fp, sep='\t')

if __name__ == '__main__':
    url = sys.argv[1]
    save_root = sys.argv[2] if len(sys.argv) > 2 else '/tmp'

    out_path = os.path.join(save_root, 'allenrank', 'data', url.split('/')[-1].split('.')[0].lower())
    os.makedirs(out_path, exist_ok=True)
    
    df = pd.read_csv(cached_path(url), sep='\t')
    train, test = train_test_split(df, test_size=0.3, random_state=42)
    train, valid = train_test_split(train, test_size=0.2, random_state=42)
    
    print(f'Saving to {out_path}.')
    
    save(train, os.path.join(out_path, 'train.tsv'))
    save(valid, os.path.join(out_path, 'valid.tsv'))
    save(test, os.path.join(out_path, 'test.tsv'))