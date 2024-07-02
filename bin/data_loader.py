import numpy as np
import pandas as pd
from pathlib import Path

# local
from data_agg import DataAggregator

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'

class DataLoader(DataAggregator):
    def __init__(self):
        super().__init__()
        self.y = pd.read_csv(DATA / 'train_target.csv', index_col=0)
        self.y.rename(columns={'flag': 'target'}, inplace=True)

    def load(self, n, prepare=True):
        assert 0 <= n < 12
        df = pd.read_parquet(DATA / 'train_data' / f'train_data_{n}.pq')
        # df = df.join(self.y, on='id', how='left')
        if prepare:
            df = self.prepare(df)
        return df

    def save_new_feat(self, n):
        df = self.load(n)
        self.new_feat(df)
        df.drop(self.enc_paym_cols, axis=1, inplace=True)

        dirpath = DATA / 'with_new_feat'
        dirpath.mkdir(parents=True, exist_ok=True)
        path = dirpath / f'part{n:02d}.pq'

        df.to_parquet(path)
        print("DataFrame with new features saved to", path)

    def load_with_new_feat(self, n):
        assert 0 <= n < 12
        path = DATA / 'with_new_feat' / f'part{n:02d}.pq'

        df = pd.read_parquet(path)
        if self.enc_paym_cols[0] in df.columns:
            df.drop(self.enc_paym_cols, axis=1, inplace=True)
        return df

    def split_target(self, dfa):
        y = dfa['target']
        dfa.drop(['target'], axis=1, inplace=True)
        return y
        
    def split(self, df):
        y = df['target']
        X = df.drop(['target'], axis=1, inplace=False)
        return X, y


def save_with_new_feat():
    dl = DataLoader()
    for n in range(12):
        dl.save_new_feat(n)


def make_samples(n, load_with_new_feat=True):
    if n == 'all':
        for n in range(12):
            make_samples(n, load_with_new_feat)
        return

    dl = DataLoader()
    print('Load data no.', n)
    if load_with_new_feat:
        df = dl.load_with_new_feat(n)
        assert 'id' in df.columns
    else:
        df = dl.load(n, prepare=True)
        dl.new_feat(df)
        df.drop(dl.enc_paym_cols, axis=1, inplace=True)
        assert 'id' in df.columns
    dfa = dl.ohe_aggregate(df)
    assert 'id' in dfa.columns

    dirpath = DATA / 'aggregated'
    dirpath.mkdir(parents=True, exist_ok=True)
    pq_path = dirpath / f'data{n:02d}.pq'
    dfa.to_parquet(pq_path)
    print('DataFrame saved to', pq_path)

if __name__ == '__main__':
    # save_with_new_feat()
    # make_samples('all', load_with_new_feat=True)
    make_samples('all', load_with_new_feat=False)

