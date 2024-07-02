# import pickle
from collections import OrderedDict
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'

def save_cols(i=0):
    path = DATA / 'colunms.csv'
    if not path.exists():
        df = pd.read_parquet(DATA / 'train_data' / f'train_data_{i}.pq')
        cols = df.columns
        pd.DataFrame(cols).to_csv(path, index=None, header=None)
        print('column list saved to', path)

    return path

class MyOneHotEncoder:
    def __init__(self, drop_first=True, dtype='int8'):
        self.dtype = dtype
        self.val_dict = OrderedDict({
                'enc_loans_account_holder_type' : [0, 1, 2, 3, 4, 5, 6],
                'enc_loans_credit_status' : [0, 1, 2, 3, 4, 5, 6],
                'enc_loans_credit_type' : [0, 1, 2, 3, 4, 5, 6, 7],
                'enc_loans_account_cur': [0, 1, 2, 3],

                'pre_since_opened': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                'pre_since_confirmed': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
                'pre_pterm': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
                'pre_fterm': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                'pre_till_pclose': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                'pre_till_fclose': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                'pre_loans_credit_limit': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                'pre_loans_next_pay_summ': [0, 1, 2, 3, 4, 5, 6],
                'pre_loans_outstanding': [1, 2, 3, 4, 5],
                'pre_loans_total_overdue': [0, 1],
                'pre_loans_max_overdue_sum': [0, 1, 2, 3],
                'pre_loans_credit_cost_rate': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],

                'pre_loans5': [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 13, 16],
                'pre_loans530': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                'pre_loans3060': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                'pre_loans6090': [0, 1, 2, 3, 4],
                'pre_loans90': [2, 3, 8, 10, 13, 14, 19],
                'pre_util': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                'pre_maxover2limit': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                'pre_over2limit': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                      })

        if drop_first:
            self.val_dict = {k : v[1:] for k, v in self.val_dict.items()}

        self.ohe_cols = []
        for col, value_list in self.val_dict.items():
            for value in value_list:
                self.ohe_cols.append(f'{col}_{value}')

    def transform(self, df):
        df_ohe = pd.DataFrame(index=df.index, columns=['id']+self.ohe_cols)
        df_ohe['id'] = df['id']
        for col, value_list in self.val_dict.items():
            for value in value_list:
                df_ohe[f'{col}_{value}'] = (df[col] == value).astype(self.dtype)
        df_ohe = df_ohe[['id'] + self.ohe_cols].groupby('id').agg('sum').astype('int8')
        return df_ohe


class DataAggregator:
    def __init__(self):
        self.ohe = MyOneHotEncoder()

        path = save_cols()
        cols = pd.read_csv(path, index_col=None, header=None).iloc[:, 0]

        self.enc_paym_cols = [col for col in cols if col.startswith('enc_paym')]
        self.bin_cols = [col for col in cols if col.startswith('is_zero') or col.endswith('flag')]
        self.not_cols = ['not' + col for col in self.bin_cols]

        self.cols4sum = self.bin_cols
        self.cols4max = ['rn', 'pre_util', 'pre_over2limit', 'pre_maxover2limit', 'pre_since_opened'] + self.bin_cols + self.not_cols
        self.cols4mean = [col for col in cols if col.startswith('pre_')] + self.bin_cols + self.not_cols


        self.cols4sum.extend(['total1', 'total2', 'total3', 'credit_length'])
        self.cols4max.extend(['max_period12', 'frac1', 'frac2', 'frac3'])
        self.cols4mean.extend(['total1', 'total2', 'total3', 'credit_length', 'max_period12'])

        self.cols = list(cols.values)

        self.sum_cols = ["sum_" + col for col in self.cols4sum]
        self.max_cols = ["max_" + col for col in self.cols4max]
        self.mean_cols = ["mean_" + col for col in self.cols4mean]

    def prepare(self, df):
        enc_shifted_cols = [self.enc_paym_cols[i] for i in [11, 20, 24]]
        for col in enc_shifted_cols:
            df[col] = df[col].apply(lambda i: i -1)
        convert_dict = {col: 'int16' for col in df.columns if col != 'id'}
        df = df.astype(convert_dict)
        return df


    def max_paym(self, df):
        total1 = 0
        total2 = 0
        max_period12 = 0
        last_period12 = 0
        total3 = 0
        last_period3 = 0
        for col in self.enc_paym_cols:
            value = df[col]
            if value == 0:
                total3 += last_period3
                last_period3 = 0
                max_period12 = max(max_period12, last_period12)
                last_period12 = 0
            elif value == 3:
                last_period3 += 1
                max_period12 = max(max_period12, last_period12)
                last_period12 = 0
            else: # 1 or 2
                last_period12 += 1
                total3 += last_period3
                last_period3 = 0
                if value == 1:
                    total1 += 1
                else:
                    total2 += 1

        max_period12 = max(max_period12, last_period12)
        return pd.Series([total1, total2, max_period12, total3, 25 - last_period3], dtype=np.int16)

    def new_feat(self, df):
        df[['total1', 'total2', 'max_period12', 'total3', 'credit_length']] = df.apply(self.max_paym, axis=1)  # inplace
        df['frac1'] = df.apply(lambda x: x['total1'] / (1e-10 + x['credit_length']), axis=1).astype('float32')
        df['frac2'] = df.apply(lambda x: x['total2'] / (1e-10 + x['credit_length']), axis=1).astype('float32')
        df['frac3'] = df.apply(lambda x: x['total3'] / (1e-10 + x['credit_length']), axis=1).astype('float32')
        for col in self.bin_cols:
            df['not' + col] =  df.apply(lambda x: 1 - x[col], axis=1).astype('int16')

    def agg(self, df):
        dfa = pd.DataFrame(index=df['id'].unique())  # for df aggregated
        dfa['id'] = df['id'].unique()

        dfa.loc[:, self.cols4sum] = df[['id'] + self.cols4sum].groupby('id').agg('sum').astype('int16')
        rename_dict = {col: "sum_" + col for col in self.cols4sum}
        dfa.rename(columns=rename_dict, inplace=True)

        dfa.loc[:, self.cols4max] = df[['id'] + self.cols4max].groupby('id').agg('max')
        rename_dict = {col: "max_" + col for col in self.cols4max}
        dfa.rename(columns=rename_dict, inplace=True)

        dfa.loc[:, self.cols4mean] = df[['id'] + self.cols4mean].groupby('id').agg('mean').astype('float32')
        rename_dict = {col: "mean_" + col for col in self.cols4mean}
        dfa.rename(columns=rename_dict, inplace=True)
        return dfa

    def ohe_aggregate(self, df):
        df_ohe = self.ohe.transform(df)
        assert not df_ohe.isna().any().any()
        dfa = self.agg(df).join(df_ohe, how='left')
        return dfa

    def fit(self, *args, **kwargs):
        return self

    def transform(self, df, **kwargs):
        df = self.prepare(df)
        self.new_feat(df)
        df.drop(self.enc_paym_cols, axis=1, inplace=True)
        df = self.ohe_aggregate(df)
        df.drop(['id'], axis=1, inplace=True)
        return df

    def fit_transform(self, df, *args, **kwargs):
        return self.transform(df)


if __name__ == '__main__':
    pass

