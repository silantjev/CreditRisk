import numpy as np
import pandas as pd
from pathlib import Path
import logging

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
LOGDIR = ROOT / 'logs'

N_PROC = 7

# device for CatBoost, XGBClassifier:
DEVICE = "CPU"
# DEVICE = "GPU"

class TargetLoader():
    def __init__(self):
        self.y = pd.read_csv(DATA / 'train_target.csv', index_col=0)
        # self.y.rename(columns={'flag': 'target'}, inplace=True)

    def __call__(self, X):
        df = pd.DataFrame(index=X['id'].unique())
        
        df = df.join(self.y, how='left')
        assert not df.isna().any().any()
        return X.drop(['id'], axis=1), df['flag']


class Tuning:
    def __init__(self, name, n_j, with_sts=False, device=DEVICE):
        self.name = name
        self.with_sts = with_sts  # with STandard Scaling
        self.device = device
        self.set_log()
        self.target_loader = TargetLoader()
        if name == 'forest':
            self.forest(n_j)
        elif name == 'logistic':
            self.logistic(n_j)
        elif name == 'lightGBM':
            self.lightGBM(n_j)
        elif name == 'catboost':
            self.catboost(n_j)
        elif name == 'xgboost':
            self.xgboost(n_j)

    def set_log(self):
        self.logger = logging.getLogger(str(self.name))
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler(LOGDIR/ f'{self.name}.log', 'a')
        datefmt = '%Y-%m-%d %H:%M:%S'
        formatter = logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s', datefmt=datefmt)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        self.logger.addHandler(logging.StreamHandler())

    def load_aggregated(self, n_list=[0], aim='train'):
        if isinstance(n_list, int):
            n_list = [n_list]
        df_list = []
        for n in n_list:
            pq_path = DATA / 'aggregated' / f'data{n:02d}.pq'
            df = pd.read_parquet(pq_path)
            assert 'id' in df.columns
            self.logger.info(f"Loaded a train DataFrame of shape {df.shape} from {pq_path} (for {aim})")
            df_list.append(df)

        X = pd.concat(df_list)
        X, y = self.target_loader(X)
        assert 'id' not in X.columns

        return X, y

    def load_train(self, n_list):
        self.xt, self.yt = self.load_aggregated(n_list)

        if self.with_sts and self.name != 'forest':
            self.sts = StandardScaler()
            self.xt = self.sts.fit_transform(self.xt)

    def load_val(self, n_list):
        self.xv, self.yv = self.load_aggregated(n_list, aim='validation')
        if self.with_sts and self.name != 'forest':
            self.xv = self.sts.transform(self.xv)

    def load_shuffle(self, n_list=[9, 10, 11], shuffle=False):
        X, y = self.load_aggregated(n_list, aim='train and val')
        if shuffle:
            self.xt, self.xv, self.yt, self.yv = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
        else:
            self.xt, self.xv, self.yt, self.yv = train_test_split(X, y, test_size=0.2, shuffle=False)

        if self.with_sts and self.name != 'forest':
            self.sts = StandardScaler()
            self.xt = self.sts.fit_transform(self.xt)
            self.xv = self.sts.transform(self.xv)
    
    def forest(self, n_j=1):
        self.model = RandomForestClassifier(n_jobs=n_j, class_weight='balanced', n_estimators=160, max_depth=10, min_samples_split=5)
        self.params = {
                'n_estimators': np.arange(450, 581, 50), # 450 (160)
                'max_depth': np.arange(11, 13, 1), # 11 (10)
                # 'max_features': ['sqrt', 'log2', None], # 'sqrt' (default)
                'min_samples_leaf': [5, 6],  # 5
                # 'min_samples_split': [2, 3], # 2 (default)
            }

    def logistic(self, n_j):
        self.model = LogisticRegression(n_jobs=n_j, class_weight='balanced', solver='newton-cg', C=0.92)
        self.params = {
                'C': np.arange(0.1, 1.0, 0.01)  # 0.92
                # 'class_weight': ['balanced', None, {0: 0.03, 1: 0.97}],
                # 'solver': ['newton-cg', 'lbfgs']
            }

    def lightGBM(self, n_j):
        self.model = LGBMClassifier(n_jobs=n_j, force_col_wise=True, verbose=0,
                class_weight='balanced',
                n_estimators=110,
                max_depth=8,
                num_leaves=26,
                min_child_samples=26,
            )
        self.params = {
                'n_estimators': np.arange(100, 1000, 100), # 110
                'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.5],
                'max_depth': np.arange(7, 9, 1), # 8
                'num_leaves': np.arange(26, 27, 1), # 26
                # 'min_child_samples': np.arange(24, 29, 1), # 26
            }

    def catboost(self, n_j):
        self.model = CatBoostClassifier(thread_count=n_j, verbose=0,
                iterations=1750,
                depth=9,
                learning_rate=0.030,
                loss_function='Logloss',
                task_type=self.device.upper(),
                devices='0',
            )
        self.params = {
                'depth': np.arange(7, 10, 1), # 9
                'iterations': np.arange(1500, 1801, 50), # 1750
                'learning_rate': [0.02, 0.03, 0.04, 0.05, 0.06], # 0.03
            }

    def xgboost(self, n_j):
        self.model = XGBClassifier(n_jobs=n_j,
                n_estimators=1000,
                max_depth=8,
                learning_rate=0.03,
                device=self.device.lower(),
            )
        self.params = {
                # 'n_estimators': np.arange(90, 111, 10), #
                # 'learning_rate': [0.1, 0.3, 1, 2, 5, 10],
                'max_depth': np.arange(6, 9, 1), # 7
                'max_leaves': np.arange(16, 25, 1), # 20
            }

    def grid_search(self, n_j=N_PROC):
        gs = GridSearchCV(self.model, self.params, cv=3, scoring='roc_auc', n_jobs=n_j, refit=False, verbose=3)
        gs.fit(self.xt, self.yt)
        self.logger.info(f"Best parameters: {gs.best_params_}, roc auc = {gs.best_score_:.4f}")
        cv_res = gs.cv_results_

        for p, m, s in zip(cv_res['params'], cv_res['mean_test_score'], cv_res['std_test_score']):
            self.logger.info(f'{p} \u2014 mean = {m:.5f}, std = {s:.5f}')

    def train_and_val(self, val=True):
        self.logger.info(self.model.get_params())
        self.model.fit(self.xt, self.yt)
        if val:
            proba = self.model.predict_proba(self.xv)
            rocauc = roc_auc_score(self.yv, proba[:, 1])
            self.logger.info(f'{self.name}, ROC AUC on validation data : {rocauc:.4f}')


def tune_by_gs(name, j_model=1, j_gs=N_PROC):
    tun = Tuning(name, j_model)
    n_list = [3, 4, 5,]
    tun.load_train(n_list)
    tun.grid_search(j_gs)


def validation(name, j_model=N_PROC, full=False, with_sts=False):
    tun = Tuning(name, j_model, with_sts=with_sts)
    if full:
        vallist = [0, 1, 2]
        trainlist = [i for i in range(12) if i not in vallist]
        tun.load_train(trainlist)
        tun.load_val(vallist)
    else:
        tun.load_train([9,10,11])
        tun.load_val([8])
    tun.train_and_val()


def validation_shuffle(name, j_model=N_PROC, with_sts=False, shuffle=False):
    tun = Tuning(name, j_model, with_sts=with_sts)
    tun.load_shuffle([3], shuffle=shuffle)
    tun.train_and_val()


if __name__ == '__main__':
    # name = 'forest'
    # name = 'logistic'
    # name = 'lightGBM'
    name = 'catboost'
    # name = 'xgboost'

    if DEVICE.upper() == "CPU" and name in ['catboost', 'xgboost']:
        j_gs = 1
        j_model = N_PROC
    else:
        j_gs = N_PROC
        j_model = 1

    # tune_by_gs(name, j_gs=j_gs, j_model=j_model)
    # validation(name, j_model=N_PROC, with_sts=False)
    # validation(name, j_model=N_PROC, with_sts=True)
    validation(name, j_model=N_PROC, full=True, with_sts=False)
    # validation_shuffle(name, j_model=N_PROC, shuffle=False)
