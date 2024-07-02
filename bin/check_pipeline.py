import numpy as np
import pandas as pd
from pathlib import Path
import dill
import argparse

from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

# local
from data_agg import DataAggregator

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'

DEFAUT_MODEL = "catboost"
# DEFAUT_MODEL = "logistic"
DEFAULT_INPUT = str(DATA / 'train_data' / 'train_data_0.pq')
DEFAULT_OUTPUT = str(DATA / 'prediction_0.pq')
def parse_args():
    """Defines and parses command-line arguments."""
    parser = argparse.ArgumentParser(description=
    f"""
    Make and save pipline with a (pretrained) model:
    python3 final_pipeline.py --name <model name or path> [--input <input file>] [--output <output file>]
    Supported model names:
        forest (RandomForest)
        logistic
        lightGBM
        catboost
        xgboost
    """
    , formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--name', type=str, help='Name of model or path to a dill-pickle file with a pipeline', default=DEFAUT_MODEL)
    parser.add_argument('--input', type=str, help='Path to a data set (parquet or csv)', default=DEFAULT_INPUT)
    parser.add_argument('--output', type=str, help='Path to the output file', default=DEFAULT_OUTPUT)

    return parser.parse_args()

def load_pipe(name):
    path = Path(name)
    if not path.exists():
        path = ROOT / f'pipeline_{name}.pkl'
    assert path.exists(), f'pipeline with model {name} not found'

    with open(str(path), 'br') as f:
        model = dill.load(f)

    return model


def load_data(input_path):
    path = Path(input_path)
    assert path.exists(), f'File {input_path} not found'
    if path.suffix in ['.pq', '.parquet']:
        X = pd.read_parquet(path)
    elif path.suffix in ['.csv']:
        X = pd.read_csv(path)
    else:
        assert False, f'Wrong extension {path.suffix}'

    assert not X.isna().any().any()

    print(f'Dataset of shape {X.shape} loaded from {path}')

    target = pd.DataFrame(index=X['id'].unique())
    target['id'] = X['id'].unique()
    y = pd.read_csv(DATA / 'train_target.csv', index_col=0)
    print(f'Target loaded from train_target.csv')
    target = target.join(y, on=['id'], how='left')
    assert not target.isna().any().any()
    y = target['flag']
    assert 'id' in X.columns

    return X, y


def check_pipeline(name, input_path):
    pipe = load_pipe(name)

    X, y = load_data(input_path)

    proba = pipe.predict_proba(X)[:, 1]
    rocauc = roc_auc_score(y, proba)
    print(f'{name}, ROC AUC on test data : {rocauc:.4f}')

    predict = (proba > 0.5).astype('int8')

    return pd.DataFrame({'id': X['id'].unique(), 'proba': proba, 'predict': predict})


def save_predict(df, output_path):
    path = Path(output_path)
    if path.suffix in ['.pq', '.parquet']:
        df.to_parquet(path)
    else:
        df.to_csv(path, index=None)

    print(f'Probability and predictions are saved to {path}')


def main():
    args = parse_args()
    df = check_pipeline(name=args.name, input_path=args.input)
    save_predict(df, args.output)

if __name__ == '__main__':
    main()
