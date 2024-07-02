import numpy as np
import pandas as pd
from pathlib import Path
import dill
import argparse
from sklearn.pipeline import Pipeline

# local
from data_agg import DataAggregator
from check_pipeline import load_data

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'

MODELDIR = ROOT / 'models'

DEFAUT_MODEL = "catboost"
# DEFAUT_MODEL = "logistic"

def parse_args():
    """Defines and parses command-line arguments."""
    parser = argparse.ArgumentParser(description=
    f"""
    Make and save pipline with a (pretrained) model:
    python3 final_pipeline.py --name <model name or path> [--nosave] [--test-file <path>]
    Supported model names:
        forest (RandomForest)
        logistic
        lightGBM
        catboost
        xgboost
    nosave: do not save the trained model
    """
    , formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--name', type=str, help='Name of model or path to a dill-pickle file with a model', default=DEFAUT_MODEL)
    parser.add_argument('--nosave', action='store_true', help='Do not save the trained model')
    parser.add_argument('--test-file', type=str, help='Path to the test data', default='')

    return parser.parse_args()



def load_model(name):
    path = Path(name)
    if not path.exists():
        path = MODELDIR / f'model_{name}.pkl'
    assert path.exists(), f'Model {name} not found'

    with open(str(path), 'br') as f:
        model = dill.load(f)

    return model


def make_pipeline(name, nosave, input_path):
    model = load_model(name)
    agg = DataAggregator()
    pipe = Pipeline(steps=[('preprocess', agg), ('classification', model)])

    # input_path = str(DATA / 'train_data' / 'train_data_0.pq')



    if not nosave:
        path = ROOT / f'pipeline_{name}.pkl'
        with open(str(path), 'bw') as f:
            dill.dump(pipe, f)
        print(f"Pipeline saved to {path}")

    if input_path:
        check_pipeline(pipe, input_path)

def check_pipeline(pipe, input_path):

    X, y = load_data(input_path)

    proba = pipe.predict_proba(X)[:, 1]
    rocauc = roc_auc_score(y, proba)
    print(f'ROC AUC on test data : {rocauc:.4f}')

def main():
    args = parse_args()

    make_pipeline(name=args.name, nosave=args.nosave, input_path=args.test_file)

if __name__ == '__main__':
    main()
