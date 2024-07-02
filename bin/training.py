import numpy as np
import pandas as pd
from pathlib import Path
import dill
import argparse

# local
from tuning import Tuning

ROOT = Path(__file__).resolve().parents[1]
MODELDIR = ROOT / 'models'

N_PROC = 7

# device for CatBoost, XGBClassifier:
DEVICE = "CPU"
# DEVICE = "GPU"

DEFAUT_MODEL = "catboost"
# DEFAUT_MODEL = "logistic"

def parse_args():
    """Defines and parses command-line arguments."""
    parser = argparse.ArgumentParser(description=
    f"""
    Train and save model:
    python3 training.py --name <model name> [--n-proc <n>] [--gpu] [--cpu] [--nosave]
    Supported model names:
        forest (RandomForest)
        logistic
        lightGBM
        catboost
        xgboost
    n-proc: n_job
    gpu: use GPU for catboost and xgboost
    cpu: use CPU for catboost and xgboost
    nosave: do not save the trained model
    """
    , formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--name', type=str, help='Name of model', default=DEFAUT_MODEL)
    parser.add_argument('--n-proc', type=int, help='Number of jobs', default=N_PROC)
    parser.add_argument('--gpu', action='store_true', help='Use GPU for catboost and xgboost')
    parser.add_argument('--cpu', action='store_true', help='Use CPU for catboost and xgboost')
    parser.add_argument('--nosave', action='store_true', help='Do not save the trained model')

    return parser.parse_args()

def train(name, j_model=N_PROC, device=DEVICE, nosave=True, with_sts=False):
    tun = Tuning(name, j_model, device=device, with_sts=with_sts)
    vallist = [0, 1, 2]
    trainlist = [i for i in range(12) if i not in vallist]
    tun.load_train(trainlist)
    vallist = [0]
    if vallist:
        tun.load_val(vallist)

    tun.train_and_val(val=bool(vallist))

    if nosave:
        return

    MODELDIR.mkdir(parents=True, exist_ok=True)
    path = MODELDIR / f'model_{tun.name}.pkl'

    with open(str(path), 'bw') as f:
        dill.dump(tun.model, f)

    tun.logger.info(f"Model saved to {path}")

def main():
    args = parse_args()
    device = DEVICE
    if args.gpu:
        device = 'GPU'
    if args.cpu:
        device = 'CPU'

    train(name=args.name, j_model=args.n_proc, device=device, nosave=args.nosave)

if __name__ == '__main__':
    main()
