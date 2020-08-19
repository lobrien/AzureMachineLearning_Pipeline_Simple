import argparse 
import numpy as np
from sklearn.naive_bayes import GaussianNB
from joblib import dump

parser = argparse.ArgumentParser()
parser.add_argument('--X_train_path', dest='X_train_path', required=True)
parser.add_argument('--y_train_path', dest='y_train_path', required=True)
parser.add_argument('--model_path', dest='model_path', required=True)

args = parser.parse_args()

X_train = np.genfromtxt(args.X_train_path, delimiter=',')
y_train = np.genfromtxt(args.y_train_path, delimiter=',')

# Training (trivial)
model = GaussianNB()
model.fit(X_train, y_train)

dump(model, args.model_path)