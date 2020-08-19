import argparse 
from sklearn.metrics import accuracy_score
from joblib import load
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', dest='model_path', required=True)
parser.add_argument('--X_test_path', dest='X_test_path', required=True)
parser.add_argument('--y_test_path', dest='y_test_path', required=True)

args = parser.parse_args()

model = load(args.model_path)
X_test = np.genfromtxt(args.X_test_path, delimiter=',')
y_test = np.genfromtxt(args.y_test_path, delimiter=',')

# Evaluation (trivial)
prediction = model.predict(X_test)
print(f'Accuracy: {accuracy_score(prediction, y_test):3f}')