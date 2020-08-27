import argparse 
from sklearn.metrics import accuracy_score
from joblib import load
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', dest='model_dir', required=True)
parser.add_argument('--X_test_dir', dest='X_test_dir', required=True)
parser.add_argument('--y_test_dir', dest='y_test_dir', required=True)

args = parser.parse_args()

model = load(os.path.join(args.model_dir, "model.pkl"))
X_test = np.genfromtxt(os.path.join(args.X_test_dir, "data.txt"), delimiter=',')
y_test = np.genfromtxt(os.path.join(args.y_test_dir, "data.txt"), delimiter=',')

# Evaluation (trivial)
prediction = model.predict(X_test)
print(f'Accuracy: {accuracy_score(prediction, y_test):3f}')