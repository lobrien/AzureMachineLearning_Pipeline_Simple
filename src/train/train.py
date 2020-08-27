import argparse 
import numpy as np
from sklearn.naive_bayes import GaussianNB
from joblib import dump

parser = argparse.ArgumentParser()
parser.add_argument('--X_train_dir', dest='X_train_dir', required=True)
parser.add_argument('--y_train_dir', dest='y_train_dir', required=True)
parser.add_argument('--model_dir', dest='model_dir', required=True)

args = parser.parse_args()

print(f"X_train received as {args.X_train_dir}")

x_path = os.path.join(args.X_train_dir, "data.txt")
y_path = os.path.join(args.y_train_dir, "data.txt")

X_train = np.genfromtxt(x_path, delimiter=',')
y_train = np.genfromtxt(y_path, delimiter=',')

# Training (trivial)
model = GaussianNB()
model.fit(X_train, y_train)

model_path = os.path.join(args.model_dir, "model.pkl")
dump(model, model_path)