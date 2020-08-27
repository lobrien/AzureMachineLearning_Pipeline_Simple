import argparse 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
from azureml.core import Run

# The dataset is specified at the pipeline definition level.

RANDOM_STATE = 42

parser = argparse.ArgumentParser()

parser.add_argument('--X_train_dir', dest='X_train_dir', required=True)
parser.add_argument('--X_test_dir', dest='X_test_dir', required=True)
parser.add_argument('--y_train_dir', dest='y_train_dir', required=True)
parser.add_argument('--y_test_dir', dest='y_test_dir', required=True)

args = parser.parse_args()

ds = Run.get_context().input_datasets['iris_baseline']

# Now the actual data prep (trivial)
df = ds.to_pandas_dataframe()
le = LabelEncoder()
le.fit(df['species'])
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,1:4], le.transform(df['species']), test_size=0.2, random_state=RANDOM_STATE)

# Write outputs as `OutputFileDatasetConfig`
x_train_fname = os.path.join(args.X_train_dir, "data.txt")
x_test_fname = os.path.join(args.X_test_dir, "data.txt")
y_train_fname = os.path.join(args.y_train_dir, "data.txt")
y_test_fname = os.path.join(args.y_test_dir, "data.txt")

print(f"X_train written to {x_train_fname}")

## And save the data
np.savetxt(x_train_fname, X_train, delimiter=',')
np.savetxt(x_test_fname, X_test, delimiter=',')
np.savetxt(y_train_fname, y_train, delimiter=',')
np.savetxt(y_test_fname, y_test, delimiter=',')

