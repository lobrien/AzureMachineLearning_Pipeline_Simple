import argparse 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
from azureml.core import Run

# The dataset is specified at the pipeline definition level.

RANDOM_STATE = 42

parser = argparse.ArgumentParser()

parser.add_argument('--X_train_path', dest='X_train_path', required=True)
parser.add_argument('--X_test_path', dest='X_test_path', required=True)
parser.add_argument('--y_train_path', dest='y_train_path', required=True)
parser.add_argument('--y_test_path', dest='y_test_path', required=True)

args = parser.parse_args()

ds = Run.get_context().input_datasets['iris_baseline']

# Now the actual data prep (trivial)
df = ds.to_pandas_dataframe()
le = LabelEncoder()
le.fit(df['species'])
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,1:4], le.transform(df['species']), test_size=0.2, random_state=RANDOM_STATE)

# Write outputs as `PipelineData`

## Make directory for file
os.makedirs(os.path.dirname(args.X_train_path), exist_ok=True)
os.makedirs(os.path.dirname(args.X_test_path), exist_ok=True)
os.makedirs(os.path.dirname(args.y_train_path), exist_ok=True)
os.makedirs(os.path.dirname(args.y_test_path), exist_ok=True)

## And save the data
np.savetxt(args.X_train_path, X_train, delimiter=',')
np.savetxt(args.X_test_path, X_test, delimiter=',')
np.savetxt(args.y_train_path, y_train, delimiter=',')
np.savetxt(args.y_test_path, y_test, delimiter=',')

