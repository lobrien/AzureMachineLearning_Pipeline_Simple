# A Simple 3-Step AzureML Pipeline (Dataprep, Training, and Evaluation)

![Illustration of pipeline graph](./media/pipeline_graph.png)

This demonstrates how you create a multistep AzureML pipeline using a series of `PythonScriptStep` objects. 

In this case, the calculation is extremely trivial: predicting Iris species using scikit-learn's Gaussian Naive Bayes. This pipeline could be solved (very quickly) using this code: 

```python
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# These two lines become the data ingestion and dataprep steps 
df = pd.read_csv("iris.csv", header=None)
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,1:4], df.iloc[:,4:5], test_size=0.2, random_state=42)

# These two lines become the training step
model = GaussianNB()
model.fit(X_train, y_train.values.ravel())

# These two lines become the evaluation step
prediction = model.predict(X_test)
print(f'Accuracy: {accuracy_score(prediction, y_test):3f}')
```

The point of this notebook is to show the construction of the AzureML pipeline, not demonstrate any kind of complex machine learning. 

