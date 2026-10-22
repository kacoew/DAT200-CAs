"""CA3 - Classification of Wine. Group 6.

Narges Zareigolbaghi & Kachina Aschenberg Oewre

Trains the classifier chosen in section 7 of CA3_Group6.ipynb and writes the
submission that was uploaded to Kaggle. Run from this folder with dat200_env
active:

    python CA3_Group6.py

The model and its settings were selected in the notebook. This file contains
only the final training and the prediction behind the Kaggle file.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# --- data ---------------------------------------------------------------
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
features = [col for col in train.columns if col not in ("quality", "Id")]

# Cleaning, as in section 4 of the notebook: drop the row label, then drop
# exact duplicates. 835 rows become 760.
data = train.drop(columns=["Id"]).drop_duplicates()
X = data[features].to_numpy()
y = data["quality"].to_numpy()

# --- the chosen model, trained on every cleaned row ---------------------
# A random forest needs no scaling: a tree splits one feature at a time on a
# threshold, so rescaling a feature cannot change which split is best.
model = RandomForestClassifier(
    n_estimators=200, max_depth=5, random_state=1, n_jobs=-1
)
model.fit(X, y)
print(f"trained on {X.shape[0]} rows x {X.shape[1]} features")
print(f"accuracy on the training set: {model.score(X, y):.4f}")

# --- prediction and submission ------------------------------------------
# Submission format copied from the example code on the competition's Overview
# page: a column named "Quality", with the index named "ID" so that the row
# numbers 0..208 are written out as the ID column.
predictions = model.predict(test[features].to_numpy())
submission = pd.DataFrame(predictions, columns=["Quality"])
submission.index.name = "ID"
submission[["Quality"]].to_csv("submission.csv")

print(f"wrote submission.csv: {submission.shape[0]} rows")
print("predicted classes:", submission["Quality"].value_counts().to_dict())
