#!/usr/bin/env python
# coding: utf-8

# # **CA5** - Predicting Cognitive Health from Lifestyle Habits
# 
# # **Group 6** - Narges Zareigolbaghi & Kachina Aschenberg Øwre 
# # **Kaggle names** - nargeszareigolbaghi & kachasch

# ## 1. Imports
# 
# - numpy and pandas: data, matplotlib and seaborn: plotting
# - scikit-learn tools for preprocessing, pipelines, cross-validation, hyperparameter tuning and the MAE metric
# - the models: Ridge, RandomForestClassifier and RandomForestRegressor
# - `RANDOM_STATE = 42`, used everywhere so the results are the same every time the notebook is run

# In[1]:


# Imports for data analysis and visualization
import numpy as np  # numerical computing
import pandas as pd  # dataframes and series
import matplotlib.pyplot as plt  # plotting
import seaborn as sns  # statistical plotting

# Imports for preprocessing, pipelines and model evaluation
from sklearn.pipeline import Pipeline  # chains preprocessing and model
from sklearn.compose import ColumnTransformer  # different preprocessing for numeric and text columns
from sklearn.impute import SimpleImputer  # fills in missing values
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.model_selection import KFold, GridSearchCV, cross_val_predict
from sklearn.metrics import mean_absolute_error  # the Kaggle metric
from sklearn.base import clone  # fresh, untrained copy of a model

# Imports for models
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor

# Constants
RANDOM_STATE = 42  # same random state everywhere, so results can be reproduced


# ## 2. Reading the data into pandas DataFrames
# 
# - `id` is the index, so it's feature and the submission gets the right ids
# - We check the size of both sets: train has 367,500 rows (15 features and `target`), test has 150,000 rows (15 features)
# - The first rows already show missing values (`NaN`) and an impossible value (negative water intake), which we look at in section 3

# In[2]:


# Read the data; id becomes the index, so it is never used as a feature
df_train = pd.read_csv("train.csv", index_col="id")
df_test = pd.read_csv("test.csv", index_col="id")

# Size of the data: (rows, columns)
print("Train:", df_train.shape)
print("Test: ", df_test.shape)

# First rows of the training data
df_train.head()


# ## 3. Data exploration and visualisation before cleaning
# 
# This is to find out what is wrong with it and which features are related to the target. Each part ends with a short conclusion.
# - 3a: data types, summary statistics and missing values in train and test
# - 3b: the distribution of the target, and how many participants have the maximum score of 30
# - 3c: histograms of all numeric features, and the smallest and largest values in train and test
# - 3d: the number of duplicate rows
# - 3e: a correlation heatmap, and the correlation of each feature with the target
# - 3f: the target by smoking status

# In[3]:


# Section 3a - a look at the data

# 1) Data types and how many values are NOT missing in each column
df_train.info()

# 2) Summary statistics; .T turns the table so each feature is one row
display(df_train.describe().T)

# 3) Missing values per column in train and test, side by side
missing = pd.DataFrame({"train_missing": df_train.isna().sum(),
                        "test_missing": df_test.isna().sum()})
display(missing)

# 4) Share of rows in train with at least one missing value
print("Rows with at least one missing value:", round(df_train.isna().any(axis=1).mean(), 3))


# **Conclusion 3a:** The training set has 367,500 rows with 15 features and the target; the test set has 150,000 rows.
# All features are numeric except `smoking_status`, which is text (Never / Former / Current), and the target is an integer.
# Every feature in train has missing values, from about 5,300 (`social_media_hrs`) to 28,600 (`screen_time_recreational_hrs`) per column, while the test set has none.
# More than half of the training rows have at least one missing value, so we cannot simply drop them; we fill them in with an imputer inside the pipeline.
# `describe()` also shows impossible minimum and maximum values (e.g. age −129 and 218, sleep quality −11 and 28), which we look at in 3c.

# In[4]:


# Section 3b - The target: how many participants have each score?
counts = df_train["target"].value_counts().sort_index()
counts.plot.bar(figsize=(10, 4), color="steelblue")
plt.title("Number of participants per cognitive score")
plt.xlabel("Cognitive score (target)")
plt.ylabel("Number of participants")
plt.show()

print("Share with the maximum score 30:", round((df_train["target"] == 30).mean(), 3))

# Baseline: how far off are we on average if we always predict 28? A useful model must beat this MAE
print("Baseline MAE if we always predict 28:", round((df_train["target"] - 28).abs().mean(), 3))


# **Conclusion 3b:** The scores range from 17 to 30, so no one scores below 17 even though the scale goes down to 5.
# Most participants score high (mean 27.8, median 28), and 23% have exactly 30, the ceiling of the scale.
# This matters for the models: a regression model can predict values above 30, which are always wrong, and the large group at 30 is the reason for the two-step pipeline C.
# As a reference, always predicting 28 gives an MAE of 1.43, so our models must do clearly better than that.

# In[5]:


# Section 3c - Distribution of each numeric feature
num_cols = [c for c in df_test.columns if c != "smoking_status"]

df_train[num_cols].hist(bins=50, figsize=(16, 12))
plt.suptitle("Feature distributions in train (before cleaning)", fontsize=16)
plt.tight_layout()
plt.show()

# Smallest and largest values in train and test, side by side
ranges = pd.DataFrame({"train_min": df_train[num_cols].min(),
                       "test_min": df_test[num_cols].min(),
                       "test_max": df_test[num_cols].max(),
                       "train_max": df_train[num_cols].max()})
display(ranges)


# **Conclusion 3c:** All 14 numeric features in train contain a small number of impossible values far outside the realistic range, which makes the x-axes of the histograms much wider than the bars (e.g. age from −129 to 218, blood pressure from −7 to 259, negative hours of sleep and screen time).
# The test set only contains realistic values (e.g. age 18–90, scores 1–10), so we use the test ranges as limits when cleaning.
# Apart from the errors, most features look roughly bell-shaped, while `cardio_mins_wk`, `alcohol_drinks_wk` and `novel_learning_hrs_wk` are right-skewed: most people do a little and a few do a lot.

# In[6]:


# Section 3d - Duplicate rows (id is the index, so only the values are compared)
n_duplicates = df_train.duplicated().sum()
print("Duplicate rows in train:", n_duplicates)
print("Share of train:", round(n_duplicates / len(df_train), 3))


# **Conclusion 3d:** 17,500 rows (4.8% of train) are exact copies of other rows, leaving 350,000 unique rows.
# We remove them, because copies add no new information and give some participants double weight.
# They can also make cross-validation too optimistic: if a row is in the training fold and its copy in the validation fold, the model is tested on data it has already seen.

# In[7]:


# Section 3e - Correlation between all numeric columns, including the target
corr = df_train.corr(numeric_only=True)

plt.figure(figsize=(14, 11))
sns.heatmap(corr, cmap="coolwarm", center=0, annot=True, fmt=".2f", square=True)
plt.title("Correlation heatmap (before cleaning)")
plt.show()

# Correlation of each feature with the target, from most negative to most positive
display(corr["target"].drop("target").sort_values().to_frame("corr_with_target"))


# **Conclusion 3e:** The bottom row of the heatmap shows how each feature relates to the target.
# Cardio (+0.35), sleep quality (+0.26), novel learning (+0.17), sleep duration (+0.15) and social connection (+0.11) go with a higher score, while blood pressure (−0.32), ultra-processed diet (−0.23), chronic stress (−0.23) and age (−0.20) go with a lower score.
# Water intake, omega-3, alcohol, screen time and social media have a correlation of almost exactly 0 with the target, so they may be noise; we test this later.
# Most features are not correlated with each other. The exceptions match the data description: screen time and social media (+0.47, social media is part of screen time), age and blood pressure (+0.32), and stress with worse sleep quality (−0.31) and shorter sleep (−0.23).
# The impossible values found in 3c weaken all these numbers, so we check the correlations again after cleaning.

# In[8]:


# Section 3f - Cognitive score by smoking status
order = ["Never", "Former", "Current"]

sns.boxplot(data=df_train, x="smoking_status", y="target", order=order)
plt.title("Cognitive score by smoking status")
plt.xlabel("Smoking status")
plt.ylabel("Cognitive score (target)")
plt.show()

# Mean score, number of participants and share of participants (of those who answered) per group
smoking_summary = df_train.groupby("smoking_status")["target"].agg(["mean", "count"]).loc[order]
smoking_summary["share"] = smoking_summary["count"] / smoking_summary["count"].sum()
display(smoking_summary.round(3))


# **Conclusion 3f:** Most participants have never smoked (about 65% of those who answered).
# The score drops with smoking: the mean is 28.1 for never-smokers, 27.7 for former smokers and 26.8 for current smokers, and the boxes in the boxplot move down in the same order (median 28, 28 and 27).
# Never-smokers also reach the maximum score of 30 more often, since their box goes all the way up to 30.
# `smoking_status` is therefore a useful feature. It is text, so we one-hot encode it, and its missing values are filled in by an imputer in the pipeline.

# ## 4. Data cleaning
# 
# - Remove the 17,500 exact duplicate rows
# - Use the range of the clean test set as the valid range of each feature
# - Turn values outside this range into missing values (NaN) instead of dropping the rows, so we keep the good values in those rows
# - The missing values are not filled in here; that is done by an imputer inside the pipeline (section 6)

# In[9]:


# Section 4 - Data cleaning

# 1) Remove exact duplicate rows
print("Rows before:", len(df_train))
df_train = df_train.drop_duplicates()
print("Rows after removing duplicates:", len(df_train))

# 2) Valid range of each numeric feature = the range in the clean test set
lower = df_test[num_cols].min()
upper = df_test[num_cols].max()

# 3) Find impossible values (True where a value is outside the valid range)
outside = (df_train[num_cols] < lower) | (df_train[num_cols] > upper)

# 4) Turn impossible values into missing values (NaN); the imputer fills them in later
df_train[num_cols] = df_train[num_cols].mask(outside)

# 5) Check the result
summary = pd.DataFrame({"impossible_values": outside.sum(),
                        "missing_after_cleaning": df_train[num_cols].isna().sum()})
display(summary)
print("Rows with at least one impossible value:", outside.any(axis=1).sum())
print("Missing smoking_status:", df_train["smoking_status"].isna().sum())


# **Conclusion 4:** We removed the 17,500 duplicate rows, which leaves 350,000 rows.
# As the valid range of each feature we used the range of the clean test set. Between 1,279 and 1,702 values per feature were outside it, and 22,084 rows (6.3%) had at least one impossible value.
# We set these impossible values to missing (NaN) instead of dropping the rows, because the other values in those rows are still good.
# All missing values (6,700–28,900 per feature, plus 24,112 in `smoking_status`) are filled in by an imputer inside the pipeline (section 6), so during cross-validation the imputer only learns from the training folds.
# The test set is not changed.

# ## 5. Data exploration and visualisation after cleaning
# 
# Now, we check that the cleaning worked, and look at the data again without the impossible values.
# - 5a: histograms of all numeric features after cleaning
# - 5b: a correlation heatmap after cleaning, and the correlation with the target before and after cleaning

# In[10]:


# Section 5a - Histograms after cleaning: all values should now be realistic
df_train[num_cols].hist(bins=50, figsize=(16, 12))
plt.suptitle("Feature distributions in train (after cleaning)", fontsize=16)
plt.tight_layout()
plt.show()


# **Conclusion 5a:** After cleaning, all features lie within realistic ranges (e.g. age 18–90, scores 1–10, percentages 0–100).
# Now we can see the real shapes: sleep duration, water intake, blood pressure and ultra-processed diet are roughly bell-shaped, while cardio, alcohol, novel learning and social media are right-skewed (most people do a little, a few do a lot).
# The small peaks at the limits (age 18, cardio 500, diet 0 and 100) are not errors; the simulation caps the values there.

# In[11]:


# Section 5b - Correlation heatmap after cleaning
corr_clean = df_train.corr(numeric_only=True)

plt.figure(figsize=(14, 11))
sns.heatmap(corr_clean, cmap="coolwarm", center=0, annot=True, fmt=".2f", square=True)
plt.title("Correlation heatmap (after cleaning)")
plt.show()

# Correlation with the target before and after cleaning
compare = pd.DataFrame({"before_cleaning": corr["target"],
                        "after_cleaning": corr_clean["target"]}).drop("target")
display(compare.sort_values("after_cleaning").round(3))


# **Conclusion 5b:** Cleaning made all important correlations with the target stronger, because the impossible values were random noise that hid the patterns (e.g. cardio from +0.35 to +0.41, blood pressure from −0.32 to −0.39).
# The strongest features are cardio (+0.41), blood pressure (−0.39), sleep quality (+0.30), ultra-processed diet (−0.27), chronic stress (−0.27), age (−0.23), novel learning (+0.20), sleep duration (+0.18) and social connection (+0.13).
# Water intake, omega-3, alcohol, screen time and social media still have a correlation of 0.00 with the target. This fits the data description, which does not list them among the penalties and bonuses, and the description calls water intake a noise variable. We keep them for now and test whether removing them helps in section 7a.
# Some features are related to each other (screen time and social media +0.67, stress and sleep quality −0.44, age and blood pressure +0.45), which is something to keep in mind for the linear model.

# ## 6. Data preprocessing
# 
# Prepare the data for the models. All preprocessing is put inside scikit-learn pipelines, so during cross-validation it only learns from the training folds (no data leakage).
# - Split the cleaned training data into the features `X` and the target `y`
# - Numeric features: fill in missing values with the median and standardise them (`SimpleImputer` + `StandardScaler`)
# - `smoking_status`: fill in missing values with the most common value and one-hot encode it (`SimpleImputer` + `OneHotEncoder`)
# - Combine both in one `ColumnTransformer`, and set up 5-fold cross-validation (`KFold`)
# - The training set is large, so we tune the models on a random sample of 50,000 rows and train the final model on all rows

# In[12]:


# Section 6 - Preprocessing

# 1) Features (X) and target (y) from the cleaned training data
X = df_train.drop(columns="target")
y = df_train["target"]
cat_cols = ["smoking_status"]  # the only text column; num_cols (the 14 numeric columns) is from 3c

# 2) Numeric columns: fill in missing values with the median, then standardise
num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler())])

# 3) Text column: fill in missing values with the most common value, then one-hot encode
cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                     ("onehot", OneHotEncoder(handle_unknown="ignore"))])

# 4) Use the right pipeline for each group of columns
preprocess = ColumnTransformer([("num", num_pipe, num_cols),
                                ("cat", cat_pipe, cat_cols)])

# 5) 5-fold cross-validation with the same folds for every model
cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# 6) Random sample of 50,000 rows for tuning (grid search on all 350,000 rows takes too long)
X_tune = X.sample(n=50_000, random_state=RANDOM_STATE)
y_tune = y.loc[X_tune.index]

# Check what the preprocessing produces for the tuning sample
X_tune_prepared = preprocess.fit_transform(X_tune)
print("X:", X.shape, "| X_tune:", X_tune.shape, "| after preprocessing:", X_tune_prepared.shape)
print("Missing values after preprocessing:", np.isnan(X_tune_prepared).sum())
print("Output column names:", list(preprocess.get_feature_names_out()))
print("Mean target - all rows:", round(y.mean(), 3), "| tuning sample:", round(y_tune.mean(), 3))
print("Share with 30 - all rows:", round((y == 30).mean(), 3), "| tuning sample:", round((y_tune == 30).mean(), 3))


# **Conclusion 6:** The preprocessing turns the 15 original columns into 17 numeric columns (14 standardised numeric features and 3 one-hot columns for smoking status), with no missing values left.
# It is the first step of every model pipeline, so during cross-validation the medians, means and standard deviations are learned only from the training folds.
# The 50,000-row tuning sample has the same mean score (27.82) and the same share of 30s (22.9%) as all 350,000 rows, so it represents the full training set well.

# ## 7a. Pipeline A1 - Ridge regression
# 
# We now have to build our first model: a regression pipeline based on linear regression, which predicts the score directly.
# - The pipeline is: preprocessing → `PolynomialFeatures` → `Ridge` (linear regression with regularisation)
# - `PolynomialFeatures` adds squared and combined features, so the linear model can also follow curved relationships
# - We tune the polynomial degree and the regularisation strength `alpha` with `GridSearchCV`, using 5-fold cross-validation and MAE as the score, and compare the result with the baseline of always predicting 28 (MAE 1.43)
# - Feature selection: we test the same pipeline without the five features that showed no relation to the target in section 5b, and use the selected features in the rest of the notebook
# - Finally, we plot the predictions against the true scores and show what the model has learned

# In[13]:


# Section 7a.1 - Build pipeline A1: preprocessing -> polynomial features -> Ridge

# 1) Pipeline: preprocessing -> polynomial features -> Ridge
pipe_a = Pipeline([("prep", preprocess),
                   ("poly", PolynomialFeatures(include_bias=False)),
                   ("model", Ridge())])

# 2) Hyperparameters to try (step name + two underscores + parameter name)
param_grid_a = {"poly__degree": [1, 2, 3],
                "model__alpha": [1, 100, 1000, 10000, 100000]}


# In[14]:


# Section 7a.2 - Grid search for pipeline A1

# 3) Grid search with 5-fold CV. scikit-learn always maximises, so MAE is given as a negative number.
#    n_jobs=2: degree 3 makes over 1,000 features, so only two fits run at the same time to save memory
gs_a = GridSearchCV(pipe_a, param_grid_a, cv=cv, scoring="neg_mean_absolute_error", n_jobs=2)
gs_a.fit(X_tune, y_tune)

# 4) Best result
mae_a = -gs_a.best_score_
print("Best hyperparameters:", gs_a.best_params_)
print("CV MAE pipeline A1 (all features):", round(mae_a, 3))
print("Baseline MAE (always predict 28):", round((y_tune - 28).abs().mean(), 3))
print("Number of features after PolynomialFeatures:", gs_a.best_estimator_.named_steps["poly"].n_output_features_)

# 5) CV MAE for every combination: rows = alpha, columns = polynomial degree
results_a = pd.DataFrame(gs_a.cv_results_)
table_a = -results_a.pivot(index="param_model__alpha", columns="param_poly__degree", values="mean_test_score")
display(table_a.round(3))

table_a.plot(marker="o", logx=True, figsize=(7, 4))
plt.title("Pipeline A: CV MAE for each alpha and polynomial degree")
plt.xlabel("alpha (Ridge regularisation strength, log scale)")
plt.ylabel("CV MAE (lower is better)")
plt.legend(title="Polynomial degree")
plt.show()


# In[15]:


# Section 7a.3 - Feature selection: remove the 5 features that showed no relation to the target in 5b
noise_cols = ["water_intake_liters", "diet_omega3_freq", "alcohol_drinks_wk",
              "screen_time_recreational_hrs", "social_media_hrs"]
num_cols_sel = [c for c in num_cols if c not in noise_cols]

# Same preprocessing as in section 6, but only for the selected columns (all other columns are dropped)
preprocess_sel = ColumnTransformer([("num", num_pipe, num_cols_sel),
                                    ("cat", cat_pipe, cat_cols)])

# Same pipeline and grid search as above, now without the noise features
pipe_a_sel = Pipeline([("prep", preprocess_sel),
                       ("poly", PolynomialFeatures(include_bias=False)),
                       ("model", Ridge())])
gs_a_sel = GridSearchCV(pipe_a_sel, param_grid_a, cv=cv, scoring="neg_mean_absolute_error", n_jobs=2)
gs_a_sel.fit(X_tune, y_tune)
mae_a_sel = -gs_a_sel.best_score_

comparison_a = pd.DataFrame({"original_features": [len(num_cols) + 1, len(num_cols_sel) + 1],
                             "best_params": [gs_a.best_params_, gs_a_sel.best_params_],
                             "cv_mae": [mae_a, mae_a_sel]},
                            index=["all features", "without noise features"])
display(comparison_a.round(3))

# Out-of-fold predictions of the best Ridge model (same 5 folds), used again in section 10
pred_a = cross_val_predict(gs_a_sel.best_estimator_, X_tune, y_tune, cv=cv, n_jobs=2)
print("Check - MAE of the out-of-fold predictions:", round(mean_absolute_error(y_tune, pred_a), 3))


# In[16]:


# Section 7a.4 - Visualisations of pipeline A1

# Two small helper functions, used again for the other pipelines

def plot_pred_vs_true(pred, title, ax=None):
    """Boxplot of the predicted score for each true score, with the line of a perfect prediction."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(x=y_tune, y=np.asarray(pred, dtype=float), native_scale=True, color="steelblue",
                fliersize=1, ax=ax)
    ax.plot([17, 30], [17, 30], color="red", linestyle="--", label="perfect prediction")
    ax.set_title(title)
    ax.set_xlabel("True score")
    ax.set_ylabel("Predicted score")
    ax.legend()


def plot_effects(model, title):
    """Predicted score for a typical participant when ONE feature changes and the rest stay typical."""
    # Typical participant: the median of every numeric feature and the most common smoking status
    typical = X_tune.median(numeric_only=True).to_frame().T
    typical["smoking_status"] = X_tune["smoking_status"].mode()[0]
    typical = typical[X_tune.columns]  # same column order as the training data

    features_to_show = ["cardio_mins_wk", "blood_pressure_sys", "sleep_quality_score", "chronic_stress_score"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    for ax, feature in zip(axes, features_to_show):
        values = np.linspace(X_tune[feature].min(), X_tune[feature].max(), 100)
        participants = pd.concat([typical] * len(values), ignore_index=True)
        participants[feature] = values  # only this feature changes
        ax.plot(values, model.predict(participants))
        ax.axhline(30, color="grey", linestyle="--", linewidth=1)  # the ceiling
        ax.set_title(feature)
        ax.set_xlabel(feature)
    axes[0].set_ylabel("Predicted score")
    fig.suptitle(title)
    plt.tight_layout()


# 1) How well does it predict? (out-of-fold predictions from 7a.3)
plot_pred_vs_true(pred_a, "Pipeline A1 (Ridge): predicted score for each true score")
plt.show()

# 2) What did it learn?
plot_effects(gs_a_sel.best_estimator_, "Pipeline A1 (Ridge): predicted score when one feature changes")
plt.show()


# **Conclusion 7a:** Pipeline A1 (Ridge regression with polynomial features) clearly beats the baseline: with all features it reaches a cross-validated MAE of 0.919 (polynomial degree 3, alpha = 100), compared with 1.434 when we always predict 28.
# 
# **Polynomial degree and alpha.** The polynomial degree matters most: degree 1 (a straight line) gives 1.034, degree 2 gives 0.939 and degree 3 gives 0.919. The relationship between the features and the score is therefore not linear, for example because the score cannot go above 30. Alpha makes almost no difference up to 1,000 and makes the model worse above that (0.948 at 10,000 and 1.024 at 100,000 for degree 3). With 40,000 training rows in each fold, the model does not overfit, so it needs little regularisation, and a too strong penalty makes it too flat.
# 
# **Feature selection.** Removing the five features that showed no relation to the target in 5b (water intake, omega-3, alcohol, screen time and social media) lowers the MAE to 0.910 (degree 3, alpha = 1). With degree 3, the 15 original features become 1,139 columns, and many of them are combinations with these noise features. From here on, all pipelines use the 10 selected features.
# 
# **What the plots show.** Imputation, scaling, one-hot encoding, polynomial features and Ridge are all steps of one pipeline, so they are trained together in every fold. In the predicted-vs-true boxplot the boxes rise with the true score, but they are flatter than the red line: participants with low true scores get too high predictions, and participants with 30 get around 29. The model pulls its guesses towards the middle, because most participants score 26–30 and extreme scores are rare; this is the weakness that pipeline C tries to fix. The effect curves show what the model learned for a typical participant: more cardio raises the predicted score and flattens just below 30, so the polynomial features let the model follow the ceiling. Blood pressure hardly matters up to about 140 and then lowers the score steeply, and more stress lowers it steadily. Sleep quality changes the prediction very little when the other features are held fixed, although it correlates with the score (+0.30); part of its effect is probably shared with stress and cardio. These curves describe the model, not cause and effect.

# ## 7b. Pipeline A2 - Gradient boosting regression
# 
# We now have to test a second regression model that is not linear: gradient boosting, an ensemble of many small decision trees.
# - Each new tree is trained to correct the errors of the trees before it
# - We use `loss="absolute_error"`, so the model is trained on the same measure as the Kaggle score (MAE)
# - It uses the selected features from section 7a
# - We tune the learning rate and the tree depth with `GridSearchCV` (5-fold CV, MAE)
# - Finally, we plot the predictions, what the model has learned and which features it uses most

# In[17]:


# Section 7b.1 - Build pipeline A2: preprocessing (selected features) -> gradient boosting,
# trained directly on the absolute error (the Kaggle metric)
pipe_gbr = Pipeline([("prep", preprocess_sel),
                     ("model", GradientBoostingRegressor(loss="absolute_error", n_estimators=300,
                                                         random_state=RANDOM_STATE))])
param_grid_gbr = {"model__learning_rate": [0.05, 0.1],
                  "model__max_depth": [3, 5]}


# In[18]:


# Section 7b.2 - Grid search for pipeline A2

gs_gbr = GridSearchCV(pipe_gbr, param_grid_gbr, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1)
gs_gbr.fit(X_tune, y_tune)
mae_gbr = -gs_gbr.best_score_
print("Best hyperparameters:", gs_gbr.best_params_)
print("CV MAE gradient boosting:", round(mae_gbr, 3), "| Ridge without noise features:", round(mae_a_sel, 3))

# CV MAE for every combination: rows = learning rate, columns = tree depth
results_gbr = pd.DataFrame(gs_gbr.cv_results_)
display((-results_gbr.pivot(index="param_model__learning_rate", columns="param_model__max_depth",
                            values="mean_test_score")).round(3))

# Out-of-fold predictions of the best model (same 5 folds), used again in sections 9 and 10
pred_gbr = cross_val_predict(gs_gbr.best_estimator_, X_tune, y_tune, cv=cv, n_jobs=-1)


# In[19]:


# Section 7b.3 - Visualisations of pipeline A2

# 1) How well does it predict? (out-of-fold predictions from 7b.2)
plot_pred_vs_true(pred_gbr, "Pipeline A2 (gradient boosting): predicted score for each true score")
plt.show()

# 2) What did it learn? The same curves as for Ridge in 7a.4
plot_effects(gs_gbr.best_estimator_, "Pipeline A2 (gradient boosting): predicted score when one feature changes")
plt.show()

# 3) Feature importance: how much each feature helped the trees reduce the error
names = pd.Series(gs_gbr.best_estimator_["prep"].get_feature_names_out())
names = names.str.replace("num__", "").str.replace("cat__", "")
importance = pd.Series(gs_gbr.best_estimator_["model"].feature_importances_, index=names).sort_values()
importance.plot.barh(figsize=(8, 5), color="steelblue")
plt.title("Pipeline A2: feature importance in the gradient boosting model")
plt.xlabel("Importance (share of the total error reduction)")
plt.show()


# **Conclusion 7b:** Gradient boosting builds many small decision trees, where each new tree corrects the errors of the trees before it. We use `loss="absolute_error"`, so the model is trained on the same measure as the Kaggle score (MAE). The best version (learning rate 0.1, tree depth 5, 300 trees) has a CV MAE of 0.898, better than the best Ridge model (0.910). All four combinations lie between 0.898 and 0.905, so the model is not very sensitive to these two hyperparameters.
# 
# **What the plots show.** The predicted-vs-true boxplot looks much like the one for Ridge: the predictions follow the true score but are pulled towards the middle, while the participants with 30 get slightly higher predictions than with Ridge. The effect curves are steps instead of smooth lines, because every tree splits the data at fixed values. More cardio raises the prediction until it reaches exactly 30 at about 350–400 minutes per week, so the trees follow the ceiling without any polynomial features, and more stress lowers the prediction step by step. Where there are few participants (very low sleep quality, very high blood pressure) the curves jump, because the trees have little data there.
# 
# **Feature importance.** The trees use cardio by far the most, followed by age, ultra-processed diet, stress, novel learning, smoking and social connection. Blood pressure, sleep duration and sleep quality are hardly used, although they are correlated with the score (5b). This fits the data description: the score is a baseline minus penalties for age, stress, poor diet and smoking plus bonuses for cardio, learning and social connection. Blood pressure and sleep duration depend on some of these features, and sleep quality is strongly related to stress and cardio (5b), so they add little once the trees have the original features.

# ## 8. Pipeline B - Classification on the binned target
# 
# Turn the regression problem into a classification problem: the score is split into bins, and an ensemble classifier predicts which bin each participant belongs to.
# - The target is binned into equally wide bins with `pd.cut`
# - The pipeline is: preprocessing → `RandomForestClassifier`, tuned with `GridSearchCV`
# - Each predicted bin is turned back into a score (the median score of that bin in the training data), so we can compute the MAE
# - We test 3, 5, 7, 10 and 14 bins with 5-fold cross-validation and comment on how the number of bins affects the MAE
# - Afterwards, a heatmap shows which scores the classifier predicts for each true score

# In[20]:


# Section 8.1 - Bins and pipeline B: classification on the binned target

# 1) Which scores end up in the same bin? (equally wide bins with pd.cut)
bin_list = [3, 5, 7, 10, 14]
for n_bins in bin_list:
    bins = pd.cut(y_tune, bins=n_bins, labels=False)
    groups = y_tune.groupby(bins).agg(lambda s: f"{s.min()}-{s.max()}" if s.min() != s.max() else f"{s.min()}")
    print(f"{n_bins:>2} bins:", " | ".join(groups))

# 2) Pipeline B: preprocessing (selected features) -> random forest classifier
pipe_b = Pipeline([("prep", preprocess_sel),
                   ("model", RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE))])
param_grid_b = {"model__max_depth": [10, 20, None],
                "model__min_samples_leaf": [1, 5]}


# In[21]:


# Section 8.2 - Tune pipeline B and test the number of bins

# 1) Tune the random forest classifier once, on 14 bins (one bin per score)
gs_b = GridSearchCV(pipe_b, param_grid_b, cv=cv, scoring="accuracy")
gs_b.fit(X_tune, pd.cut(y_tune, bins=14, labels=False))
print("Best hyperparameters (14 bins):", gs_b.best_params_)

# 2) For each number of bins: 5-fold CV. Each predicted bin is turned back into a score
#    (the median score of that bin in the training folds), so we can compute the MAE
preds_b = {}
for n_bins in bin_list:
    pred = pd.Series(np.nan, index=y_tune.index)
    for train_idx, val_idx in cv.split(X_tune):
        X_tr, X_val = X_tune.iloc[train_idx], X_tune.iloc[val_idx]
        y_tr = y_tune.iloc[train_idx]

        bins_tr = pd.cut(y_tr, bins=n_bins, labels=False)   # score -> bin number
        bin_median = y_tr.groupby(bins_tr).median()          # bin number -> score
        model = clone(gs_b.best_estimator_).fit(X_tr, bins_tr)
        pred.iloc[val_idx] = bin_median.loc[model.predict(X_val)].to_numpy()
    preds_b[n_bins] = pred

results_b = pd.Series({n: mean_absolute_error(y_tune, p) for n, p in preds_b.items()}, name="cv_mae")
results_b.index.name = "n_bins"
display(results_b.to_frame().round(3))

best_bins = results_b.idxmin()
mae_b = results_b.min()
pred_b = preds_b[best_bins]  # out-of-fold predictions of the best version, used again in section 10
print("Best number of bins:", best_bins, "| CV MAE pipeline B:", round(mae_b, 3))

results_b.plot(marker="o", figsize=(7, 4), label="Pipeline B")
plt.axhline(mae_a_sel, color="grey", linestyle="--", label="Pipeline A1 (Ridge)")
plt.title("Pipeline B: CV MAE for different numbers of bins")
plt.xlabel("Number of bins")
plt.ylabel("CV MAE (lower is better)")
plt.legend()
plt.show()


# In[22]:


# Section 8.3 - Visualisation of pipeline B with 14 bins (one class per score)
# Rows = true score, columns = predicted score. Each row shows where the participants with that
# true score ended up, so each row adds up to 100%. A perfect model would only fill the diagonal.
confusion_b = pd.crosstab(y_tune, pred_b.astype(int), normalize="index")
plt.figure(figsize=(11, 7))
sns.heatmap(confusion_b, annot=True, fmt=".0%", cmap="Blues", cbar=False)
plt.title("Pipeline B: where the participants with each true score are predicted")
plt.xlabel("Predicted score")
plt.ylabel("True score")
plt.show()


# **Conclusion 8:** The number of bins has a large effect on the MAE. With few bins the prediction is coarse: with 3 bins, everyone in the bin 27–30 gets the same score, so the MAE is 1.326 even when the right bin is chosen.
# More bins generally give a lower MAE (5 bins: 1.134, 7 bins: 1.068), and 14 bins, one bin per score, is best with an MAE of 0.959.
# 10 bins is an exception (1.168, worse than 7 bins): because `pd.cut` makes equally wide bins, 29 and 30 end up in the same bin, and these are the two largest groups. Where the bin edges are placed therefore matters as much as how many bins there are.
# Even the best version of pipeline B is worse than pipeline A1 (0.910): a classifier does not know that a score of 28 is closer to 29 than to 22, so it cannot use the order of the scores the way a regression model does.
# 
# **What the heatmap shows (14 bins).** Each row shows where the participants with one true score end up. The classifier is best at 30: 73% of the participants with 30 are predicted correctly. For the other scores, most participants get a neighbouring score instead of the right one; for example, only 17% of the participants with 29 get 29, while 35% get 30 and 31% get 28. Low scores are hardly ever predicted correctly: participants with 19–21 are mostly predicted as 23, because there are very few low scores to learn from. Most errors are only 1–2 points, which is why the MAE is still 0.959 even though many participants do not get exactly the right score.

# ## 9. Pipeline C - Two-step model
# 
# We now have to build a two-step model that uses the fact that 23% of the participants have the maximum score of 30.
# 
# ```
# participant ──► step 1: random forest classifier ──"30"────────► prediction = 30
#                                │
#                                └──"below 30"──► step 2: regressor ──► prediction = regressor value
# ```
# 
# - Step 1: a `RandomForestClassifier` predicts whether a participant scores 30 or below 30
# - Step 2: a regression model predicts the score of everyone the classifier puts below 30
# - The results are combined into one prediction vector: 30 if the classifier says so, otherwise the regression value
# - We test two versions of step 2: C1, a Ridge model trained only on participants below 30, and C2, the gradient boosting model from 7b trained on all participants
# - All predictions are out-of-fold predictions from the same 5 folds as the other pipelines, so the MAE values can be compared directly
# - Finally, we plot how well step 1 finds the 30s and compare the predictions of C1 and C2

# In[23]:


# Section 9.1 - Step 1 of pipeline C: a classifier for "30 or below 30?"

# Step 1: classifier - does the participant score 30 (1) or below 30 (0)?
y_is30 = (y_tune == 30).astype(int)
pipe_clf = Pipeline([("prep", preprocess_sel),
                     ("model", RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE))])
param_grid_clf = {"model__max_depth": [10, 20, None],
                  "model__min_samples_leaf": [1, 5]}


# In[24]:


# Section 9.2 - Tune both steps of pipeline C and combine them

gs_clf = GridSearchCV(pipe_clf, param_grid_clf, cv=cv, scoring="accuracy", n_jobs=-1)
gs_clf.fit(X_tune, y_is30)
print("Step 1 - best hyperparameters:", gs_clf.best_params_)
print("Step 1 - CV accuracy:", round(gs_clf.best_score_, 3),
      "| always guessing 'below 30':", round(1 - y_is30.mean(), 3))

# Out-of-fold predictions of step 1 (same 5 folds): True where the classifier says 30
is30 = cross_val_predict(gs_clf.best_estimator_, X_tune, y_is30, cv=cv, n_jobs=-1) == 1

# Step 2, version C1: Ridge (from section 7a) trained only on the participants below 30
below = y_tune < 30
param_grid_reg = {"poly__degree": [2, 3], "model__alpha": [1, 100, 1000]}
gs_reg = GridSearchCV(pipe_a_sel, param_grid_reg, cv=cv, scoring="neg_mean_absolute_error", n_jobs=2)
gs_reg.fit(X_tune[below], y_tune[below])
print("Step 2 (C1) - best hyperparameters:", gs_reg.best_params_)
print("Step 2 (C1) - CV MAE on participants below 30:", round(-gs_reg.best_score_, 3))

reg_below = pd.Series(np.nan, index=y_tune.index)
for train_idx, val_idx in cv.split(X_tune):
    X_tr, y_tr = X_tune.iloc[train_idx], y_tune.iloc[train_idx]
    reg = clone(gs_reg.best_estimator_).fit(X_tr[y_tr < 30], y_tr[y_tr < 30])
    reg_below.iloc[val_idx] = reg.predict(X_tune.iloc[val_idx])

# Step 2, version C2: gradient boosting from section 7b, trained on ALL participants
# (pred_gbr already holds its out-of-fold predictions from the same 5 folds)

# Combine both steps: 30 where the classifier says 30, otherwise the regressor's prediction
pred_c1 = np.where(is30, 30, reg_below)
pred_c2 = np.where(is30, 30, pred_gbr)
mae_c1 = mean_absolute_error(y_tune, pred_c1)
mae_c2 = mean_absolute_error(y_tune, pred_c2)

# How well did step 1 find the participants with 30? (rows = true, columns = predicted)
display(pd.crosstab(y_tune == 30, is30, rownames=["true score is 30"], colnames=["classifier says 30"]))

# Compare the two versions, split by what the classifier says
abs_err_c1 = np.abs(pred_c1 - y_tune)
abs_err_c2 = np.abs(pred_c2 - y_tune)
comparison_c = pd.DataFrame({
    "MAE when classifier says 30": [abs_err_c1[is30].mean(), abs_err_c2[is30].mean()],
    "MAE when classifier says below 30": [abs_err_c1[~is30].mean(), abs_err_c2[~is30].mean()],
    "CV MAE (all)": [mae_c1, mae_c2]},
    index=["C1: regressor trained on < 30", "C2: regressor trained on all"])
display(comparison_c.round(3))


# In[25]:


# Section 9.3 - Visualisations of pipeline C

# 1) Step 1: how often is the classifier right? Each row adds up to 100%
confusion_step1 = pd.crosstab(np.where(y_tune == 30, "30", "below 30"), np.where(is30, "30", "below 30"),
                              rownames=["true score"], colnames=["classifier says"], normalize="index")
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_step1, annot=True, fmt=".0%", cmap="Blues", cbar=False)
plt.title("Pipeline C, step 1: true vs predicted '30'")
plt.show()

# 2) C1 and C2 side by side: predicted vs true score
fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
plot_pred_vs_true(pred_c1, "C1: classifier + Ridge trained on < 30", ax=axes[0])
plot_pred_vs_true(pred_c2, "C2: classifier + gradient boosting trained on all", ax=axes[1])
plt.tight_layout()
plt.show()


# **Conclusion 9:** Step 1, the random forest classifier (no depth limit, min_samples_leaf 5), is right 85.8% of the time, compared with 77.1% if we always guess "below 30". It finds 6,018 of the 11,445 participants who score 30, but misses 5,427 of them.
# In version C1, the regressor (Ridge, degree 3, alpha 100) is trained only on participants below 30. It is accurate for them (MAE 0.896), but the 5,427 missed participants with 30 get far too low a prediction, so C1 has a CV MAE of 0.933, worse than pipeline A1 (0.910).
# In version C2, the regressor is the gradient boosting model from 7b, trained on all participants, so the missed 30s also get a high prediction. The MAE for the participants the classifier puts below 30 drops from 1.045 to 0.990, and the total CV MAE drops to 0.887.
# C2 is the best pipeline so far, slightly better than gradient boosting alone (0.898): setting the participants the classifier marks as 30 to exactly 30 gives them an MAE of only 0.315.
# 
# **What the plots show.** The heatmap of step 1 shows that the classifier finds 53% of the participants with 30 and misses 47% of them, while it wrongly says 30 for only 4% of the participants below 30. The boxplots show what happens to the missed 30s. In C1 they get about 28, because the regressor has only seen participants below 30, so the box for true score 30 stretches down to about 28. In C2 the regressor has also seen the 30s, so the same participants get about 29 and the box is shorter. In both versions, the dots at 30 for lower true scores are the 4% of participants that the classifier wrongly puts at 30.

# ## 10. Model comparison and best model
# 
# Compare pipelines and choose the best one.
# - A table with the cross-validated MAE of each pipeline
# - We test two small changes to the predictions: cutting off values above 30 (`np.clip`), since no one can score more than 30, and rounding to whole numbers (`np.round`), since the true scores are integers. We keep a change only if it lowers the MAE.
# - We name the best model and explain why it is the best
# - An overview diagram shows how the pipelines differ and the result of each one

# In[26]:


# Section 10 - Model comparison and best model

# Out-of-fold predictions of every pipeline: the same 50,000 participants and the same 5 folds
oof_preds = {"A1: Ridge (selected features)": pred_a,
             "A2: Gradient boosting": pred_gbr,
             "B: Random forest on 14 bins": pred_b,
             "C1: classifier + Ridge trained on < 30": pred_c1,
             "C2: classifier + gradient boosting": pred_c2}

# For each pipeline: the MAE as it is, after cutting predictions above 30 down to 30,
# and after also rounding to whole numbers (the true scores are integers from 17 to 30)
rows = []
for name, pred in oof_preds.items():
    pred = np.asarray(pred, dtype=float)
    rows.append({"pipeline": name,
                 "cv_mae": mean_absolute_error(y_tune, pred),
                 "clipped_at_30": mean_absolute_error(y_tune, np.clip(pred, None, 30)),
                 "clipped_and_rounded": mean_absolute_error(y_tune, np.round(np.clip(pred, None, 30)))})
results = pd.DataFrame(rows).set_index("pipeline").sort_values("clipped_and_rounded")
display(results.round(3))

# How many predictions were above 30 before clipping?
for name, pred in oof_preds.items():
    print(f"{name}: {np.mean(np.asarray(pred, dtype=float) > 30):.1%} of the predictions are above 30")

results.iloc[::-1].plot.barh(figsize=(10, 4.5))
plt.title("CV MAE of every pipeline (lower is better)")
plt.xlabel("CV MAE")
plt.ylabel("")
plt.xlim(0.8, results.values.max() + 0.02)
plt.legend(["as predicted", "clipped at 30", "clipped and rounded"], loc="upper left", bbox_to_anchor=(1.01, 1))
plt.tight_layout()
plt.show()

best_name = results["clipped_and_rounded"].idxmin()
print("Best pipeline:", best_name, "| CV MAE after clipping and rounding:",
      round(results.loc[best_name, "clipped_and_rounded"], 3))


# In[27]:


# Section 10 - Overview of all pipelines: what is the same, what is different, and the result
# The numbers in the bottom row come from the table `results` above
steps = ["Input features", "Preprocessing", "What the model\nlearns", "Model", "Final prediction",
         "Tuned with\nGridSearchCV", "Evaluation", "CV MAE: as\npredicted →\nclipped + rounded"]
pipelines = {
    "A1: Ridge\n(section 7a)": [
        "15 features, then\nthe 10 selected", "impute, scale,\none-hot encode", "the score (17–30)",
        "PolynomialFeatures\n+ Ridge (linear)", "the predicted\nscore", "polynomial\ndegree, alpha",
        "5-fold CV, same\n50,000 rows, MAE"],
    "A2: Gradient\nboosting (7b)": [
        "10 selected\nfeatures", "impute, scale,\none-hot encode", "the score (17–30)",
        "gradient boosting\n(regression trees)", "the predicted\nscore", "learning rate,\ntree depth",
        "5-fold CV, same\n50,000 rows, MAE"],
    "B: Bins +\nclassifier (8)": [
        "10 selected\nfeatures", "impute, scale,\none-hot encode", "the bin of the\nscore (pd.cut)",
        "random forest\n(classifier)", "median score of\nthe predicted bin", "depth, leaf size,\nnumber of bins",
        "5-fold CV, same\n50,000 rows, MAE"],
    "C1: Two steps\n(section 9)": [
        "10 selected\nfeatures", "impute, scale,\none-hot encode", "1) 30 or below?\n2) score, only < 30",
        "1) random forest\n2) Ridge", "30, or the\nRidge score", "both steps",
        "5-fold CV, same\n50,000 rows, MAE"],
    "C2: Two steps\n(section 9)": [
        "10 selected\nfeatures", "impute, scale,\none-hot encode", "1) 30 or below?\n2) score, everyone",
        "1) random forest\n2) grad. boosting", "30, or the\nboosting score", "step 1 (step 2\ntuned in 7b)",
        "5-fold CV, same\n50,000 rows, MAE"],
}

# Add the result of each pipeline from the table in this section (same order as the columns)
result_names = ["A1: Ridge (selected features)", "A2: Gradient boosting", "B: Random forest on 14 bins",
                "C1: classifier + Ridge trained on < 30", "C2: classifier + gradient boosting"]
best = results["clipped_and_rounded"].idxmin()
for texts, name in zip(pipelines.values(), result_names):
    raw, final = results.loc[name, "cv_mae"], results.loc[name, "clipped_and_rounded"]
    texts.append(f"{raw:.3f} → {final:.3f}" + ("\nbest model" if name == best else ""))

colours = ["#cfe2f3", "#d9ead3", "#fce5cd", "#e6d7f2", "#f4cccc"]  # one colour per pipeline
grey = "#eeeeee"                                                   # used where all pipelines are the same

# Sizes are chosen so the figure is still readable when it is printed on an A4 page
box_w, box_h, gap_x, gap_y, left = 2.45, 1.0, 0.15, 0.45, 2.3
fig, ax = plt.subplots(figsize=(15, 14))
for col, ((name, texts), colour) in enumerate(zip(pipelines.items(), colours)):
    x = left + col * (box_w + gap_x)
    ax.text(x + box_w / 2, 0.8, name, ha="center", va="center", fontsize=14, fontweight="bold")
    for row, text in enumerate(texts):
        y_top = -row * (box_h + gap_y)
        same_for_all = all(other[row] == text for other in pipelines.values())
        is_best = row == len(steps) - 1 and result_names[col] == best  # the result box of the best pipeline
        ax.add_patch(plt.Rectangle((x, y_top - box_h), box_w, box_h,
                                   facecolor=grey if same_for_all else colour,
                                   edgecolor="black" if is_best else "grey", linewidth=3 if is_best else 1))
        ax.text(x + box_w / 2, y_top - box_h / 2, text, ha="center", va="center", fontsize=13,
                fontweight="bold" if is_best else "normal")
        if row < len(texts) - 1:  # arrow down to the next step
            ax.annotate("", xy=(x + box_w / 2, y_top - box_h - gap_y), xytext=(x + box_w / 2, y_top - box_h),
                        arrowprops=dict(arrowstyle="->", color="grey"))

for row, step in enumerate(steps):  # step names on the left
    ax.text(left - 0.15, -row * (box_h + gap_y) - box_h / 2, step, ha="right", va="center", fontsize=13)

bottom = -len(steps) * (box_h + gap_y)
ax.text(left, bottom, "Grey: the same for all pipelines.   Coloured: where they differ.   Lower MAE is better.",
        fontsize=13, va="top")
ax.set_xlim(0, left + len(pipelines) * (box_w + gap_x))
ax.set_ylim(bottom - 0.6, 1.4)
ax.axis("off")
ax.set_title("Overview of the five pipelines and their results", fontsize=17)
plt.show()


# **Overview of all pipelines:** All five pipelines use the same preprocessing and are evaluated in the same way, with 5-fold cross-validation on the same 50,000 participants and MAE as the score, so their results can be compared directly. They all use the 10 selected features; only A1 is first tested with all 15, which is how the five noise features were found. They differ in what the model learns and which model is used:
# - **A1** and **A2** predict the score directly: A1 with a linear model (Ridge with polynomial features) and A2 with regression trees (gradient boosting).
# - **B** turns the score into bins, predicts the bin with a random forest classifier, and turns the bin back into a score.
# - **C1** and **C2** first ask "30 or below 30?" with a random forest classifier and use a regression model for the rest: C1 a Ridge model trained only on participants below 30, and C2 the gradient boosting model from A2, trained on everyone.
# 
# The bottom row shows what these differences mean for the result. Predicting the score directly (A1, A2) works better than predicting a bin (B). A two-step model only helps when the regressor in step 2 has also seen the participants with 30 (C2: 0.887, C1: 0.933). The best pipeline, C2, combines a classifier that sets the clear 30s to exactly 30 with gradient boosting for everyone else, and reaches a CV MAE of 0.864 after clipping and rounding.

# **Conclusion 10:** The table compares all pipelines on the same 50,000 participants and the same 5 folds.
# Clipping predictions at 30 helps the two regression models, which sometimes predict impossible scores above 30 (3.4% of the Ridge predictions and 7.2% of the gradient boosting predictions). Rounding to whole numbers helps every pipeline except B, whose predictions are already whole numbers. It helps because the true scores are integers: for example, gradient boosting goes from 0.898 to 0.867.
# Our best model is **C2: the random forest classifier combined with gradient boosting, with the predictions clipped at 30 and rounded**, with a CV MAE of 0.864. Gradient boosting alone (0.867) and Ridge (0.873) are close behind, while C1 (0.918) and B (0.959) are clearly worse.
# The hyperparameters were chosen with the same 5 folds, so all these MAE values are slightly optimistic. This affects all pipelines in the same way, so the comparison is still fair; the Kaggle score on the test set shows the real performance.

# ## 11. Kaggle submission
# 
# We have to use the best model to predict the cognitive score of the 150,000 participants in the test set.
# - The best model is trained again on all 350,000 cleaned training rows, not only the 50,000-row sample
# - We predict the test set and apply the changes that helped in section 10
# - The predictions are saved as `submission.csv` with the columns `id` and `target`
# - We upload the file to Kaggle and report the public leaderboard score, which has to beat the BEAT ME benchmark

# In[28]:


# Section 11 - Kaggle submission with the best model (C2: classifier + gradient boosting)

# 1) Train both steps of C2 again on ALL 350,000 cleaned training rows (not only the 50,000-row sample),
#    with the best hyperparameters found in sections 9 and 7b
final_clf = clone(gs_clf.best_estimator_).set_params(model__n_jobs=-1)  # n_jobs=-1: use all CPU cores
final_clf.fit(X, (y == 30).astype(int))
final_gbr = clone(gs_gbr.best_estimator_)
final_gbr.fit(X, y)  # this step takes several minutes

# 2) Predict the test set: 30 where the classifier says 30, otherwise the gradient boosting prediction
is30_test = final_clf.predict(df_test) == 1
pred_test = np.where(is30_test, 30, final_gbr.predict(df_test))

# 3) Clip at 30 and round to whole numbers, because this lowered the CV MAE in section 10
pred_test = np.round(np.clip(pred_test, None, 30)).astype(int)

# 4) Save the submission file: one row per test participant, with the columns id and target
submission = pd.DataFrame({"target": pred_test}, index=df_test.index)
submission.index.name = "id"
submission.to_csv("submission.csv", index=True)

# 5) Check the file before uploading it
example = pd.read_csv("example_submission.csv")
print("Shape:", submission.shape)
print("Same ids in the same order as example_submission.csv:", (submission.index.values == example["id"].values).all())
print("Missing values:", submission["target"].isna().sum())
print("Predicted scores from", submission["target"].min(), "to", submission["target"].max())
print("Share predicted 30:", round((submission["target"] == 30).mean(), 3),
      "| share with 30 in train:", round((y == 30).mean(), 3))
display(submission.head())

# Does the distribution of our predictions look like the true scores in train?
score_shares = pd.DataFrame({"train (true scores)": y.value_counts(normalize=True),
                             "test (our predictions)": submission["target"].value_counts(normalize=True)}).sort_index()
score_shares.plot.bar(figsize=(10, 4))
plt.title("Share of participants per score: true scores in train vs our predictions for test")
plt.xlabel("Cognitive score")
plt.ylabel("Share of participants")
plt.show()


# **Conclusion 11:** The final model C2 was trained again on all 350,000 cleaned training rows: the random forest classifier decides who gets 30, and gradient boosting predicts the score for everyone else. The predictions were clipped at 30 and rounded, as in section 10.
# The submission file has 150,000 rows, the same ids in the same order as `example_submission.csv`, no missing values, and scores from 18 to 30.
# Compared with the true scores in train, our predictions are more concentrated around 27–29, and fewer participants are predicted to have 30 (17% vs 23%). This is expected: a model that minimises the MAE only predicts an extreme score when it is fairly sure, so uncertain participants get a score in the middle.
# 
# **Kaggle result:** Our submission scored an MAE of **0.82417** on the public leaderboard (3rd place when we submitted). This is better than our cross-validated MAE of 0.864, probably for two reasons: the test set is clean (no missing or impossible values), while our cross-validation also included the messy training rows, and the final model was trained on all 350,000 rows instead of 40,000 per fold.
