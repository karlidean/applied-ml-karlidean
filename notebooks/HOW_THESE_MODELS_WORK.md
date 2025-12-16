# How These Models Work
## Model 1: California Housing Price Prediction
You can find this model [here](https://github.com/karlidean/applied-ml-karlidean/blob/main/notebooks/project01/ml01.ipynb)!
- Utilized Jupyter Notebooks to predict housing prices in the state of California
- Used visualization techniques from the Seaborn package, including boxplots, histograms, and pairplots
- Trained a model using `train_test_split()`
- Evaluated the model using R², RMSE, and MAE scores
- Predicted the **median house price** based on selected home features:
  - The owner's median income (`MedInc`)
  - The average number of rooms in a home (`AveRooms`)
## Model 2: Data - Inspect, Explore, Split, & Engage
- Utilized Seaborn package `Titanic` data set to illustrate stratification of the data
- Used visualization techniques from the Seaborn package, including histograms, scatterplots, and countplots
- Engineered features by changing alphabetic features to digits and creating a new field (called `family_size`)
- Split and trained my data using `train_test_split()` and `StratifiedShuffleSplit()`
- Evaluated the survival rates of Titanic passengers of each class in each split portion of data
  - Training of `train_test_split()`
  - Testing of `train_test_split()`
  - Training of `StratifiedShuffleSplit()`
  - Testing of `StratifiedShuffleSplit()`
