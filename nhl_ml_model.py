"""
This script contains examples of model training and function definitions for 
machine learning models that generate recommendations for betting on nhl-games.

Predicting with the ensemble of deep learning models generates a set of 
"recommended weights" to bet on each outcome for a given game, while
the XGBoost model generates estimated probabilities for each outcome, that
can be compared to bookmaker odds. I leave the task combining these two
different sets of predictions up to the discretion of the reader.

Note that the actual live implementation of generating predictions is not
demonstrated in this sample script, but should be fairly evident from the 
function definitions and descriptions.


Author: Lauri Heikka

"""

# Import relevant libraries
from keras.layers import BatchNormalization, Dense, Input, Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.pipeline import Pipeline
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.decomposition import PCA
from keras import backend as K

"""
########################################################
########## Function definitions ########################
########################################################
"""

def prep_data(filename):
    """
    This function prepares the data for model training
    Input: the filename for the file containing processed features for training
    
    Output: X, the set of features to be used for model training
            y_true, a set of one-hotencoded labels along with the game odds
                to be used in model training
    """
    data = pd.read_csv(filename, index_col=0)
    data = data[~data.index.duplicated(keep=False)]
    data.drop(['home', 'away','wt_shots','wt_shots_allowed',
               'wt_shots_away','wt_shots_allowed_away', 'svp','svp_away'], axis=1, inplace=True)
    y_true = (pd.get_dummies(data, columns=['result']).iloc[:,[31,32,33,28,29,30]])
    y_true = y_true.iloc[:,[0,2,1,3,4,5]]
    
    # We reduce the number of features further by using home stats minus away 
    # stats, to get the differences between the two instead of using home and
    # away stats as separate features
    h = data.iloc[:,:14]
    a = data.iloc[:,14:28]
    a.columns=h.columns
    X = h-a
    X = X.join(data.iloc[:,-4:-1])
    return X, y_true


def odds_loss(y_true, y_pred):
    """
    This function implements a custom loss function adapted from one by 
    Charles Malafosse.
    Refer to here for the details:
    https://towardsdatascience.com/machine-learning-for-sports-betting-not-a-basic-classification-problem-b42ae4900782
    
    Inputs
    y_true is a dataframe with 6 columns, 3 for each label and 3 for the odds
    y_pred : a vector of probabilities with a column for each label
    

    The function returns the loss value
    """
   
    win_home_team = y_true[:, 0:1]
    draw = y_true[:, 1:2]
    win_away = y_true[:, 2:3]
    
    odds_1 = y_true[:,3:4]
    odds_x = y_true[:,4:5]
    odds_2 = y_true[:,5:6]
    gain_loss_vector = K.concatenate([win_home_team * (odds_1 - 1) + (1 - win_home_team) * -1,
                                      win_away * (odds_2 - 1) + (1 - win_away) * -1,
                                      draw * (odds_x - 1) + (1 - draw) * -1], axis=-1)
    return -1 * K.mean(K.sum((gain_loss_vector * y_pred), axis=1))


def get_ensemble_preds(x_test, n_members = 9):
    """
    This function takes in as argument, a set of features, x_test,
    and the number of saved models that we want to use for the prediction,
    n_members.
    The function returns an ensemble prediction, averaged from a set of neural
    networks that have been pretrained and saved on disk.
    
    """
    from keras.models import model_from_json
    json_file = open('model_optimize1.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights('model_optimize1.h5')
    print("Loaded model from disk")
    loaded_model.compile(optimizer='adam', loss=odds_loss, metrics=['accuracy'])
    y_pred = loaded_model.predict_proba(x_test)
        
    for i in range(n_members-1):
        json_file = open('model_optimize{}.json'.format(i+2), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights('model_optimize{}.h5'.format(i+2))
        print("Loaded model from disk")
        loaded_model.compile(optimizer='adam', loss=odds_loss, metrics=['accuracy'])
        y_pred = y_pred + loaded_model.predict_proba(x_test)
    y_pred = y_pred / n_members
    return y_pred

def prep_current(filename):
    """
    This function prepares the data from current games available for betting
    to generate model predictions.
    
    Input: the filename for the file containing processed features for games
    available for betting
    
    Output: X, the set of features in the model
            names, the team names for each game
    """
    data = pd.read_csv(filename, index_col=0)
    names = data[['home', 'away']]
    X = (data.drop(['home', 'away'], axis=1))
    return X, names

def odds_evaluate(y_true, y_pred):
    """
    This function computes the per-game expected return and t-statistic
    as an initial evaluation of the strategy on test data

    Inputs
    y_true is a dataframe with 6 columns, 3 for each label and 3 for the odds
    y_pred : a vector of probabilities with a column for each label
    

    mean: the average return of the strategy per game
    stat: one-sample t-statistic of the series of betting results
    """
    win_home_team = y_true.result_1
    tie = y_true.result_x
    win_away = y_true.result_2
    
    odds_1 = y_true.odds_1
    odds_2 = y_true.odds_2
    odds_x = y_true.odds_x
    homewins = (win_home_team * (odds_1 - 1) + (1 - win_home_team) * -1)*y_pred[:,0]
    awaywins = (win_away * (odds_2 - 1) + (1 - win_away) * -1)*(y_pred[:,1])
    ties = (tie * (odds_x - 1) + (1 - tie) * -1)*(y_pred[:,2])
    mean = round(np.mean(homewins+awaywins+ties), 3)
    stat = round(stats.ttest_1samp(homewins+awaywins+ties, 0).statistic, 3)
    return mean, stat

"""
########################################################
########## Data input and preprocessing ################
########################################################
"""

X, y_true = prep_data('training.csv')

X, X_test, y, y_test = train_test_split(X, y_true, test_size=0.2)
scaler = preprocessing.MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X.values))
X_test = pd.DataFrame(scaler.transform(X_test.values))



"""
The code below shows an example of how the models were trained.
To generate predictions we employ an ensemble of these deep learning models,
pretrained with slightly varying model parameters.

"""

n_cols = X.shape[1]
input_shape = (n_cols,)
# Specify the model
model = Sequential()
model.add(Dense(n_cols, input_shape = input_shape, activation='relu'))
model.add(Dense(n_cols, activation='relu'))
model.add(Dense(n_cols, activation='relu'))
model.add(Dense(n_cols, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax'))
# Compile the model
model.compile(optimizer='adam', loss=odds_loss)

# Define an early stopping monitor based on loss result from validation split
early_stopping_monitor = EarlyStopping(monitor='val_loss', patience=10)

# Fit model
training = model.fit(X, y, epochs=100, validation_split=0.3,
                     batch_size=256, callbacks=[early_stopping_monitor], verbose=True)

# Plot the model training history
plt.plot(training.history['val_loss'], 'r')
plt.xlabel('Epochs')
plt.ylabel('Validation score')
plt.show()

# Finally, evaluate model on test and training datasets
print(odds_evaluate(y_test, model.predict_proba(X_test).round(2)))
print(odds_evaluate(y, model.predict_proba(X).round(2)))

"""
#############################################
########## XGBoost Model functions ##########
#############################################
"""

def prep_data_xgb(filename):
      """
    This function prepares the data for Gradient boosting model training
    Input: the filename for the file containing processed features for training
    
    Output: X, the set of features to be used for model training
            y_true, a set of one-hotencoded labels along with the game odds
                to be used in model training
            odds, the set of odds to be used for model training
    
    We now drop the odds from the set of features, because the gradient
    boosting model instead generates predictions for game outcomes and we want
    to separate the predictions made by the gradient boosting model from 
    predictions implied by odds as much as possible. 
    The odds loss function cannot be used for gradient boosted model training
    because it is not twice separable.
    
    """
    data = pd.read_csv(filename, index_col=0)
    data.drop(['home', 'away','wt_shots','wt_shots_allowed',
               'wt_shots_away','wt_shots_allowed_away', 'svp','svp_away'], axis=1, inplace=True)
    h = data.iloc[:,:14]
    a = data.iloc[:,14:28]
    a.columns=h.columns
    odds = data[['odds_1', 'odds_x','odds_2']]
    X = h-a
    y = pd.Series(np.select([data.result=='1', data.result=='x', data.result=='2'], [0,1,2]))
    y.index = data.index
    return X, y, odds

def odds_eval(preds, dtrain):
    """
    Evaluation function for the gradient boosted model, to be used in early
    stopping.
    
    Inputs
    preds: Predicted probabilities for each three labels
    dtrain: DMatrix object with training, or testing data.
    
    Output
    strat_return: the return of a betting strategy, that sizes the bet based
    on how much higher the predicted probability is than the probability
    implied by bookmaker odds
    
    """
    labels = dtrain.get_label()
    # When objective=softprob, preds has shape (N, K)
    labels = preprocessing.OneHotEncoder(sparse=False, categories='auto').fit_transform(labels.reshape(-1, 1))
    win_home_team = labels[:,0]
    tie = labels[:,1]
    win_away = labels[:,2]
    
    # get odds for training data or test data based on the length of the data
    # provided as an argument. Assumes that the data is not evenly split into
    # training and test sets
    
    if len(labels) == len(o_test):
        odds = o_test
    else:
        odds = o
        
    odds_1 = odds.odds_1.values
    odds_x = odds.odds_x.values
    odds_2 = odds.odds_2.values
    
    # Label is a vector of class indices for each input example
    homes = (preds[:,0]-1/odds_1)
    homes[homes<0] = 0
    aways = (preds[:,2]-1/odds_2)
    aways[aways<0] = 0
    ties = (preds[:,1]-1/odds_x)
    ties[ties<0] = 0
    
    
    homewins = (win_home_team * (odds_1 - 1) + (1 - win_home_team) * -1)*homes
    awaywins = (win_away * (odds_2 - 1) + (1 - win_away) * -1)*aways
    ties = (tie * (odds_x - 1) + (1 - tie) * -1)*ties
    gain_loss = np.transpose(np.array([homewins, awaywins, ties]))
    strat_return = np.mean(gain_loss * preds)
    
    # Return as 1-d vectors
    return ('odds_eval', strat_return)

"""
#############################################
########## XGBoost Model data prep ##########
#############################################
"""

X_train, y_train, o _train = prep_data_xgb('training.csv')
X_test, y_test, o_test = prep_data_xgb('testing.csv')


scaler = preprocessing.MinMaxScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train.values))
X_test = pd.DataFrame(scaler.fit_transform(X_test.values))
X_train.index = y_train.index
X_test.index = y_test.index

dtrain = xgb.DMatrix(X_train, y_train)
dtest = xgb.DMatrix(X_test, y_test)


"""
########################################################
########## Model specification and training ############
########################################################
"""

"""
For the XGBoost model, no separate ensembling is required, because the
algorithm is itself an ensemble.

"""
xgb_params = {
    'silent': 1,
    'subsample': 0.2,
    'learning_rate': 0.001,
    'objective':'multi:softprob',
    'max_depth': 4,
    'num_class': 3
    }


model = xgb.train(xgb_params, dtrain, 1000,
                  feval=odds_eval,
                  maximize=True,
                  early_stopping_rounds = 30,
                  evals=[(dtrain, 'dtrain'),(dtest, 'dtest')])

print(model.predict(dtrain).round(2))
print(odds_evaluate(y_train, model.predict(dtrain), o_train))
print(odds_evaluate(y_test, model.predict(dtest), o_test))
print(odds_evaluate(y, model.predict(dx), X_o))
