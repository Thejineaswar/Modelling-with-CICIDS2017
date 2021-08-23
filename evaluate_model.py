from sklearn.metrics import precision_recall_fscore_support,confusion_matrix
from sklearn.metrics import roc_auc_score,accuracy_score
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from .getLists import *

PATHS,col = getList()
MODEL_TYPE = 'AE+ANN' # Either AE+ANN or ANN
MODEL_PATH = ' ' # Set the model path of the model type

def print_result(MODEL_TYPE,valid_df,model):
    if MODEL_TYPE == 'ANN':
        res = model.predict(valid_df[col])
        preds = res
    elif MODEL_TYPE == 'AE+ANN':
        res = model.predict(valid_df[col])
        preds = res[1]

    res = np.argmax(preds, axis=1)

    truth = np.argmax(valid_df.iloc[:, -15:].values, axis=1)

    truth = np.argmax(valid_df.iloc[:, -15:].values, axis=1)
    accuracy = accuracy_score(truth, res)
    roc_auc = roc_auc_score(valid_df.iloc[:,-15:], preds, multi_class='ovr')
    precision, recall, _, _ = precision_recall_fscore_support(truth, res)

    print(f"Accuracy : {accuracy}")
    print(f"ROC AUC Score :{roc_auc}")
    print(f"Precision : {precision}")
    print(f"Recall : {recall}")


if __name__ == '__main__':
    valid_df = pd.read_csv('valid_df.csv')
    model = load_model(MODEL_PATH)
    print_result(MODEL_TYPE,valid_df,MODEL_PATH)



