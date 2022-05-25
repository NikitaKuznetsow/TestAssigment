from sklearn.metrics import f1_score, roc_auc_score, log_loss, classification_report, plot_roc_curve, \
    plot_precision_recall_curve
import pandas as pd


def get_metrics(model, X_train, X_test, y_train, y_test, cut_off):
    '''
    Calculate metrics (roc_auc, gini, f1-score) and plot graphs (roc curve, precision-recall curve)

    :param model: some kind of classifier
    :param X_train: pd.DataFrame
            train data
    :param X_test: pd.DataFrame
            test data
    :param y_train:
            train label
    :param y_test: pd.Series
            test label
    :param cut_off: float
            decision threshold - numder of decision for converting a predicted probability
                                 into a class label
    '''
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba > cut_off).astype('int16')

    print('ROC_AUC:  ', round(roc_auc_score(y_test, y_proba), 3))
    print('Gini:     ', round(2 * roc_auc_score(y_test, y_proba) - 1, 3))

    print('F1_score: ', round(f1_score(y_test, y_pred), 3))
    print('Log_loss: ', round(log_loss(y_test, y_pred), 3))

    print('Classification_report: \n',
          classification_report(pd.Series(y_proba).apply(lambda x: 1 if x > cut_off else 0),
                                y_test))

    plot_curves(model, X_test, y_test)


def plot_curves(model, X_test, y_test):
    """
    Plot roc and precision-recall curves

    :param model: some kind of classifier
    :param X_test: pd.DataFrame
        test data
    :param y_test: pd.Series
        test label
    """
    plot_roc_curve(model, X_test, y_test)
    plot_precision_recall_curve(model, X_test, y_test)
