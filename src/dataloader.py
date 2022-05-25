import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings('ignore')


class DataLoader:
    """
    Load a dataset and prepare for the prediciton.

    Methods:
    ------
    get_test_train_data()
        Key method for the getting train and test data
    """

    def __init__(self):
        self.num_cols = ['default', 'account_amount_added_12_24m', 'account_days_in_dc_12_24m',
                         'account_days_in_rem_12_24m', 'account_days_in_term_12_24m', 'age',
                         'avg_payment_span_0_12m', 'max_paid_inv_0_12m',
                         'num_active_div_by_paid_inv_0_12m', 'num_active_inv',
                         'num_arch_dc_0_12m', 'num_arch_dc_12_24m', 'num_arch_ok_0_12m',
                         'num_arch_rem_0_12m', 'num_arch_written_off_0_12m',
                         'num_arch_written_off_12_24m', 'num_unpaid_bills',
                         'status_last_archived_0_24m', 'status_2nd_last_archived_0_24m',
                         'status_3rd_last_archived_0_24m', 'status_max_archived_0_6_months',
                         'status_max_archived_0_12_months', 'recovery_debt',
                         'sum_capital_paid_account_0_12m', 'sum_capital_paid_account_12_24m',
                         'sum_paid_inv_0_12m', 'time_hours']

        self.cat_cols = ['merchant_category', 'merchant_group', 'has_paid', 'name_in_email']

    def get_test_train_data(self):
        """
        Load, clean and prepare data for the prediction
        :return: X_train, y_train : pd.DataFrame, pd.Series
                    train data
                 X_test, y_test : pd.DataFrame, pd.Series
                    test data
        """
        df = pd.read_csv('../data/dataset.csv', delimiter=';')[self.num_cols + self.cat_cols]
        df = df.dropna().reset_index(drop=True)
        df['has_paid'] = df['has_paid'].astype('int16')
        X, y = df.drop(columns='default'), df['default']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        X_train.reset_index(inplace=True, drop=True), X_test.reset_index(inplace=True, drop=True)
        cols_for_ohe = ['merchant_group', 'name_in_email']
        X_train, X_test = self._apply_ohe(cols_for_ohe, X_train, X_test)
        X_train, X_test = self._apply_frequency_encoder('merchant_category', X_train, X_test)
        X_train, y_train = self._apply_SMOTE(X_train, y_train)
        return X_train, X_test, y_train, y_test

    @staticmethod
    def _apply_ohe(cols, X_train, X_test):
        """
        Apply OneHotEncoder to the dataset
        ...

        :param cols: list,
            columns for the OneHotEncoding
        :param X_train: pd.DataFrame,
            train dataframe
        :param X_test: pd.DataFrame,
            test dataframe
        :return: X_train : pd.DataFrame,
                 X_test : pd.DataFrame
        """
        ohe = OneHotEncoder(handle_unknown='ignore')
        ohe.fit(X_train[cols])
        cols_after_ohe = ohe.get_feature_names()
        X_train_ohe = pd.DataFrame(data=ohe.transform(X_train[cols]).toarray(), columns=cols_after_ohe)
        X_test_ohe = pd.DataFrame(data=ohe.transform(X_test[cols]).toarray(), columns=cols_after_ohe)
        X_train = pd.concat([X_train, X_train_ohe], axis=1)
        X_test = pd.concat([X_test, X_test_ohe], axis=1)
        X_train.drop(columns=cols, inplace=True)
        X_test.drop(columns=cols, inplace=True)
        return X_train, X_test

    @staticmethod
    def _apply_frequency_encoder(col, X_train, X_test):
        """
        Apply FrequencyEncoder to the dataset
        ...

        :param col: str,
            column for the FrequencyEncoding
        :param X_train: pd.DataFrame,
            train dataframe
        :param X_test: pd.DataFrame,
            test dataframe
        :return: X_train : pd.DataFrame,
                 X_test : pd.DataFrame
        """
        for X in [X_train, X_test]:
            fq = X.groupby(col).size() / len(X)
            X.loc[:, f"{col}_freq_encode"] = X[col].map(fq)
            X.drop([col], inplace=True, axis=1)
        return X_train, X_test

    @staticmethod
    def _apply_SMOTE(X, y):
        """
        Apply method of oversampling - SMOTE
        ...

        :param X: pd.DataFrame
        :param y: pd.Series
        :return: X_train_smote: pd.DataFrame
                 y_train_smote: pd.Series
        """
        smote = SMOTE()
        X_train_smote, y_train_smote = smote.fit_resample(X, y)
        return X_train_smote, y_train_smote
