from sklearn import preprocessing
import pandas as pd

"""
Approaches:
    - label encoding
    - one hot encoding
    - binarization
"""

class CategoricalFeatures:
    def __init__(self, df, categorical_features, encoding_type, handle_na=False):
        """
        df: DataFrame
        categorical_features: list of column names e.g.["ord_1", "nom_0", ...]
        encoding_type: label, onehot, binary
        handle_na: True/False
        """
        self.df = df
        self.output_df = self.df.copy(deep=True)
        self.categorical_features = categorical_features
        self.enc_type =  encoding_type
        self.handle_na = handle_na
        self.label_encoders = dict() # store LabelEncoder object for later encoding testing data
        self.binary_encoders = dict() # store LabelBinarizer object for later encode testing data

        # Impute Nan with dummy value: works for LabelEncoding only
        if self.handle_na:
            for col in self.categorical_features:
                self.df.loc[:, col] = self.df.loc[:, col].astype(str).fillna("-9999999")


    def _label_encoding(self):
        for col in self.categorical_features:
            label = preprocessing.LabelEncoder()
            label.fit(self.df[col].values)
            self.output_df.loc[:, col] = label.transform(self.df[col].values)
            self.label_encoders[col] = label
        return self.output_df

    def _label_binarization(self):
        for col in self.categorical_features:
            label = preprocessing.LabelBinarizer()
            label.fit(self.df[col].values)
            result = label.transform(self.df[col].values) # Array of the same dimension as the no. of label available
            # Drop the current col
            self.output_df = self.output_df.drop(col, axis=1)
            # Merge new cols to the output dataframe
            for j in range(result.shape[1]):
                new_col_name = col + f"_new_{j}"
                self.output_df[new_col_name] = result[:, j]
            self.binary_encoders[col] = label

        return self.output_df

    # In case of first creation of encoders
    def fit_transform(self):
        if self.enc_type == 'label':
            return self._label_encoding()
        elif self.enc_type == 'binary':
            return self._label_binarization()
        else:
            raise Exception("Encoding type not understood")

    # In case of loading encoders, for testing data
    def transform(self, dataframe):
        # Impute Nan with dummy value: works for LabelEncoding only
        if self.handle_na:
            for col in self.categorical_features:
                dataframe.loc[:, col] = dataframe.loc[:, col].astype(str).fillna("-9999999")

        # Load label encoders saved earlier and transform the dataset
        if self.enc_type == "label":
            for col, label in self.label_encoders.items():
                dataframe.loc[:, col] = label.transform(dataframe[col].values)
            return dataframe

        # Load label encoders saved earlier and transform the dataset
        elif self.enc_type == "binary":
            for col, label in self.binary_encoders.items():
                result = label.transform(self.df[col].values)  # Array of the same dimension as the no. of label available
                # Drop the current col
                dataframe = dataframe.drop(col, axis=1)

                # Merge new cols to the output dataframe
                for j in range(result.shape[1]):
                    new_col_name = col + f"_new_{j}"
                    dataframe[new_col_name] = result[:, j]
            return dataframe

        else:
            raise Exception("Encoding type not understood")

if __name__ == "__main__":
    # Step 1: load datasets(train and test)
    df_train = pd.read_csv("../input/categorical_2_train.csv") #.head(500)
    df_test = pd.read_csv("../input/categorical_2_test.csv")

    # Step 2: keep track on their respective indices
    df_train_index = df_train["id"].values
    df_test_index = df_test["id"].values

    # Step 3: Concatenate train and test
    df_test['target'] = -1   # fake target column bz test data does not have
    full_data = pd.concat([df_train, df_test])

    # Step 4: Transform all cols except "id" and "target"
    cols = [c for c in full_data.columns if c not in ['id', 'target']]

    # Step 5: Transform label for the full data
    cat_features = CategoricalFeatures(full_data,
                                       categorical_features=cols,
                                       encoding_type='label',
                                       handle_na=True)
    # Step 6: Get the final transformed dataset(train and test)
    full_data = cat_features.fit_transform()

    # Step 7: Split train and test set
    df_train = full_data[full_data['id'].isin(df_train_index)].reset_index(drop=True)
    df_test = full_data[full_data['id'].isin(df_test_index)].reset_index(drop=True)

    # Step 8: Remove the fake target column in test data
    df_test = df_test.drop('target', axis=1)

    print(df_train.shape)
    print(df_test.shape)

