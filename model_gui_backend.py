from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import accuracy_score, confusion_matrix

import numpy as np
import pandas as pd
import logging

from dataset import load_dataset


# Provide the user a choice of using either the LogisticRegression or DecisionTreeClassifier
# Output the most important features (factors) that result in startup success

class StartupSuccessPredModel:
    def __init__(self, dataset):
        self.startup_df = dataset

        self.X = None
        self.X_train = None
        self.X_test = None

        self.y = None
        self.y_train = None
        self.y_test = None

        self.num_cols = None
        self.cat_cols = None

        self.pipe_lr = None
        self.pipe_dt = None

        self.logger = logging.getLogger(__name__)
        logging.basicConfig(filename='startup_model_feedback.log', level=logging.DEBUG, format='[%(asctime)s] {%(levelname)s} %(name)s: #%(lineno)d - %(message)s')


    def clean_dataset(self):
        self.logger.info(self.startup_df.head())

        self.y = self.startup_df['status']
        self.X = self.startup_df.drop(columns=['status', 'name', 'region', 'country_code', 'state_code'])

        self.y = np.where((self.y == 'ipo') | (self.y == 'acquired'), 1, 
                          np.where(self.y == 'closed', 0, 
                                   np.where(self.y == 'operating', -1, np.nan)))
        
        # Turn compatible cols into dtype int or float (otherwise np.nan)
        for col in self.X.columns.tolist():
            self.X[col] = np.where((self.X[col] == '-') | (self.X[col] == ''), np.nan, self.X[col])

            try:
                self.X[col] = pd.to_numeric(self.X[col], errors='raise')
            except:
                pass

        self.logger.debug(self.X.info(verbose=True))

        # Take the year instead of the entire date
        for col in ['founded_at', 'first_funding_at', 'last_funding_at']:
            self.X[col] = self.X[col].apply(lambda x: x if isinstance(x, str) == False else x.split('-')[0])

            self.X[col] = np.where(self.X[col] == '2105', '2015', self.X[col])

        self.logger.info(self.X.head())

        # Simplify categories by taking the first one rather than the whole list
        self.X['category_list'] = self.X['category_list'].apply(lambda x: x if isinstance(x, str) == False else x.split('|')[0])

        self.logger.info(self.X.head())
        self.logger.info(self.X['category_list'].value_counts())
        self.logger.info(self.X['funding_rounds'].value_counts())        
        self.logger.info(self.X['city'].value_counts())

    
    def get_df_subsets(self):
        self.num_cols = self.X.select_dtypes(include=np.number).columns.tolist()
        self.cat_cols = self.X.select_dtypes(include=['object']).columns.tolist()

        self.logger.info('Splitting numerical cols and categorical cols')


    def train_model(self):
        self.logger.debug('train_test_split dataset')

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, train_size=0.9, test_size=0.1, random_state=42)

        self.logger.info('Creating pipeline')

        num_pipe = Pipeline([('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())])
        cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))])

        preprocessor = ColumnTransformer(
            transformers=[('num_pipe', num_pipe, self.num_cols), ('cat_pipe', cat_pipe, self.cat_cols)]
        )

        self.pipe_lr = Pipeline([('preprocessor', preprocessor), ('clf', LogisticRegression(max_iter=1000, solver='liblinear'))])
        self.pipe_dt = Pipeline([('preprocessor', preprocessor), ('clf', DecisionTreeClassifier(max_depth=7, random_state=42))])

        self.logger.debug('fit() training dataset')

        self.pipe_lr.fit(self.X_train, self.y_train)
        self.pipe_dt.fit(self.X_train, self.y_train)

    
    def get_model_score(self, clf_type):
        if clf_type == 'lr':
            y_pred = self.pipe_lr.predict(self.X_test)
            self.logger.info('Conduct predict() inference on model')

            return accuracy_score(self.y_test, y_pred)
        
        elif clf_type == 'dt':
            y_pred = self.pipe_dt.predict(self.X_test)
            self.logger.info('Conduct predict() inference on model')

            return accuracy_score(self.y_test, y_pred)
        

    def get_model_confusion_matrix(self, clf_type):
        if clf_type == 'lr':
            self.logger.info('Conduct predict() inference on model')

            y_pred = self.pipe_lr.predict(self.X_test)

            self.logger.info('Fetching confusion_matrix()')

            confusion_matrix_lr = confusion_matrix(self.y_test, y_pred, labels=[1, 0, -1])
            index = ['actual_success', 'actual_unsuccess', 'actual_operating']
            columns = ['predicted_success', 'predicted_unsuccess', 'predicted_operating']

            test_confusion_matrix = pd.DataFrame(data=confusion_matrix_lr, index=index, columns=columns)
            return test_confusion_matrix
        
        elif clf_type == 'dt':
            self.logger.info('Conduct predict() inference on model')

            y_pred = self.pipe_dt.predict(self.X_test)

            self.logger.info('Fetching confusion_matrix()')

            confusion_matrix_dt = confusion_matrix(self.y_test, y_pred, labels=[1, 0, -1])
            index = ['actual_success', 'actual_unsuccess', 'actual_operating']
            columns = ['predicted_success', 'predicted_unsuccess', 'predicted_operating']


            test_confusion_matrix = pd.DataFrame(data=confusion_matrix_dt, index=index, columns=columns)
            return test_confusion_matrix
        
    
    def get_user_startup_pred(self, clf_type, category, total_funding, city, funding_rounds, founded_at, first_funding_at, last_funding_at):
        startup_info = {
            'category_list': [category.title()],
            'funding_total_usd': [float(total_funding)],
            'city': [city.title()],
            'funding_rounds': [int(funding_rounds)],
            'founded_at': [founded_at],
            'first_funding_at': [first_funding_at],
            'last_funding_at': [last_funding_at]
        }

        self.logger.info('Gathering user data of startup')

        startup_info = pd.DataFrame(data=startup_info)
        
        if clf_type == 'lr':
            self.logger.debug('Model inference on pipe_lr; fetch results of startup data')
            y_pred_result = self.pipe_lr.predict(startup_info).tolist()
            return y_pred_result
        
        elif clf_type == 'dt':
            self.logger.debug('Model inference on pipe_dt; fetch results of startup_data')
            y_pred_result = self.pipe_dt.predict(startup_info).tolist()
            return y_pred_result
        

    def get_most_important_factors(self, clf_type):
        if clf_type == 'lr':
            coefficients = self.pipe_lr.named_steps['clf'].coef_[0]

            self.logger.info('Processing most influential features of dataset')

            col_names = self.X_train.columns.tolist()
            coef_features = list(zip(col_names, coefficients))

            coef_features.sort(key=lambda x: np.abs(x[-1]), reverse=True)

            most_important_features = coef_features[:3]
            return most_important_features
        
        elif clf_type == 'dt':
            importances = self.pipe_dt.named_steps['clf'].feature_importances_

            self.logger.info('Processing most influential features of dataset')

            col_names = self.X_train.columns.tolist()
            importances_features = list(zip(col_names, importances))

            importances_features.sort(key=lambda x: np.abs(x[-1]), reverse=True)

            most_important_features = importances_features[:3]
            return most_important_features
        

    # Testing purposes only and not to be used by the GUI
    def test_model(self, clf_type):
        startup_info = {
            'category_list': ['Software', 'Cloud Computing', 'FinTech', 'Application Platforms', 'Software'],
            'funding_total_usd': [90000000, 30000000, 10000000, 5000000, 1000000000],
            'city': ['San Francisco', 'Palo Alto', 'Menlo Park', 'San Francisco', 'Palo Alto'],
            'funding_rounds': [3, 2, 3, 1, 4],
            'founded_at': [2022, 2015, 2015, 2010, 2017],
            'first_funding_at': [2022, 2016, 2017, 2010, 2017],
            'last_funding_at': [2024, 2024, 2023, 2022, 2024]
        }

        self.logger.info('Gathering data of startup')

        startup_info = pd.DataFrame(data=startup_info)

        if clf_type == 'lr':
            self.logger.info('predict() and fetch results of the startup\'s status')
            y_pred = self.pipe_lr.predict(startup_info).tolist()
            return y_pred
        
        elif clf_type == 'dt':
            self.logger.info('predict() and fetch results of the startup\'s status')
            y_pred = self.pipe_dt.predict(startup_info).tolist()
            return y_pred


    # Used only once
    def run_train_model(self):
        self.clean_dataset()
        self.get_df_subsets()
        self.train_model()


    # For diagostics of the ML model
    def run_diagnostics_model(self, clf_type):
        model_score = self.get_model_score(clf_type)
        model_confusion_matrix = self.get_model_confusion_matrix(clf_type)

        assert self.test_model(clf_type) == [1.0, 1.0, -1.0, 0.0, 1.0] or self.test_model(clf_type) == [1.0, 1.0, -1.0, -1.0, 1.0], 'Trained model failed to correctly predict startup success'

        most_important_factors = self.get_most_important_factors(clf_type)

        return model_score, model_confusion_matrix, most_important_factors

    
    # To be used repeatedly
    def run_user_model(self, clf_type, category, total_funding, city, funding_rounds, founded_at, first_funding_at, last_funding_at):
        user_startup_pred = self.get_user_startup_pred(clf_type, category, total_funding, city, funding_rounds, founded_at, first_funding_at, last_funding_at)

        if user_startup_pred == [1]:
            return 'Your startup will become a success! You will either sell your company or go public (IPO)!'
        elif user_startup_pred == [0]:
            return 'Unfortunately, your startup will fail. Your company will close down before your get to see its full potential!'
        elif user_startup_pred == [-1]:
            return 'Your startup will be up and running! You will neither sell your company nor go public (IPO)!'


if __name__ == '__main__':
    startup_success_pred_model = StartupSuccessPredModel(load_dataset())
    startup_success_pred_model.run_train_model()

    diagnostics = startup_success_pred_model.run_diagnostics_model('lr')
    for diagnostic in diagnostics:
        print()
        print(diagnostic)
        print()

    print(startup_success_pred_model.run_user_model('lr', 'Cloud Computing', 70000000, 'Palo Alto', 3, 2022, 2022, 2024))
    








      









        

