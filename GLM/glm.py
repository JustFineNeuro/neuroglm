# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import OneHotEncoder, SplineTransformer, PolynomialFeatures
# from sklearn.compose import ColumnTransformer

import numpy as np
import pandas as pd
from GLM.utils import PatsyTransformer, calculate_aic_bic_poisson
from sklearn.metrics import mean_poisson_deviance
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import TimeSeriesSplit,train_test_split, cross_val_score, GridSearchCV


# TODO: Out fo sample testing on models using X_test and y_test
# TODO circular shift
# TODO: stimulis history coding


class PoissonGLM:
    def __init__(self):
        '''

        :param spl_df: List indexing continuous variables and indexing number of spline bases to use
        :param spl_order: List indexing continuous variables and indexing order of spline bases to use
        '''
        self.fit_params = None
        self.test_size = None
        self.scores = None
        self.formulas = None
        self.pipeline = None
        self.X = None
        self.y = None

    def add_data(self, X=None, y=None):
        '''
        Add data before splitting
        :return:
        '''
        self.X = X
        self.y = y

        return self
    
    
    def split_test(self,test_size=0.2):
        '''
        Add data before splitting
        :return:
        '''
        self.test_size = test_size
        self.X, self.X_test, self.y, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=42)

        return self


    def make_preprocessor(self, formulas=None, metric='cv', l2reg=0.0001):
        '''

        :param metric: 'cv','score'
        :param l2reg: l2 regularization
        :param formulas: model formulas in patsy format
        :return:
        '''
        self.formulas = formulas
        if type(formulas) is list:
            '''
            For model selection 
            '''
            if metric == 'cv':
                # Build the pipeline
                self.pipeline = Pipeline([
                    ('patsy', PatsyTransformer(formula=formulas[0])),  # Placeholder formula
                    ('model', PoissonRegressor(alpha=l2reg))
                ])
            elif metric == 'score':
                #Make a list of pipelines to iterate over
                pipelines = []
                for formula in formulas:
                    pipelines.append(Pipeline([
                        ('patsy', PatsyTransformer(formula=formula)),  # Placeholder formula
                        ('model', PoissonRegressor(alpha=l2reg))
                    ]))
                self.pipeline = pipelines

        elif type(formulas) is str:
            self.pipeline = Pipeline([
                ('patsy', PatsyTransformer(formula=formulas)),  # Placeholder formula
                ('model', PoissonRegressor(alpha=l2reg))
            ])

        return self

    def fit(self, params={'cv': 5, 'shuffleTime': True}):
        '''
        Main call to fit a model
        :param params:
        :return:
        '''
        self.fit_params = params

        # Optimizing over different models via cross-validation
        if type(self.formulas) is list:
            # Do cross-validation metrics
            if params['cv'] > 0:
                self.scores = pd.DataFrame(columns=['mean', 'std', 'model'])
                # Setup all formula to optimize over
                param_grid = {
                    'patsy__formula': self.formulas
                }

                # Are we using TimeSeriesShuffle?
                if params['shuffleTime'] is True:
                    cv = TimeSeriesSplit(n_splits=params['cv'])

                    # Grid search
                    grid_search = GridSearchCV(
                        estimator=self.pipeline,
                        param_grid=param_grid,
                        scoring='neg_mean_poisson_deviance',
                        cv=cv,
                        n_jobs=-1
                    )

                else:
                    # Grid search
                    grid_search = GridSearchCV(
                        estimator=self.pipeline,
                        param_grid=param_grid,
                        scoring='neg_mean_poisson_deviance',
                        cv=params['cv'],
                        n_jobs=-1
                    )

                grid_search.fit(self.X, self.y)
                cv_results = pd.DataFrame(grid_search.cv_results_)
                self.scores = cv_results[['param_patsy__formula', 'mean_test_score', 'std_test_score']]
                self.scores = self.scores.rename(columns={'param_patsy__formula':'model'})
                self.best_fit_from_search = grid_search.best_estimator_.fit(self.X, self.y)
                self.best_pipeline = grid_search.best_estimator_

            elif params['cv'] <= 0:

                self.scores = pd.DataFrame(columns=['aic', 'bic', 'model'])

                for i, pipeline in enumerate(self.pipeline, 1):
                    #Fit the formula
                    pipeline.fit(self.X,self.y)

                    # Get pipeline info
                    model = pipeline.named_steps['model']  # Adjust based on the actual name in your pipeline
                    patsy_transformer = pipeline.named_steps['patsy']
                    k = patsy_transformer.transform(self.X).shape[1] + 1
                    #Make predicted data
                    y_pred = pipeline.predict(self.X)
                    [aic, bic] = calculate_aic_bic_poisson(len(self.y), mean_poisson_deviance(self.y, y_pred), k)
                    self.scores.loc[len(self.scores)] = [aic, bic, patsy_transformer.formula]
                    #TODO:  self.best_from_search

        elif type(self.formulas) is str:
            NotImplemented
            # TODO fit a single model
            #TODO: Cv (time or kfold) through cross-val score, or AIC/BIC
        #     if params['shuffleTime'] is True:
        #         cv = TimeSeriesSplit(n_splits=params['cv'])
        #
        # score = cross_val_score(pipeline, X, y, cv=tscv, scoring='neg_mean_poisson_deviance').mean()
        # scores[f'Model {i}'] = score
    # self.pipeline.fit(self.X, self.y)
        return self

    def predict(self, pred_data,predict_params={'data':'X','whichmodel':'best'}):
        #TODO: make an intelligent search over all model terms and make predicted tuning curves
        '''
        predict at specific levels to make estimated curves to comapre against empirical
        '''
        design_matrix = self.best_pipeline[:-1].transform(pred_data)  # Transform without the final model step
        model = self.best_pipeline.named_steps['model']
        linear_predictor = model.intercept_ + design_matrix.dot(model.coef_)
        self.predicted_y = np.exp(linear_predictor)  # Poisson GLM applies exponential to linear predictor






