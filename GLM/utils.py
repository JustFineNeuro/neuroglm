import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from patsy import dmatrices, build_design_matrices


class PatsyTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, formula):
        self.formula = formula
        self.design_info_ = None

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("y cannot be None for PatsyTransformer fit method.")
        data = X.copy()
        data['y'] = y

        # Build design matrices to capture design_info
        y_design, X_design = dmatrices(self.formula, data, return_type='dataframe')

        # Save design_info for later use
        self.design_info_ = X_design.design_info
        return self

    def transform(self, X):
        if self.design_info_ is None:
            raise ValueError("The PatsyTransformer has not been fitted yet.")

        # Use build_design_matrices with the saved design_info
        X_transformed = build_design_matrices([self.design_info_], X)[0]

        # Convert to DataFrame for consistency
        X_transformed = pd.DataFrame(X_transformed, columns=self.design_info_.column_names)
        return X_transformed


def calculate_aic_bic_poisson(n, neg_mean_poisson_deviance, k):
    # Convert mean negative deviance to total log-likelihood
    log_likelihood = -0.5 * n * neg_mean_poisson_deviance

    # Calculate AIC and BIC
    aic = 2 * k - 2 * log_likelihood
    bic = k * np.log(n) - 2 * log_likelihood
    return aic, bic

