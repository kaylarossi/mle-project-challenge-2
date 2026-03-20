import os
import sklearn.metrics
import numpy as np

class Metrics_Summary:
    """Class for calculating and storing model performance metrics."""

    def __init__(self, y_true, y_pred, model_name=None):
        self.y_true = y_true
        self.y_pred = y_pred
        self.model_name = model_name

    def _calculate_mae(self):
        """Calculate mean absolute error."""
        return sklearn.metrics.mean_absolute_error(self.y_true, self.y_pred)
    
    def _calc_r2(self):
        """Calculate R-squared."""
        return sklearn.metrics.r2_score(self.y_true, self.y_pred)
    
    def _calc_rmse(self):
        """Calculate root mean squared error."""
        return np.sqrt(sklearn.metrics.mean_squared_error(self.y_true, self.y_pred))

    def _calc_mape(self):
        """Calculate mean absolute percentage error."""
        return np.mean(np.abs((self.y_true - self.y_pred) / self.y_true)) * 100
    
    def as_dict(self):
        """Return metrics as a dictionary to be logged to comet."""
        return {
            "mean_absolute_error": (self._calculate_mae()).round(4),
            "r_squared": (self._calc_r2()).round(4),
            "root_mean_squared_error": (self._calc_rmse()).round(4),
            "mean_absolute_percentage_error": (self._calc_mape()).round(4)
        }
    
    def print_summary(self):
        """Print a summary of the metrics."""
        if self.model_name:
            print(f"Metrics for {self.model_name}:")
        print(f"Mean Absolute Error: {self._calculate_mae():.4f}")
        print(f"R-squared: {self._calc_r2():.4f}")
        print(f"Root Mean Squared Error: {self._calc_rmse():.4f}")
        print(f"Mean Absolute Percentage Error: {self._calc_mape():.4f}")