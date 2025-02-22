import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns


class LinearRegressor:
    """
    Extended Linear Regression model with support for categorical variables and gradient descent fitting.
    """

    def __init__(self):
        self.coefficients = None
        self.intercept = None

    """
    This next "fit" function is a general function that either calls the *fit_multiple* code that
    you wrote last week, or calls a new method, called *fit_gradient_descent*, not implemented (yet)
    """

    def fit(self, X, y, method="least_squares", learning_rate=0.01, iterations=1000):
        """
        Fit the model using either normal equation or gradient descent.

        Args:
            X (np.ndarray): Independent variable data (2D array).
            y (np.ndarray): Dependent variable data (1D array).
            method (str): method to train linear regression coefficients.
                          It may be "least_squares" or "gradient_descent".
            learning_rate (float): Learning rate for gradient descent.
            iterations (int): Number of iterations for gradient descent.

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """
        if method not in ["least_squares", "gradient_descent"]:
            raise ValueError(
                f"Method {method} not available for training linear regression."
            )
        if np.ndim(X) == 1:
            X = X.reshape(-1, 1)

        X_with_bias = np.insert(
            X, 0, 1, axis=1
        )  # Adding a column of ones for intercept

        if method == "least_squares":
            self.fit_multiple(X_with_bias, y)
        elif method == "gradient_descent":
            self.fit_gradient_descent(X_with_bias, y, learning_rate, iterations)

    def fit_multiple(self, X, y):
        """
        Fit the model using multiple linear regression (more than one independent variable).

        This method applies the matrix approach to calculate the coefficients for
        multiple linear regression.

        Args:
            X (np.ndarray): Independent variable data (2D array), with bias.
            y (np.ndarray): Dependent variable data (1D array).

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """
        # Replace this code with the code you did in the previous laboratory session

        # Store the intercept and the coefficients of the model
        X_transpose = np.transpose(X)
        X_transpose_dot_X = X_transpose.dot(X)
        inverse_X_transpose_dot_X = np.linalg.inv(X_transpose_dot_X)
        X_transpose_dot_y = X_transpose.dot(y)

        coefficients = inverse_X_transpose_dot_X.dot(X_transpose_dot_y)
        
        # Store the intercept and the coefficients of the model
        self.intercept = coefficients[0]
        self.coefficients = coefficients[1:]

    def fit_gradient_descent(self, X, y, learning_rate=0.01, iterations=1000):
        """
        Fit the model using either normal equation or gradient descent.

        Args:
            X (np.ndarray): Independent variable data (2D array), with bias.
            y (np.ndarray): Dependent variable data (1D array).
            learning_rate (float): Learning rate for gradient descent.
            iterations (int): Number of iterations for gradient descent.

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """

        # Initialize the parameters to very small values (close to 0)
        m = len(y)
        self.coefficients = np.random.rand(X.shape[1] - 1) * 0.01  # Small random numbers
        self.intercept = np.random.rand() * 0.01
        epochs = []
        mses = []
        w = []
        b = []
        w_recta = []
        b_recta = []

        # Implement gradient descent (TODO)
        for epoch in range(iterations):
            predictions = X[:,1:]@self.coefficients + self.intercept
            error = predictions - y
            
            # TODO: Write the gradient values and the updates for the paramenters
            gradient_coefficients = (1/m)*np.transpose(X[:,1:])@error
            gradient_intercept = (1/m)*np.sum(error) 
            self.intercept -= learning_rate*gradient_intercept
            self.coefficients -= learning_rate*gradient_coefficients

            # TODO: Calculate and print the loss every 10 epochs
            if epoch % 10 == 0:
                mse = np.mean(error**2)
                epochs.append(epoch)
                mses.append(mse)
                print(f"Epoch {epoch}: MSE = {mse}")

                if len(self.coefficients) == 1:
                    w.append(self.coefficients[0])
                    b.append(self.intercept)

                    if epoch % 100 == 0:
                        w_recta.append(self.coefficients[0])
                        b_recta.append(self.intercept)


        if len(self.coefficients) == 1:
            # Plot the predicted values with the real values every 100 steps
            plt.figure(figsize = (20,  10))
            plt.scatter(X[:, 1:], y, marker='o', color="purple")
            x_linspace = np.linspace(np.min(X[:, 1:]), np.max(X[:, 1:]), 100)
            for i in range(len(w_recta)):
                plt.plot(x_linspace, w_recta[i]*x_linspace+b_recta[i], label = f"predictions step {i*100}")
                
            plt.title("Fitting process")
            plt.xlabel("X true")
            plt.ylabel("Y true")
            plt.legend()
            plt.show()

            # Plot w and b values as well as the level curves from the mse
            def mse_function(w0, w1, X, y):  # Definir función del MSE
                y_pred = w0 + w1 * X
                mse = np.mean((y_pred - y) ** 2)
                return mse

            w0_range = np.linspace(-2, 4, 100)  # Definir rangos de parámetros
            w1_range = np.linspace(-1, 2, 100)

            W0, W1 = np.meshgrid(w0_range, w1_range)  # Crear una malla de puntos para los parámetros
            Z = np.zeros_like(W0)  # Calcular el MSE para cada combinación de parámetros
            X = X[:,1:]
            for i in range(W0.shape[0]):
                for j in range(W0.shape[1]):
                    Z[i, j] = mse_function(W0[i, j], W1[i, j], X[:, 0], y)

            plt.figure(figsize = (10,5))
            plt.contour(W0, W1, Z, levels=30, cmap='viridis')  # Pintar curvas de nivel con contour
            plt.scatter(w, b, marker="x", color="red")  # Pintar w y b
            plt.legend()
            plt.title("Weight and Bias")
            plt.xlabel("Weight")
            plt.ylabel("Bias")
            plt.show()

        plt.figure(figsize=(9,4))
        plt.plot(epochs, mses, color="magenta")
        plt.scatter(epochs, mses, color="black", s=30, marker=".")
        plt.ylabel("Loss function (MSE)")
        plt.xlabel("Epoch")
        plt.title("Progress of the Loss Function")


    def predict(self, X):
        """
        Predict the dependent variable values using the fitted model.

        Args:
            X (np.ndarray): Independent variable data (1D or 2D array).
            fit (bool): Flag to indicate if fit was done.

        Returns:
            np.ndarray: Predicted values of the dependent variable.

        Raises:
            ValueError: If the model is not yet fitted.
        """

        # Paste your code from last week

        if self.coefficients is None or self.intercept is None:
            raise ValueError("Model is not yet fitted")

        if np.ndim(X) == 1:
            # TODO: Predict when X is only one variable
            predictions = self.intercept + self.coefficients*X
        else:
            # TODO: Predict when X is more than one variable
            ones = np.ones((X.shape[0], 1))
            X = np.hstack([ones, X])
            w = np.hstack([self.intercept, self.coefficients])
            predictions = np.dot(X,w) 

        return predictions

def evaluate_regression(y_true, y_pred):
    """
    Evaluates the performance of a regression model by calculating R^2, RMSE, and MAE.

    Args:
        y_true (np.ndarray): True values of the dependent variable.
        y_pred (np.ndarray): Predicted values by the regression model.

    Returns:
        dict: A dictionary containing the R^2, RMSE, and MAE values.
    """

    # R^2 Score
    # TODO
    rss = np.sum((y_true-y_pred)**2)
    tss = np.sum((y_true-np.mean(y_true))**2)
    r_squared = 1-(rss/tss)

    # Root Mean Squared Error
    # TODO
    rmse = np.sqrt(np.mean((y_pred-y_true)**2))

    # Mean Absolute Error
    # TODO
    mae = (1/len(y_true))*np.sum(np.abs(y_true-y_pred))

    return {"R2": r_squared, "RMSE": rmse, "MAE": mae}

def one_hot_encode(X, categorical_indices, drop_first=False):
    """
    One-hot encode the categorical columns specified in categorical_indices. This function
    shall support string variables.

    Args:
        X (np.ndarray): 2D data array.
        categorical_indices (list of int): Indices of columns to be one-hot encoded.
        drop_first (bool): Whether to drop the first level of one-hot encoding to avoid multicollinearity.

    Returns:
        np.ndarray: Transformed array with one-hot encoded columns.
    """
    X_transformed = X.copy()
    for index in sorted(categorical_indices, reverse=True):
        # TODO: Extract the categorical column
        categorical_column = X_transformed[:,index]

        # TODO: Find the unique categories (works with strings)
        unique_values = np.unique(categorical_column)

        # TODO: Create a one-hot encoded matrix (np.array) for the current categorical column
        one_hot = np.array([[1 if val == word else 0 for word in unique_values] for val in categorical_column])


        # Optionally drop the first level of one-hot encoding
        if drop_first:
            one_hot = one_hot[:, 1:]

        # TODO: Delete the original categorical column from X_transformed and insert new one-hot encoded columns
        X_transformed = np.delete(X_transformed, index, axis=1)  # Remove categorical column
        X_transformed = np.hstack((X_transformed[:, :index], one_hot, X_transformed[:, index:]))  # Insert one-hot


    return X_transformed
