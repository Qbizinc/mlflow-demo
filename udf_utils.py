from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
import sklearn.metrics as sk_metrics
import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
import os


class ML_FLOW:
    """"
    Args:
        problem_type ({'regression', classification}): The type of problem to work.
    """

    def __init__(self, problem_type):
        assert problem_type in ['regression', 'classification'], 'Should use one of the accepted problems'
        self.problem_type = problem_type

    def generate_data(self, n_samples = 1000, n_features = 20, n_informative = 16, n_classes = 2, test_size = 0.3):
        """ Function to generate a dataset for the demo

        Args:
           n_samples (int): The number of samples in the dataset. Default = 1000
           n_features (int): The number of features in the dataset. Default = 20
           n_informative (int): The number of informative featuers. Default 16.
           n_classes (int): The number of classes (labels) for a classification problem. Default 2
           test_size (float or int):
               If float, it should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the train split.
               If int, it represents the absolute number of test samples
        """
        if self.problem_type == 'regression':
            self.n_classes = None
            x, y = make_regression(n_samples = n_samples, n_features = n_features, n_informative = n_informative)
        elif self.problem_type == 'classification':
            self.n_classes = n_classes
            x, y = make_classification(n_samples = n_samples, n_features = n_features, n_informative = n_informative, n_classes = n_classes)


        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size = test_size)

        return self

    def _eval_metrics(self, y, yhat, prefix_name = 'train'):
        """ Function to define the metrics to consider in the demo

        Args:
            y (ndarray): 1D array with the target values
            yhat (ndarray): 1D array with the predicted values
            prefix_name (str): The prefix to use in the name of each metric, e.g. train, test, validation, etc.

        Returns:
            Dictionary with the results of the metrics according to the problem type
        """

        if self.problem_type == 'regression':
            mse = sk_metrics.mean_squared_error(y, yhat)
            rmse = np.sqrt(mse)
            mae = sk_metrics.mean_absolute_error(y, yhat)
            r2 = sk_metrics.r2_score(y, yhat)
            results_metrics = {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}

        elif self.problem_type == 'classification':
            roc_auc = sk_metrics.roc_auc_score(y, yhat, average = 'weighted')
            accuracy = sk_metrics.accuracy_score(y, yhat)
            f1 = sk_metrics.f1_score(y, yhat, average = 'weighted')
            results_metrics = {'roc_auc': roc_auc, 'accuracy': accuracy, 'f1_score': f1}

            cm = sk_metrics.confusion_matrix(y, yhat)
            if self.n_classes == 2:
                cm_binary_metrics = ['tp', 'fp', 'fn', 'tp']
                cm_binary_metrics_score = cm.ravel()
                for i in range(len(cf_binary_metrics)):
                    results_metrics[cm_binary_metrics[i]] = cm_binary_metrics_score[i]

        if prefix_name is not None:
            results_metrics_prefix = {}
            for key_name in results_metrics:
                results_metrics_prefix['{}_{}'.format(prefix_name, key_name)] = results_metrics[key_name]
        else:
            results_metrics_prefix = results_metrics

        return results_metrics_prefix

    def _get_probs_pd(self, probs, y, file_name, path_name = ''):

        probs_np = np.concatenate((probs, y.reshape(1, -1).T), axis = 1)

        columns_name = ['prob_class_{}'.format(i) for i in range(self.n_classes)]
        columns_name += ['y']
        probs_pd = pd.DataFrame(probs_np, columns = columns_name)
        full_path = os.path.abspath(path_name + file_name)
        probs_pd.to_csv(full_path)

        return full_path


    def run_experiments(self, experiment_name, run_name, model_function, hyper_params, other_params = None):
        """

        Args:
            experiment_name (str): The name of the experiment to log the information.
            run_name (str): The name given to the MLflow run associated with the project execution.
            model_function (func): A model function to train and evaluate. All the arguments must be hyper-parameters.
            params_model (dict): A dictionary with the information of the hyper-parameters to use in model_function.
            other_params (dict): A dictionary with aditional information that can't be added in the params_model dictionary. Default None.
            artifacts_functions (list of func): A lis of functions to create artifacts from the function given. Default [].
        """

        self.experiment_name = experiment_name
        # If the experiment hasn't been created
        try:
            self.experiment_id = mlflow.create_experiment(self.experiment_name)
        # If experiment is already created
        except MlflowException:
            self.experiment_id = MlflowClient().get_experiment_by_name(self.experiment_name).experiment_id

        # Rutine to start to log the information of the model.
        with mlflow.start_run(experiment_id = self.experiment_id, run_name = run_name):
            mlflow.autolog(disable = True)

            # Log the parameters used in the model
            if other_params is not None:
                mlflow.log_params(other_params)

            mlflow.log_params(hyper_params)

            # Train the model
            model = model_function(**hyper_params)
            model.fit(self.x_train,self.y_train)
            mlflow.sklearn.log_model(model, 'model')

            yhat_train = model.predict(self.x_train)
            yhat_test = model.predict(self.x_test)

            # Evaluation of the model
            results_metrics_train = self._eval_metrics(self.y_train, yhat_train, prefix_name = 'train')
            results_metrics_test = self._eval_metrics(self.y_test, yhat_test, prefix_name = 'test')

            # Log the metrics results from the trained model
            mlflow.log_metrics(results_metrics_train)
            mlflow.log_metrics(results_metrics_test)

        return 'Train and log finished'

    def run_experiments_autolog(self, experiment_name, run_name, model_function, hyper_params, other_params = None):
        """

        Args:
            experiment_name (str): The name of the experiment to log the information.
            run_name (str): The name given to the MLflow run associated with the project execution.
            model_function (func): A model function to train and evaluate. All the arguments must be hyper-parameters.
            params_model (dict): A dictionary with the information of the hyper-parameters to use in model_function.
            other_params (dict): A dictionary with aditional information that can't be added in the params_model dictionary. Default None.
            artifacts_functions (list of func): A lis of functions to create artifacts from the function given. Default [].
        """

        self.experiment_name = experiment_name
        # If the experiment hasn't been created
        try:
            self.experiment_id = mlflow.create_experiment(self.experiment_name)
        # If experiment is already created
        except MlflowException:
            self.experiment_id = MlflowClient().get_experiment_by_name(self.experiment_name).experiment_id

        # Rutine to start to log the information of the model.
        with mlflow.start_run(experiment_id = self.experiment_id, run_name = run_name):
            mlflow.autolog(log_input_examples = True)

            # Log the parameters used in the model
            if other_params is not None:
                mlflow.log_params(other_params)

            # Train the model and log metrics
            model = model_function(**hyper_params)
            model.fit(self.x_train, self.y_train)
            mlflow.sklearn.eval_and_log_metrics(model, self.x_test, self.y_test, prefix = 'test_')

            if self.problem_type == 'classification':
                # Get the probabilities for each dataset
                prob_train = model.predict_proba(self.x_train)
                prob_test = model.predict_proba(self.x_test)

                train_path = self._get_probs_pd(prob_train, self.y_train, 'train.csv')
                test_path = self._get_probs_pd(prob_test, self.y_test, 'test.csv')

                # Log the table into mlflow
                mlflow.log_artifact(train_path, 'tables/probs_train.csv')
                mlflow.log_artifact(test_path, 'tables/probs_test.csv')

        mlflow.end_run()

        return 'Train and log finished'
