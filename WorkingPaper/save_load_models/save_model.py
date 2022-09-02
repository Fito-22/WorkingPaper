

import mlflow
from mlflow.tracking import MlflowClient

import glob
import os
import time
import pickle

from colorama import Fore, Style

from tensorflow.keras import Model, models

LOCAL_REGISTRY_PATH = os.environ.get("LOCAL_REGISTRY_PATH")

def save_model(model: Model = None,
               params: dict = None,
               metrics: dict = None) -> None:
    """
    persist trained model, params and metrics
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    if os.environ.get("MODEL_TARGET") == "mlflow":

        # retrieve mlflow env params
        MLFLOW_TRACKING_URI = os.environ.get('MLFLOW_TRACKING_URI')
        MLFLOW_EXPERIMENT = os.environ.get('MLFLOW_EXPERIMENT')
        # configure mlflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(experiment_name=MLFLOW_EXPERIMENT)


        with mlflow.start_run():

            # STEP 1: push parameters to mlflow
            mlflow.log_params(params)

            # STEP 2: push metrics to mlflow
            mlflow.log_metrics(metrics)

            # STEP 3: push model to mlflow
            mlflow.keras.log_model(keras_model=model,
                           artifact_path="model",
                           keras_module="tensorflow.keras",
                           registered_model_name="taxifare_model")

        return None

    print(Fore.BLUE + "\nSave model to local disk..." + Style.RESET_ALL)

    # save params
    if params is not None:
        params_path = os.path.join(LOCAL_REGISTRY_PATH, "params", timestamp + ".pickle")
        print(f"- params path: {params_path}")
        with open(params_path, "wb") as file:
            pickle.dump(params, file)

    # save metrics
    if metrics is not None:
        metrics_path = os.path.join(LOCAL_REGISTRY_PATH, "metrics", timestamp + ".pickle")
        print(f"- metrics path: {metrics_path}")
        with open(metrics_path, "wb") as file:
            pickle.dump(metrics, file)

    # save model
    if model is not None:
        model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", timestamp)
        print(f"- model path: {model_path}")
        model.save(model_path)

    print("\n✅ data saved locally")

    return None
