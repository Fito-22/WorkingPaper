import mlflow
import glob
import os


from colorama import Fore, Style

from tensorflow.keras import Model, models

LOCAL_REGISTRY_PATH = os.environ.get("LOCAL_REGISTRY_PATH")


def load_model(save_copy_locally=False) -> Model:
    """
    load the latest saved model, return None if no model found
    """
    if os.environ.get("MODEL_TARGET") == "mlflow":
        stage = "Production"

        print(Fore.BLUE + f"\nLoad model {stage} stage from mlflow..." + Style.RESET_ALL)

        # retrieve mlflow env params
        MLFLOW_TRACKING_URI = os.environ.get('MLFLOW_TRACKING_URI')
        model_uri = "models:/taxifare_model/Production"             #"models:/"+os.environ.get('MLFLOW_MODEL_NAME')+"/Production"

        # configure mlflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        # load model from mlflow
        model = mlflow.keras.load_model(model_uri=model_uri)

        return model

    print(Fore.BLUE + "\nLoad model from local disk..." + Style.RESET_ALL)

    # get latest model version
    model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models")

    results = glob.glob(f"{model_directory}/*")
    if not results:
        return None

    model_path = sorted(results)[-1]
    print(f"- path: {model_path}")

    model = models.load_model(model_path)
    print("\n✅ model loaded from disk")

    return model
