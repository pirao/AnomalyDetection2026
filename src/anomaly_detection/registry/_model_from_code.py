"""Model-from-code entry point for ``mlflow.pyfunc.log_model``.

Passed as a file path (not an instantiated object) so mlflow packages this
script instead of CloudPickle-serializing ``AnomalyDetectorBundle``. See
https://mlflow.org/docs/latest/ml/model/models-from-code/.
"""
import mlflow

from anomaly_detection.registry.bundle import AnomalyDetectorBundle

mlflow.models.set_model(AnomalyDetectorBundle())
