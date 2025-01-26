from .acoustic import AcousticModel, AcousticModelEvaluator
from .base import Model

MODELS = {name: cls for name, cls in globals().items() if ".model." in str(cls)}
EVALUATORS = {name: cls for name, cls in globals().items() if ".evaluator." in str(cls)}
