from .dataset import AcousticDataset
from .collator import AcousticCollator

DATASETS = {name: cls for name, cls in globals().items() if ".dataset." in str(cls)}
COLLATORS = {name: cls for name, cls in globals().items() if ".collator." in str(cls)}