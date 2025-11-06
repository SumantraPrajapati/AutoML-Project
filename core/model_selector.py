from sklearn.utils.multiclass import type_of_target

def detect_task(y):
    """Detects if the target is classification or regression"""
    task = type_of_target(y)
    if task in ["binary", "multiclass"]:
        return "classification"
    else:
        return "regression"
