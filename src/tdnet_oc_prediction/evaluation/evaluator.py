from .metrics import classification_metrics
class Evaluator:
    def evaluate(self, y_true, y_proba, threshold=0.5):
        return classification_metrics(y_true, y_proba, threshold)
