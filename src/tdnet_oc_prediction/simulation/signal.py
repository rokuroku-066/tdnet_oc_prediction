import numpy as np
class SignalGenerator:
    def generate(self, predictions, threshold_long, threshold_short, allow_short=True):
        sig = np.where(predictions['y_proba'] >= threshold_long, 1, 0)
        if allow_short:
            sig = np.where(predictions['y_proba'] <= threshold_short, -1, sig)
        out = predictions.copy(); out['signal'] = sig
        return out
