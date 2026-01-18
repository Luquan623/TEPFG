import io
import logging


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, trace_func=print, tmpFile=io.BytesIO()):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score = float("-inf")
        self.delta = delta
        self.tmpFile = tmpFile
        self.trace_func = trace_func

    def __call__(self, score, model, epoch):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            logging.warning(
                f"EarlyStopping counter: {self.counter} out of {self.patience}, at epoch {epoch}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def get_best(self, model):
        if isinstance(self.tmpFile, io.IOBase):
            self.tmpFile.seek(0)
        return model.load(self.tmpFile)

    def save_checkpoint(self, val_score, model):
        if self.verbose:
            logging.info(
                f"Validation metric improved {self.val_score:.6f} -> {val_score:.6f}. Saving model."
            )
        if isinstance(self.tmpFile, io.IOBase):
            self.tmpFile.seek(0)
        model.save(self.tmpFile)
        self.val_score = val_score
