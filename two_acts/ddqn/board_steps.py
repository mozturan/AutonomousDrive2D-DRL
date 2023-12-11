import tensorflow as tf
import numpy as np
import os
import time
MODEL_NAME = "ddqn_steps"
# import tensorboard
from keras.callbacks import TensorBoard
# tfe = tf.contrib.eager
# if not tf.executing_eagerly():
#             # Enable tensorflow Eager execution
#             tfe.enable_eager_execution()

# try:
# 	     tf.enable_eager_execution ()
# except Exception:
# 	         pass
# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # *Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self._log_write_dir = os.path.join(self.log_dir, MODEL_NAME)

    # *Overriding this method to stop creating default log writer
    def set_model(self, model):
        self.model = model
        
        self._train_dir = os.path.join(self._log_write_dir, 'train')
        self._train_step = self.model._train_counter 

        self._val_dir = os.path.join(self._log_write_dir, 'validation')
        self._val_step = self.model._test_counter

        pass

    #* Overrided, saves logs with our step number
    #* (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    #* Overrided
    #* We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    #* Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    def on_train_batch_end(self, batch, logs=None):
        pass

    #* Custom method for saving own metrics
    #* Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

    #* Added because of version
    def _write_logs(self, logs, index):
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=index)
                self.step += 1
                self.writer.flush()