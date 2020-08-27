import tensorflow as tf
import tensorflow.keras.backend as K

def make_stepwise(base_lr, boundaries, multiplier):
    def sched(epoch):
        boundaries_passed = len([b for b in boundaries if b <= epoch])
        return base_lr * (multiplier ** boundaries_passed)

    return sched

def make_1cycle(min_lr, max_lr, final_anneal, total_epochs):
    def sched(epoch):
        half_cycle = (total_epochs - final_anneal) // 2
        if epoch <= half_cycle:
            return min_lr + epoch / half_cycle * (max_lr - min_lr)
        if epoch > half_cycle and epoch + final_anneal < total_epochs:
            return max_lr + (epoch - half_cycle) / half_cycle * (min_lr - max_lr)
        return (total_epochs - epoch) / final_anneal * min_lr

    return sched

class LRandWDScheduler(tf.keras.callbacks.Callback):
    def __init__(self, multiplier_schedule, base_lr, base_wd, fp16):
        super(LRandWDScheduler, self).__init__()
        self.multiplier_schedule = multiplier_schedule
        self.base_lr = base_lr
        self.base_wd = base_wd
        self.fp16 = fp16

    def on_epoch_begin(self, epoch, logs=None):
        multiplier = self.multiplier_schedule(epoch)
        optimizer = self.model.optimizer if not self.fp16 else self.model.optimizer._optimizer

        K.set_value(optimizer.lr, self.base_lr * multiplier)
        K.set_value(optimizer.weight_decay, self.base_wd * multiplier)

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        optimizer = self.model.optimizer if not self.fp16 else self.model.optimizer._optimizer

        logs['lr'] = K.get_value(optimizer.lr)
        logs['weight_decay'] = K.get_value(optimizer.weight_decay)
