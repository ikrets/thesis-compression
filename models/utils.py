def make_stepwise(base_lr, boundaries, multiplier):
    def sched(epoch):
        boundaries_passed = len([b for b in boundaries if b <= epoch])
        return base_lr * (multiplier ** boundaries_passed)

    return sched