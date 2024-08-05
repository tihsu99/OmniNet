import tensorflow as tf
import math

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, schedule_fn):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.schedule_fn = schedule_fn

    def __call__(self, step):
        return self.initial_learning_rate * self.schedule_fn(step)

def get_constant_schedule(initial_learning_rate):
    def schedule_fn(step):
        return 1.0

    return CustomSchedule(initial_learning_rate, schedule_fn)

def get_constant_schedule_with_warmup(initial_learning_rate, num_warmup_steps):
    def schedule_fn(step):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        return 1.0

    return CustomSchedule(initial_learning_rate, schedule_fn)

def get_linear_schedule_with_warmup(initial_learning_rate, num_warmup_steps, num_training_steps):
    def schedule_fn(step):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - step) / float(max(1, num_training_steps - num_warmup_steps)))

    return CustomSchedule(initial_learning_rate, schedule_fn)

def get_cosine_schedule_with_warmup(initial_learning_rate, num_warmup_steps, num_training_steps, num_cycles=0.5):
    def schedule_fn(step):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        progress = float(step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return CustomSchedule(initial_learning_rate, schedule_fn)

def get_cosine_with_hard_restarts_schedule_with_warmup(initial_learning_rate, num_warmup_steps, num_training_steps, num_cycles=1):
    def schedule_fn(step):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        progress = float(step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        if progress >= 1.0:
            return 0.0
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))))

    return CustomSchedule(initial_learning_rate, schedule_fn)

