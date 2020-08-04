import dataclasses
import tensorflow as tf

@dataclasses.dataclass
class DatasetSetup:
    train_dataset: tf.data.Dataset
    train_examples: int
    train_steps: int

    val_dataset: tf.data.Dataset
    val_examples: int
    val_steps: int

