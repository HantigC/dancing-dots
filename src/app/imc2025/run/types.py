from enum import Enum


class RunType(str, Enum):
    TRAIN = "train"
    TEST = "test"
    SUBMISSION = "submission"
