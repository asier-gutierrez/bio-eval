import enum


class ModelType(enum.Enum):
    CLM = "CLM"
    MLM = "MLM"


class MaskingStrategy(enum.Enum):
    SWM = "SWM"
    WWM = "WWM"
