

IMAGE_SIZE = 224
IMAGE_SIZE_COPY = 650
TRAIN = "train"
EVAL = "eval"
LR = 0.001
LR_DECAY = 0.1
LR_FINETUNE_LAYER = 0.0001
LR_SCHEDULE_PATIENCE = 5
WEIGHT_DECAY = 0.0001
BATCH_SIZE = 128
LABEL_NUMS = 61
EPOCH = 80

PREVIEW = False          # 正式训练时为False
PREVIEW_TEST = False    # 正式预测时为False
PREVIEW_TRAIN_NUM = 300
PREVIEW_TEST_NUM = 54

# TEST_PIC_NUM = 4959
TEST_PIC_NUM = 4982  # temper, neval number

NEED_RESTART_READ_TRAIN_DATA = True   # 第一次加载图片时为true。
NEED_RESTART_READ_TEST_DATA = False   # 第一次加载图片时为true。