import torch
import acData
import acModel
import torch.optim as Optim
import torch.utils.data as Data
import torch.autograd.variable as Variable
import aConfigration
import os

MODEL_SAVED_PATH  = "/model_saved"
MODEL_SAVED_FILE = "/alexmodel.pkl"
DATA_ROOT_PATH = '../datas'
isGPU = False

if torch.cuda.is_available():
    isGPU = True


def train():
    alexnet.train()

    for index, (x, y) in enumerate(trainLoader, 0):


        # 补充处理数据 , 现在x是torch的tensor，y是int数字。
        x = x.view(-1, 3, aConfigration.IMAGE_SIZE, aConfigration.IMAGE_SIZE)
        x = x.type(torch.FloatTensor)
        y = y.type(torch.LongTensor)

        if isGPU:
            x = x.cuda()
            y = y.cuda()

        optim.zero_grad()
        output = alexnet(x)
        loss = criterion(output, y)
        loss.backward()
        # optim.step()
        lrSchedule.step(loss)
        print("batch: " + str(index))
        print('loss is ' + str(loss))

    print("epoch: " + str(i))
    torch.save(alexnet, DATA_ROOT_PATH + MODEL_SAVED_PATH + MODEL_SAVED_FILE)


def val():
    alexnet.eval()

    for index, (x, y) in enumerate(valLoader, 0):
        # 补充处理数据 , 现在x是torch的tensor，y是int数字。
        x = x.view(-1, 3, aConfigration.IMAGE_SIZE, aConfigration.IMAGE_SIZE)
        x = x.type(torch.FloatTensor)
        y = y.type(torch.LongTensor)

        if isGPU:
            x = x.cuda()
            y = y.cuda()

        predict = alexnet(x)
        loss = criterion(predict, y)
        print('validation data loss is ' + str(loss))


if __name__ == '__main__':
    # Tdata, Tlabel = acData.myDataSet(TRAIN)   #错，dataset不是这么用
    trainDataset = acData.myDataSet(aConfigration.TRAIN)
    trainLoader = Data.DataLoader(trainDataset, aConfigration.BATCH_SIZE, shuffle=True)
    valDataset = acData.myDataSet(aConfigration.EVAL)
    valLoader = Data.DataLoader(valDataset, aConfigration.BATCH_SIZE, shuffle=True)

    alexnet = acModel.build_model()     # model初始化
    # optim = Optim.Adam(alexnet.parameters(), lr=aConfigration.LR)   # optim初始化
    optim = Optim.Adam([{'params':alexnet.features.parameters(), 'lr': aConfigration.LR_FINETUNE_LAYER},
                        {'params':alexnet.classifier.parameters()}], lr=aConfigration.LR)   # 添加分层lr
    lrSchedule = Optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min',
                                                      factor=aConfigration.LR_DECAY,
                                                      patience=aConfigration.LR_SCHEDULE_PATIENCE,
                                                      verbose=True )
    criterion = torch.nn.CrossEntropyLoss()         # loss初始化

    if isGPU:
        alexnet.cuda()


    if not os.path.exists(DATA_ROOT_PATH + MODEL_SAVED_PATH):
        os.mkdir(DATA_ROOT_PATH + MODEL_SAVED_PATH)
        print("create a 'modelSaved' file ")


    for i in range(aConfigration.EPOCH):
        train()
        val()
