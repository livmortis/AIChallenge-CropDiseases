import torch
import acData
import acModel
import aConfigration
import numpy as np
import json
import os
from tqdm import tqdm
import torch.utils.data as Data

MODEL_SAVED_PATH  = "/model_saved/alexmodel.pkl"
DATA_ROOT_PATH = '../datas'
SUBMIT = '/submit'
# SUBMIT_JOSN = '/submitjson.json'
SUBMIT_JOSN = '/submitjson_eval.json'   #temper

isGPU = False

if torch.cuda.is_available():
    isGPU = True

def test():
    listPre = []
    i = 0
    n = 0
    if aConfigration.PREVIEW_TEST:
        N = 54
    else :
        N = aConfigration.TEST_PIC_NUM

    batch_size = 9
    batch_site = []
    while n < N:
        n += batch_size
        if n < N:
            n1 = n - batch_size
            n2 = n
        else:
            n1 = n2
            n2 = N

        batch_site.append([n1, n2])


    prediction = []

    # for testItem in testTs:
    for site in tqdm(batch_site):
        # test_batch = testTs[site[0]:site[1]]
        test_batch = test_batch.view(-1, 3, aConfigration.IMAGE_SIZE, aConfigration.IMAGE_SIZE)


        # testItem = testItem.unsqueeze(dim=0)    #重要。

        if isGPU:
            test_batch = test_batch.cuda()
        print("size: "+ str(test_batch.shape))
        prediction_batch = model(test_batch)
        print("look san xia: ", str(prediction_batch))
        # prediction = model(testItem)
        print('prediction\'s index is : '+ str(i))
        i += 1

        prediction_batch = prediction_batch.cpu().data.numpy()

        # listPre.append(prediction)
        for out in prediction_batch:
            K = 5
            index = np.argpartition(out, -K)[-K:]
            prediction.append(index)
        # print('prediction :' + str(prediction))

    # outputJson(listPre)
    outputJson(prediction)

def outputJson(listPre  , testImgName):
    listPreNp = np.array(listPre)
    list = []
    i = 0
    for pre in listPre:
        dict = {}
        dict['image_id'] = testImgName[i]
        dict['disease_class'] = str(pre)

        i += 1
        list.append(dict)
    # print('dict: '+ str(list))

    if not os.path.exists(DATA_ROOT_PATH + SUBMIT):
        os.mkdir(DATA_ROOT_PATH + SUBMIT)

    jsFile = open(DATA_ROOT_PATH + SUBMIT + SUBMIT_JOSN, 'w', encoding='utf-8')
    res = json.dumps(list, ensure_ascii=False)
    jsFile.write(res)
    print('dict: '+ str(list[:50]))



def test2():
    print(6)
    prediction = []
    for index , (testdata , testImgName) in enumerate(testLoader):
        # testdata = torch.from_numpy(testdata)
        testdata = testdata.view(-1, 3, aConfigration.IMAGE_SIZE, aConfigration.IMAGE_SIZE)
        testdata = testdata.type(torch.FloatTensor)

        if torch.cuda.is_available():
            predictBatch = model(testdata.cuda())
        else:
            predictBatch = model(testdata)

        for preOne in predictBatch:
            if torch.cuda.is_available():
                preOneIdx = preOne.detach().cpu().numpy().argmax()
            else:
                preOneIdx = preOne.detach().numpy().argmax()

            prediction.append(preOneIdx)

    print(7)
    # pres = []
    # for i in range(len(prediction)):
    #     pre = prediction[i]
    #     pre = pre.detach().cpu().numpy().argmax()
    #     pres.append(pre)
    # print('see index: '+str(prediction.argmax(1)))
    outputJson(prediction , testImgName)


if __name__ == '__main__':
    # 更改数据加载为dataset法
    # testNp, testImgName = acData.readTestPic()
    testSet = acData.myTestSet()
    testLoader = Data.DataLoader(testSet, batch_size=aConfigration.BATCH_SIZE , shuffle=False)


    # 移到遍历loader的每个batch中处理。
    # print(1)
    # testTs = torch.from_numpy(testNp)
    # print(2)
    # testTs = testTs.view(-1, 3, aConfigration.IMAGE_SIZE, aConfigration.IMAGE_SIZE)
    # print(2.5)
    # testTs = testTs.type(torch.FloatTensor)




    #下面防止内存爆炸
    # print(str(len(testTs) / 3))
    # testTs1 = testTs[0: int(len(testTs) / 3)]
    # testTs2 = testTs[int(len(testTs) / 3) : int(len(testTs) / 3 * 2)]
    # testTs3 = testTs[int(len(testTs) / 3 * 2) : int(len(testTs) / 3)]
    #
    # testTs1 = testTs1.type(torch.FloatTensor).cuda(0)
    # testTs2 = testTs2.type(torch.FloatTensor).cuda(1)
    # testTs3 = testTs3.type(torch.FloatTensor).cuda(2)




    print(3)
    model = torch.load(DATA_ROOT_PATH + MODEL_SAVED_PATH)
    model.eval()
    print(4)

    if isGPU:
        model.cuda()
    print(5)

    test2()