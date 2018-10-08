import torch
import acData
import acModel
import aConfigration
import numpy as np
import json
import os

MODEL_SAVED_PATH  = "/model_saved/alexmodel.pkl"
DATA_ROOT_PATH = '../datas'
SUBMIT = '/submit'
SUBMIT_JOSN = '/submitjson.json'

isGPU = False

if torch.cuda.is_available():
    isGPU = True

def test():
    listPre = []
    for testItem in testTs:


        testItem = testItem.unsqueeze(dim=0)    #重要。

        if isGPU:
            testItem = testItem.cuda()

        prediction = model(testItem)
        # print('prediction is : '+ str(prediction))
        listPre.append(prediction)

    outputJson(listPre)

def outputJson(listPre):
    listPreNp = np.array(listPre)
    dict = {}
    list = []
    i = 0
    for pre in range(len(listPre)):

        dict['disease_class'] = pre
        dict['image_id'] = testImgName[i]

        i += 1
        list.append(dict)

    if not os.path.exists(DATA_ROOT_PATH + SUBMIT):
        os.mkdir(DATA_ROOT_PATH + SUBMIT)

    jsFile = open(DATA_ROOT_PATH + SUBMIT + SUBMIT_JOSN, 'w', encoding='utf-8')
    res = json.dumps(list, ensure_ascii=False)
    jsFile.write(res)

    print('dict: '+ str(list))

if __name__ == '__main__':
    testNp, testImgName = acData.readTestPic()
    testTs = torch.from_numpy(testNp)
    testTs = testTs.view(-1, 3, aConfigration.IMAGE_SIZE, aConfigration.IMAGE_SIZE)
    testTs = testTs.type(torch.FloatTensor)

    model = torch.load(DATA_ROOT_PATH + MODEL_SAVED_PATH)
    model.eval()

    if isGPU:
        model.cuda()
    test()