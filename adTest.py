import torch
import acData
import acModel
import aConfigration
import numpy as np
import json
import os
from tqdm import tqdm

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
        test_batch = testTs[site[0]:site[1]]
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

def outputJson(listPre):
    listPreNp = np.array(listPre)
    list = []
    i = 0
    for pre in range(len(listPre)):
        dict = {}
        dict['image_id'] = testImgName[i]
        dict['disease_class'] = str(pre)

        i += 1
        list.append(dict)
    print('dict: '+ str(list))

    if not os.path.exists(DATA_ROOT_PATH + SUBMIT):
        os.mkdir(DATA_ROOT_PATH + SUBMIT)

    jsFile = open(DATA_ROOT_PATH + SUBMIT + SUBMIT_JOSN, 'w', encoding='utf-8')
    res = json.dumps(list, ensure_ascii=False)
    jsFile.write(res)
    print('dict: '+ str(list))



def test2():
    print(6)

    print("see shape: "+str(testTs.shape))
    prediction = model(testTs.cuda())
    print(7)
    pres = []
    for i in range(len(prediction)):
        #print('see predict: '+str(prediction[i][:20]))
        pre = prediction[i]
        pre = pre.detach().cpu().numpy().argmax()
        pres.append(pre)
    print('see index: '+str(prediction.argmax(1)))
    outputJson(pres)


if __name__ == '__main__':
    testNp, testImgName = acData.readTestPic()
    print(1)
    testTs = torch.from_numpy(testNp)
    print(2)
    testTs = testTs.view(-1, 3, aConfigration.IMAGE_SIZE, aConfigration.IMAGE_SIZE)
    testTs = testTs.type(torch.FloatTensor)
    print(3)
    model = torch.load(DATA_ROOT_PATH + MODEL_SAVED_PATH)
    model.eval()
    print(4)

    if isGPU:
        model.cuda()
    print(5)

    test2()