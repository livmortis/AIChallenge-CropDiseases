from PIL import Image
import os
import numpy as np
import torch
import torch.utils.data as Data
import json
import aConfigration

dataRootPath = '../datas'
trainPath = '/AgriculturalDisease_trainingset'
validPath = '/AgriculturalDisease_validationset'
testPath = '/AgriculturalDisease_testA'
commonImgPath = '/images/'
trainLabel = '/AgriculturalDisease_train_annotations.json'
validLabel = '/AgriculturalDisease_validation_annotations.json'


def main():
    readTrainAndValPic()
    readTestPic()


def check_contain_chinese(check_str):
    for ch in check_str:
    # for ch in check_str.encode('utf-8'):
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False


def readTrainAndValLabel():
    trainJFile = open(dataRootPath + trainPath + trainLabel)
    valJFile = open(dataRootPath + validPath + validLabel)
    tLabDict = json.load(trainJFile)
    vLabDict = json.load(valJFile)
    print("see "+ str(tLabDict[:20]))
    print("see see "+ str(type(tLabDict)))
    return tLabDict, vLabDict



def readTrainAndValPic():
    # 获取图片名方法一，文件夹里读。 （弃用） 因为文件夹里读出的图片顺序和标签列表中的图片名顺序不一致）
    # listPicTrain = os.listdir(dataRootPath + trainPath + commonImgPath)
    # listPicVal = os.listdir(dataRootPath + validPath + commonImgPath)

    # 获取图片名方法二，读标签json文件里的列表。
    # 类似[{'disease_class': 1, 'image_id': '62fd8bf4d53a1b94fbac16738406f10b.jpg'}, {'disease_class': 1, 'image_id': '0bdec5cccbcade6b6e94087cb5509d98.jpg'},.....]
    listPicTrain = []
    listPicVal = []
    tLabDict, vLabDict = readTrainAndValLabel()

        # 少量数据预览模式
    if(aConfigration.PREVIEW):
        tLabDict = tLabDict[:300]
        vLabDict = vLabDict[:300]


    for TlabItem in tLabDict:
        listPicTrain.append(TlabItem['image_id'])
    for VlabItem in vLabDict:
        listPicVal.append(VlabItem['image_id'])


    # 去除文件名带有“副本”字样的图片
    # 后来发现有中文的不一定是重复的，不删他们了保留着吧。
    # fubenNumT= 0
    # fubenNumV= 0
    #
    # for pic1 in listPicTrain:
    #     if check_contain_chinese(str(pic1)):
    #         listPicTrain.remove(pic1)
    #         print("看下中文："+str(pic1))
    #         fubenNumT += 1
    # for pic2 in listPicVal:
    #     if check_contain_chinese(str(pic2)):
    #         listPicVal.remove(pic2)
    #         fubenNumV += 1
    # print('the train pic list\'s length is ' + str(len(listPicTrain)) +
    #       ' \nand deleted pic is '+str(fubenNumT),
    #       ' \nthe val pic list\'s length is ' + str(len(listPicVal)) +
    #       ' \nand deleted pic is ' + str(fubenNumV)
    #       )

    print('the train pic list\'s length is ' + str(len(listPicTrain)) +
                ' \nthe val pic list\'s length is ' + str(len(listPicVal))
                )




    '''制作标签集'''
    i = 0
    j = 0
    labTNp = np.zeros(len(tLabDict))
    for labTItem in tLabDict:
        lab = labTItem['disease_class']
        labTNp[i] = lab
        i += 1
    labVNp = np.zeros(len(vLabDict))
    for labVItem in vLabDict:
        lab2 = labVItem['disease_class']
        labVNp[j] = lab2
        j += 1





    '''制作数据集'''
    imgTNp = np.zeros([len(listPicTrain), aConfigration.IMAGE_SIZE, aConfigration.IMAGE_SIZE, 3], dtype=int)
    imgVNp = np.zeros([len(listPicVal), aConfigration.IMAGE_SIZE, aConfigration.IMAGE_SIZE, 3], dtype=int)


    t = 0
    # 处理训练集图片:挨个处理大小，转换numpy
    for pic in listPicTrain:
        imageT = Image.open(dataRootPath + trainPath + commonImgPath + pic)
        # 转换非RGB图片
        if imageT.mode != 'RGB':
            imageT = imageT.convert('RGB')
        imageT = imageT.resize((aConfigration.IMAGE_SIZE, aConfigration.IMAGE_SIZE))  #原始图片 shape(581, 256, 3)
        imageT = np.asarray(imageT)
        imgTNp[t, :, :, :] = imageT
        t += 1

    v = 0
    # # 处理验证集图片:挨个处理大小，转换numpy
    for pic in listPicVal:
        imageV = Image.open(dataRootPath + validPath + commonImgPath + pic)
        # 转换非RGB图片
        if imageV.mode != 'RGB':
            imageV = imageV.convert('RGB')
        imageV = imageV.resize((aConfigration.IMAGE_SIZE, aConfigration.IMAGE_SIZE))
        imageV = np.asarray(imageV)
        imgVNp[v, :, :, :] = imageV
        v += 1

    print('look '+ str(len(imgTNp))
                       + ' '
                       + str(len(imgVNp)))

    return imgTNp, imgVNp, labTNp, labVNp


def readTestPic():
    testFiles = os.listdir(dataRootPath + testPath + commonImgPath)
    if aConfigration.PREVIEW_TEST:
        testFiles = testFiles[:50]
    testImgNp = np.zeros([len(testFiles), aConfigration.IMAGE_SIZE, aConfigration.IMAGE_SIZE, 3])
    k = 0
    for testFile in testFiles:
        testImg = Image.open(dataRootPath + testPath + commonImgPath + testFile)
        testImg = testImg.resize((aConfigration.IMAGE_SIZE, aConfigration.IMAGE_SIZE))
        if testImg.mode != 'RGB':
            testImg = testImg.convert('RGB')
        testnp = np.asarray(testImg)
        testImgNp[k, :, :, :] = testnp
        k += 1
    return testImgNp, testFiles


class myDataSet(Data.Dataset):
    def __init__(self, type):
        imgTNp, imgVNp, labTNp, labVNp = readTrainAndValPic()
        if type == aConfigration.TRAIN:
            self.x = imgTNp
            self.y = labTNp
        elif type == aConfigration.EVAL:
            self.x = imgVNp
            self.y = labVNp

    def __getitem__(self, item):
        return torch.from_numpy(self.x[item]),  self.y[item]

    def __len__(self):
        return len(self.x)






if __name__ == '__main__':
    main()




