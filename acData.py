from PIL import Image
import os
import numpy as np
import torch
import torch.utils.data as Data
from torchvision import transforms
import json
import aConfigration
from tqdm import tqdm


dataRootPath = '../datas'
trainPath = '/AgriculturalDisease_trainingset'
validPath = '/AgriculturalDisease_validationset'
testPath = '/AgriculturalDisease_testA'
commonImgPath = '/images/'
trainLabel = '/AgriculturalDisease_train_annotations.json'
validLabel = '/AgriculturalDisease_validation_annotations.json'

dataSavedPath = "/data_np_saved"
dataSavedImgTNp = "/imgTNp.npy"
dataSavedImgVNp = "/imgVNp.npy"
dataSavedLabTNp = "/labTNp.npy"
dataSavedLabVNp = "/labVNp.npy"
dataSavedTestNp = "/testNp.npy"
dataSavedTestName = "/testImgName.npy"

def main():
    readTrainAndValPic()
    readTestPic()

# 没用到。
# 由于transforms必须放到dataset里，才能保证每次epoch都能调用。
# 但本代码里的dataset操作的是numpy，无法做增强。
# 因为需要提前将图片转为numpy，存入本地，才不用每次都读取图片。

trans = transforms.Compose([
    transforms.RandomResizedCrop(aConfigration.IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)

])

transCopyV = transforms.Compose([
    transforms.Resize((aConfigration.IMAGE_SIZE_COPY, aConfigration.IMAGE_SIZE_COPY)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

transCopyT = transforms.Compose([
    transforms.Resize((aConfigration.IMAGE_SIZE_COPY, aConfigration.IMAGE_SIZE_COPY)),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomAffine(45),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])



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
    # print("see "+ str(tLabDict[:20]))
    # print("see see "+ str(type(tLabDict)))
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
        tLabDict = tLabDict[:aConfigration.PREVIEW_TRAIN_NUM]
        vLabDict = vLabDict[:aConfigration.PREVIEW_TRAIN_NUM]


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
    # labTNp = np.zeros(len(tLabDict))
    # 为了扩大数据，将验证集加入训练集
    labTNp = np.zeros(len(tLabDict)+len(vLabDict))
    for labTItem in tLabDict:
        lab = labTItem['disease_class']
        labTNp[i] = lab
        i += 1
    labVNp = np.zeros(len(vLabDict))
    for labVItem in vLabDict:
        lab2 = labVItem['disease_class']
        labVNp[j] = lab2
        j += 1
        # 为了扩大数据，将验证集加入训练集
        labTNp[i] = lab2
        i += 1


    '''制作数据集'''
    # imgTNp = np.zeros([len(listPicTrain), aConfigration.IMAGE_SIZE, aConfigration.IMAGE_SIZE, 3], dtype=np.uint8)
    # 为了扩大数据，将验证集加入训练集
    imgTNp = np.zeros([len(listPicTrain + listPicVal), aConfigration.IMAGE_SIZE, aConfigration.IMAGE_SIZE, 3], dtype=np.uint8)
    imgVNp = np.zeros([len(listPicVal), aConfigration.IMAGE_SIZE, aConfigration.IMAGE_SIZE, 3], dtype=np.uint8)


    t = 0
    # 处理训练集图片:挨个处理大小，转换numpy
    for pic in tqdm(listPicTrain):
        imageT = Image.open(dataRootPath + trainPath + commonImgPath + pic)
        # 转换非RGB图片
        if imageT.mode != 'RGB':
            imageT = imageT.convert('RGB')



        # 为添加数据增强1，Image转numpy移除（为了将Image传到DataSet中）
        # imageT = imageT.resize((aConfigration.IMAGE_SIZE, aConfigration.IMAGE_SIZE))  #原始图片 shape(581, 256, 3)
        # imageT = trans(imageT)  # data augmentation 数据增强  ##放在这里错，只执行一次，并没有增强数据。

        # imageT = np.asarray(imageT)
        # imgTNp[t, :, :, :] = imageT
        # t += 1

    v = 0
    # # 处理验证集图片:挨个处理大小，转换numpy
    for pic in tqdm(listPicVal):
        imageV = Image.open(dataRootPath + validPath + commonImgPath + pic)
        # 转换非RGB图片
        if imageV.mode != 'RGB':
            imageV = imageV.convert('RGB')


        # 为添加数据增强2，Image转numpy移除（为了将Image传到DataSet中）
        # imageV = imageV.resize((aConfigration.IMAGE_SIZE, aConfigration.IMAGE_SIZE))
        # imageV = np.asarray(imageV)
        # imgVNp[v, :, :, :] = imageV
        # v += 1
        #
        # # 为了扩大数据，将验证集加入训练集
        # imgTNp[t, :, :, :] = imageV
        # t += 1


    #
    # print('look '+ str(len(imgTNp))
    #                    + ' '
    #                    + str(len(imgVNp)))

    # if not os.path.exists(dataRootPath + dataSavedPath):
    #     os.mkdir(dataRootPath + dataSavedPath)

    # 为添加数据增强3，舍弃图片一次加载储存功能。
    # np.save(dataRootPath + dataSavedPath + dataSavedImgTNp, imgTNp)
    # np.save(dataRootPath + dataSavedPath + dataSavedImgVNp, imgVNp)
    # np.save(dataRootPath + dataSavedPath + dataSavedLabTNp, labTNp)
    # np.save(dataRootPath + dataSavedPath + dataSavedLabVNp, labVNp)
    # return imgTNp, imgVNp, labTNp, labVNp
    return imageT, imageV, labTNp, labVNp


def readTestPic():
    testFiles = os.listdir(dataRootPath + testPath + commonImgPath)
    # testFiles = os.listdir(dataRootPath + validPath + commonImgPath)    #temper for output eval predion json
    if aConfigration.PREVIEW_TEST:
        testFiles = testFiles[:aConfigration.PREVIEW_TEST_NUM]
    testImgNp = np.zeros([len(testFiles), aConfigration.IMAGE_SIZE, aConfigration.IMAGE_SIZE, 3])
    k = 0
    for testFile in tqdm(testFiles):
        testImg = Image.open(dataRootPath + testPath + commonImgPath + testFile)
        # testImg = Image.open(dataRootPath + validPath + commonImgPath + testFile)    #temper for output eval predion json
        testImg = testImg.resize((aConfigration.IMAGE_SIZE, aConfigration.IMAGE_SIZE))
        if testImg.mode != 'RGB':
            testImg = testImg.convert('RGB')
        testnp = np.asarray(testImg)
        testImgNp[k, :, :, :] = testnp
        k += 1

    if not os.path.exists(dataRootPath + dataSavedPath):
        os.mkdir(dataRootPath + dataSavedPath)
    np.save(dataRootPath + dataSavedPath + dataSavedTestNp, testImgNp)
    np.save(dataRootPath + dataSavedPath + dataSavedTestName, testFiles)
    return testImgNp, testFiles




class myDataSet(Data.Dataset):
    def __init__(self, type):

        # 为添加数据增强4，舍弃图片一次加载储存功能。


        imgTNp, imgVNp, labTNp, labVNp = readTrainAndValPic()
        # if aConfigration.NEED_RESTART_READ_TRAIN_DATA:
        #     imgTNp, imgVNp, labTNp, labVNp = readTrainAndValPic()


        # else:
        #     imgTNp = np.load(dataRootPath + dataSavedPath + dataSavedImgTNp)
        #     imgVNp = np.load(dataRootPath + dataSavedPath + dataSavedImgVNp)
        #     labTNp = np.load(dataRootPath + dataSavedPath + dataSavedLabTNp)
        #     labVNp = np.load(dataRootPath + dataSavedPath + dataSavedLabVNp)

        if type == aConfigration.TRAIN:
            self.x = imgTNp
            self.y = labTNp
        elif type == aConfigration.EVAL:
            self.x = imgVNp
            self.y = labVNp

    def __getitem__(self, item):
        # return torch.from_numpy(self.x[item]),  self.y[item]
        if type == aConfigration.TRAIN:
            imageT = self.x
            imageT = transCopyT(imageT)
            labelT = self.y
            return imageT, labelT
        elif type == aConfigration.EVAL:
            imageV = self.x
            imageV - transCopyV(imageV)
            labelV = self.y
            return imageV, labelV



    def __len__(self):
        return len(self.x)

class myTestSet(Data.Dataset):
    def __init__(self):
        if aConfigration.NEED_RESTART_READ_TEST_DATA:
            testNp, testImgName = readTestPic()
        else:
            testNp = np.load(dataRootPath + dataSavedPath + dataSavedTestNp)
            testImgName = np.load(dataRootPath + dataSavedPath + dataSavedTestName)
            testImgName = list(testImgName)
        self.x = testNp
        self.y = testImgName

    def __getitem__(self, item):
        return torch.from_numpy(self.x[item]),  self.y[item]

    def __len__(self):
        return len(self.x)






if __name__ == '__main__':
    main()





