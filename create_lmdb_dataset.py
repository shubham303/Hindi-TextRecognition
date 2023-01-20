""" a modified version of CRNN torch repository https://github.com/bgshih/crnn/blob/master/tool/create_dataset.py """

import os

import cv2
import fire
import lmdb
import numpy as np


def isfontValid(fontList, font):
    # font is not valid if it is not in fontlist.
    for f in fontList:
        if font in f:
            return True
    return False


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def createDataset(inputPath, gtFile, outputPath, checkValid=False, max_iters=0):
    """
    Create LMDB dataset for training and evaluation.
    ARGS:
        inputPath  : input folder path where starts imagePath
        outputPath : LMDB output path
        gtFile     : list of image path and label
        checkValid : if true, check the validity of every image
    """
    os.makedirs(outputPath, exist_ok=True)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    paths = set()
    with open(gtFile, 'r', encoding='utf-8') as data:
        datalist = data.read().split("\n")[:-1]

    x = 0
    nSamples = len(datalist)
    max_iters = max_iters if max_iters > 0 else nSamples

    for i in range(nSamples):
        try:
            if i > max_iters:
                break
        
            imagePath, label = datalist[i].split('\t')
            paths.add(imagePath)
            imagePath = os.path.join(inputPath, imagePath)

            # # only use alphanumeric data
            # if re.search('[^a-zA-Z0-9]', label):
            #     continue

            if not os.path.exists(imagePath):
                print('%s does not exist' % imagePath)
                x += 1
                continue

            with open(imagePath, 'rb') as f:
                imageBin = f.read()

            if imageBin is None:
                continue

            if len(label) > 30:
                print('word length greater than 30')
                continue
            if checkValid:
                try:
                    if not checkImageIsValid(imageBin):
                        print('%s is not a valid image' % imagePath)
                        continue


                except:
                    print('error occured', i)
                    with open(outputPath + '/error_image_log.txt', 'a') as log:
                        log.write('%s-th image data occured error\n' % str(i))
                    continue

            imageKey = 'image-%09d'.encode() % cnt
            labelKey = 'label-%09d'.encode() % cnt
            cache[imageKey] = imageBin
            cache[labelKey] = label.encode()

            if cnt % 1000 == 0:
                writeCache(env, cache)
                cache = {}
                print('Written %d / %d' % (cnt, nSamples))
            cnt += 1
        except Exception:
            print("error occured")


    nSamples = cnt - 1
    cache['num-samples'.encode()] = str(nSamples).encode()
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)

    print(len(paths))
    print(x)

if __name__ == '__main__':
    fire.Fire(createDataset)
