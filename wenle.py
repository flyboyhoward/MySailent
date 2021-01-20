import cv2
import os
import glob
import argparse
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim
import time
import numpy as np
from PIL import Image
import glob

from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET # full size version 173.6 MB
from model import U2NETP # small version u2net 4.7 MB

def getImage():
    cap = cv2.VideoCapture(0)
    ret = cap.set(3,1920)
    ret = cap.set(4,1080)

    i = 0
    while(True):
        # Capture frame-by-frame
        _, frame = cap.read()
        frame = frame[230:1050, 700:1220]
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
        	print('save an image')
        	print(os.path.join('temp/' + str(i) + '.jpg'))
        	cv2.imwrite(os.path.join('temp/' + str(i) + '.jpg'), frame)
        	i += 1
        
        elif cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def calCameraPosition():
    cap = cv2.VideoCapture(0)
    ret = cap.set(3,1920)
    ret = cap.set(4,1080)

    i = 0
    while(True):
        # Capture frame-by-frame
        _, frame = cap.read()
        for i in range(frame.shape[1]):
            frame[540,i] = [0,0,255]
        for i in range(frame.shape[0]):
            frame[i,960] = [0,0,255]
        frame = frame[230:1050, 700:1220]
        cv2.imshow('frame',frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def matting(model_name):

    image_dir = os.path.join(os.getcwd(), 'test_data', 'test_images')
    prediction_dir = os.path.join(os.getcwd(), 'test_data', model_name + '_results' + os.sep)
    save_dir = os.path.join(os.getcwd(), 'test_data', 'matting' + os.sep)

    img_name_list = glob.glob(image_dir + os.sep + '*')
    # print(img_name_list)

    for i_img, img_path in enumerate(img_name_list):
        objectImage = cv2.imread(img_path)
        img_name = img_name_list[i_img].split(os.sep)[-1].split('.')[0]
        salientImage = cv2.imread(os.path.join(prediction_dir + img_name + '.png'))#
        salientImageGray = cv2.cvtColor(salientImage, cv2.COLOR_RGB2GRAY)
        _, threshImage = cv2.threshold(salientImageGray, 200, 255, cv2.THRESH_BINARY)

        # cv2.drawContours(objectImage, contours, -1, (0,255,0), 3)
        binarythreshImage = threshImage//255
        objectImage = objectImage*np.transpose(np.squeeze([[binarythreshImage],[binarythreshImage],[binarythreshImage]]),(1,2,0))

        # rgb to rgba for transparent background in image. NOTE: the image must in .png form
        threshImage = threshImage[:,:,np.newaxis]
        objectImage = np.append(objectImage, threshImage, axis = 2)

        cv2.imwrite(os.path.join(save_dir, img_name + '.png'),objectImage)

def meassure(threshImage, camera_height, camera_center):
    contours, _ = cv2.findContours(threshImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = np.squeeze(np.asarray(contours[0]))
    if len(contours.shape) > 1:
        y1 = min(contours[:,1])  # object top position
        y2 = max(contours[:,1])  # object bottom position
        if y1 < camera_center:
            objectHeight = camera_height*(1 + abs(y1 - camera_center)/abs(y2 - camera_center))
        elif y1 > camera_center:
            objectHeight = camera_height*(1 - abs(y1 - camera_center)/abs(y2 - camera_center))
        else:
            objectHeight = camera_height
        x1 = min(contours[:, 0])
        x2 = max(contours[:, 0])
        objectWidth = objectHeight*((x2-x1)/(y2-y1))
    else:
        objectHeight = 0; objectWidth = 0
        print('please put an object in the box or try to put the object again')

    return objectHeight, objectWidth

def calCameraHeight(threshImage, camera_center):
    contours, _ = cv2.findContours(threshImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = np.squeeze(np.asarray(contours[0]))
    # print(min(contours[:,0]), max(contours[:,0]))
    y1 = min(contours[:,1])  # object top position
    y2 = max(contours[:,1])  # object bottom position
    height = float(input('please input height(mm):'))

    cameraHeight = height*(abs(y2 - camera_center)/abs(y1 - y2))

    return cameraHeight

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def save_output(image_name,pred,d_dir):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

    pb_np = np.array(imo)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]

    imo.save(d_dir+imidx+'.png')

def salImg(model_name):
    print('Calibrate camera position, please set it perpendicular to object plane.')
    print('.........................PRESS Q to CONTINUE...........................')
    calCameraPosition()
    cal_flag = True
    # --------- 1. get image path and name ---------

    image_dir = os.path.join(os.getcwd(), 'test_data', 'test_images')
    prediction_dir = os.path.join(os.getcwd(), 'test_data', model_name + '_results' + os.sep)
    model_dir = os.path.join(os.getcwd(), 'saved_models', model_name, model_name + '.pth')
    save_dir = os.path.join(os.getcwd(), 'test_data', 'matting' + os.sep)

    img_name_list = glob.glob(image_dir + os.sep + '*')
    # print(img_name_list)

    # --------- 3. model define ---------
    if(model_name=='u2net'):
        print("...load U2NET---173.6 MB")
        net = U2NET(3,1)
    elif(model_name=='u2netp'):
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3,1)
    net.load_state_dict(torch.load(model_dir))
    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    cap = cv2.VideoCapture(0)
    ret = cap.set(3,1920)
    ret = cap.set(4,1080)

    camera_center = [260, 310]
    cameraHeight = 0

    while(True):
        # Capture frame-by-frame
        t1 = time.time()
        _, frame = cap.read()
        frame = frame[230:1050, 700:1220]

        cv2.imwrite(os.path.join('test_data/test_images/temp.jpg'), frame)

        test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
                                            lbl_name_list = [],
                                            transform=transforms.Compose([RescaleT(320),
                                                                          ToTensorLab(flag=0)])
                                            )
        test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                            batch_size=1,
                                            shuffle=False,
                                            num_workers=1)

    # --------- 4. inference for each image ---------
        for i_test, data_test in enumerate(test_salobj_dataloader):

            inputs_test = data_test['image']
            inputs_test = inputs_test.type(torch.FloatTensor)

            if torch.cuda.is_available():
                inputs_test = Variable(inputs_test.cuda())
            else:
                inputs_test = Variable(inputs_test)

            d1,d2,d3,d4,d5,d6,d7= net(inputs_test)

            # normalization
            pred = d1[:,0,:,:]
            pred = normPRED(pred)

            # save results to test_results folder
            if not os.path.exists(prediction_dir):
                os.makedirs(prediction_dir, exist_ok=True)
            save_output(img_name_list[i_test],pred,prediction_dir)

            del d1,d2,d3,d4,d5,d6,d7

        objectImage = cv2.imread(img_name_list[0])
        img_name = img_name_list[0].split(os.sep)[-1].split('.')[0]
        salientImage = cv2.imread(os.path.join(prediction_dir + img_name + '.png'))#
        salientImageGray = cv2.cvtColor(salientImage, cv2.COLOR_RGB2GRAY)
        _, threshImage = cv2.threshold(salientImageGray, 200, 255, cv2.THRESH_BINARY)
        
        if cal_flag:
            cameraHeight = calCameraHeight(threshImage, camera_center[1])
            print('*********************************************************************')
            print('Camera height confirmed, please enjoy this fantastic code! HAVE FUN:)')
            print('*********************************************************************')
            # print(cameraHeight)
            cal_flag = False
            continue
        else:
            objectHeight, objectWidth = meassure(threshImage, cameraHeight, camera_center[1])

        threshImage //= 255
        objectImage = objectImage*np.transpose(np.squeeze([[threshImage],[threshImage],[threshImage]]),(1,2,0))
        # print(os.path.join(save_dir, img_name + '.jpg'))
        cv2.imwrite(os.path.join(save_dir, img_name + '.jpg'),objectImage)
        cv2.imshow('salimg', objectImage)

        t2 = time.time()

        print('object height:', '%.2f' % objectHeight, 
              'mm | width:', '%.2f' % objectWidth, 
              "mm | inferencing time:", '%.2f' % (t2-t1), 's' , 
              '| image shape:', objectImage.shape[0:2])

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='wenle')
    parser.add_argument('mode', type=str)
    args = parser.parse_args()
    
    model_name='u2net'#u2netp or u2net

    if args.mode == 'cap':
        getImage()
    if args.mode == 'mat':
        matting(model_name)

    if args.mode == 'niubi':
        salImg(model_name)
    else:
        print('unknown input argument please try again')