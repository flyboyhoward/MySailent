import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob
import cv2

from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET # full size version 173.6 MB
from model import U2NETP # small version u2net 4.7 MB

def matting_error_visualize(model_name):
    matting_dir = os.path.join(os.getcwd(), 'test_data', 'matting' + os.sep)
    mask_dir =os.path.join(os.getcwd(), 'test_data', 'mask' + os.sep)
    error_dir = os.path.join(os.getcwd(), 'test_data', 'error' + os.sep)

    img_name_list = glob.glob(mask_dir + os.sep + '*')
    
    for i_img, img_path in enumerate(img_name_list):
        maskImage = cv2.imread(img_path)
        maskImage = cv2.cvtColor(maskImage, cv2.COLOR_RGB2GRAY)
        img_name = img_name_list[i_img].split(os.sep)[-1].split('.')[0]
        mattingImage = cv2.imread(os.path.join(matting_dir + img_name + '.png'), cv2.IMREAD_UNCHANGED)#
        error = abs(maskImage - mattingImage[:,:,3])      
        mattingImage[:,:,3] += error
        mattingImage[:,:,2] += error 
        print('visualizing: ', img_name, '.png')
        cv2.imwrite(os.path.join(error_dir, img_name + '.png'), mattingImage)


def matting(model_name):

    image_dir = os.path.join(os.getcwd(), 'test_data', 'test_images')
    prediction_dir = os.path.join(os.getcwd(), 'test_data', model_name + '_results' + os.sep)
    save_dir = os.path.join(os.getcwd(), 'test_data', 'matting' + os.sep)
    # save_dir_crop =os.path.join(os.getcwd(), 'test_data', 'crop' + os.sep)
    img_name_list = glob.glob(image_dir + os.sep + '*')
    # print(img_name_list)

    for i_img, img_path in enumerate(img_name_list):
        objectImage = cv2.imread(img_path)
        img_name = img_name_list[i_img].split(os.sep)[-1].split('.')[0]
        salientImage = cv2.imread(os.path.join(prediction_dir + img_name + '.png'))#
        salientImageGray = cv2.cvtColor(salientImage, cv2.COLOR_RGB2GRAY)
        _, threshImage = cv2.threshold(salientImageGray, 180, 255, cv2.THRESH_BINARY)
        # pos_list = np.where(threshImage>0)
        # cropImage = objectImage[min(pos_list[0]):max(pos_list[0]) + 1, min(pos_list[1]):max(pos_list[1]) + 1]
        # threshImage = cv2.GaussianBlur(salientImageGray, (11,11), 0) 

        # cv2.drawContours(objectImage, contours, -1, (0,255,0), 3)
        binarythreshImage = threshImage/255.
        objectImage = objectImage*np.transpose(np.squeeze([[binarythreshImage],[binarythreshImage],[binarythreshImage]]),(1,2,0))
        # objectImage = objectImage[min(pos_list[0]):max(pos_list[0]) + 1, min(pos_list[1]):max(pos_list[1]) + 1]
        # threshImage = threshImage[min(pos_list[0]):max(pos_list[0]) + 1, min(pos_list[1]):max(pos_list[1]) + 1]
        # rgb to rgba for transparent background in image. NOTE: the image must in .png form
        threshImage = threshImage[:,:,np.newaxis]
        objectImage = np.append(objectImage, threshImage, axis = 2)
        print('matting: ',img_name_list[i_img].split(os.sep)[-1])
        cv2.imwrite(os.path.join(save_dir, img_name + '.png'),objectImage.astype(int))
        # cv2.imwrite(os.path.join(save_dir_crop, img_name + '.jpg'), cropImage)
        
# normalize the predicted SOD probability map
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

def main():

    # --------- 1. get image path and name ---------
    model_name='u2net'#u2netp



    image_dir = os.path.join(os.getcwd(), 'test_data', 'test_images')
    prediction_dir = os.path.join(os.getcwd(), 'test_data', model_name + '_results' + os.sep)
    model_dir = os.path.join(os.getcwd(), 'saved_models', model_name, model_name + 'latestv2.pth')

    img_name_list = glob.glob(image_dir + os.sep + '*')
    # print(img_name_list)

    # --------- 2. dataloader ---------
    #1. dataloader
    test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
                                        lbl_name_list = [],
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=0)

    # --------- 3. model define ---------
    if(model_name=='u2net'):
        print("...load U2NET---173.6 MB")
        net = U2NET(3,1)
    elif(model_name=='u2netp'):
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3,1)

    if torch.cuda.is_available():
        net.cuda()

        net.load_state_dict(torch.load(model_dir))
        #net = nn.DataParallel(net)

    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.eval()

    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):

        print("inferencing:",img_name_list[i_test].split(os.sep)[-1])

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

if __name__ == "__main__":
    # main()
    # print('inference complete start matting')
    # matting('u2net')
    # print('matting complete starting error visualize')
    matting_error_visualize('u2net')
    print('error visualize complete')