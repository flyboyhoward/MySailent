import os
import numpy as np
import random
import cv2
import glob
import shutil
import imutils
from tqdm import tqdm

def composite_foreground2background(foreground, background, foreground_scale=0.7):
    '''
    Composite foreground to background follow the equation below:
        I = alpha*Foreground + (1-alpha)*background
    Tricks: 
        Foreground will be put on the central area of background randomly
    Param: 
        foreground: foreground 4 channel image (RGBA) 
        background: background 3 channel image
        foreground_scale: scale of the shorter side of background image, which foreground size should smaller than it
    Return: 
        composite_image: 3 channel image with foreground, dtype = np.uint8
        composite_mask: 1 channel mask, dtype = np.uint8
    '''
    foreground_height, foreground_width, _ = foreground.shape
    background_height, background_width, _ = background.shape

    edge_thresh = 0.1   # threshold for placing foreground in the central area of background image

    # resize foreground to proper size which foreground size should smaller than background size
    if foreground_width >= foreground_scale*background_width and foreground_height < foreground_scale*background_height:
        resize_scale = (foreground_scale*background_width)/foreground_width
    elif foreground_width < foreground_scale*background_width and foreground_height >= foreground_scale*background_height:
        resize_scale = (foreground_scale*background_height)/foreground_height
    elif foreground_width >= foreground_scale*background_width and foreground_height >= foreground_scale*background_height:
        resize_scale = min((foreground_scale*background_width)/foreground_width,(foreground_scale*background_height)/foreground_height)
    else: 
        resize_scale = 1
    if resize_scale != 1:
        foreground = cv2.resize(foreground, (int(resize_scale*foreground_width), int(resize_scale*foreground_height)), interpolation = cv2.INTER_AREA)
    
    foreground_height, foreground_width, _ = foreground.shape
    foreground_new = np.zeros((background_height, background_width,4))
    # generate random composite position
    position = [random.randint(int(edge_thresh*background_height), background_height - foreground_height - int(edge_thresh*background_height)), 
                random.randint(int(edge_thresh*background_width), background_width - foreground_width - int(edge_thresh*background_width))] 
    
    foreground_mask = foreground[:,:,3]/255.    # Normalize
    foreground_mask4compsite = np.array([foreground[:,:,0]*foreground_mask, 
                                        foreground[:,:,1]*foreground_mask,
                                        foreground[:,:,2]*foreground_mask,
                                        foreground[:,:,3]]).transpose((1,2,0))
    foreground_new[position[0]:position[0] + foreground_height, position[1]:position[1] + foreground_width] = foreground_mask4compsite
    
    background_mask = foreground_new[:,:,3].astype(np.uint8)
    background_mask4composite = 1 - cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)/255 # 1 channel to 3 channels
    background_new = background * background_mask4composite
    # composite foreground with background
    composite_image = background_new + foreground_new[:,:,:3]
    composite_mask = background_mask
    
    return composite_image, composite_mask

def prune_foreground(foreground):
    '''
    Prune foreground by removing regions where alpha == 0 
    Param:
        foreground: 4 channel png (RGBA)
    Return:
        foreground_new: prune foreground, dtype = uint8
    '''
    foreground_mask = foreground[:,:,3]
    pos_list = np.where(foreground_mask>0)
    # crop image where Alpha > 0
    foreground_new = foreground[min(pos_list[0]):max(pos_list[0]) + 1, min(pos_list[1]):max(pos_list[1]) + 1]

    return foreground_new

def refresh_folder(folder_path):
    '''
    Create folder if not exist 
    Clean all files in folder if exist
    Param:
        folder_path: path to folder
    '''
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        shutil.rmtree(folder_path)
        os.makedirs(folder_path)

def load_foreground(file_path):
    '''
    Load and pre-process image from given file path
    Foreground file should be 4 channel image (RGBA)
    Param: 
        file_path: path to the image file
    Return:
        foreground: 4 channel (RGBA) pruned foreground, dtype = uint8
        flag: flag for whether this foreground will be use in later process
    '''
    foreground = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    if foreground.shape[2] == 4:
        foreground = prune_foreground(foreground)   # prune foreground
        flag = True
    else:
        print(file_path.split(os.sep)[-1], ' Invalid foreground. Will skip the composition of this foreground.')
        foreground = None
        flag = False

    return foreground, flag

def load_background(file_path):
    '''
    Load background image
    Background should be 3 channel image
    Param: 
        file_path: path to the image file
    Return:
        background: 3 channel background, dtype = uint8
        flag: flag for whether this background will be use in later process
    '''
    background = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    if background.shape[2] == 3:
        background = background
        flag = True
    else:
        print(file_path.split(os.sep)[-1], ' Invalid background. Will skip this background.')
        background = None
        flag = False

    return background, flag

def generate_random_background(foreground):
    '''
    Generate random background directly from input foreground
    Hopefully, will improve training by adding random generated background
    Return:
        random_background: 3 channel background, dtype = uint8
    '''
    height, width = foreground.shape[:2]
    random_background = np.random.randint(255, size = [int(height*1.3), int(width*1.3),3], dtype = np.uint8)

    return random_background

def generate_white_background(foreground):
    '''
    Generate white background directly from input foreground
    "Brilliant" idea proposed by David Zhang
    Return:
        white_background: 3 channel background, dtype = uint8
    '''
    height, width = foreground.shape[:2]
    white_background = np.ones((int(height*1.3), int(width*1.3), 3), dtype = int)*255

    return white_background

def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

def random_flip_rotate(image):
    '''
    Flip image horizonally with probability of 0.5 & rotate image randomly
    Return:
        image: horizonal flip
    '''
    if random.random() < 0.5:
        image = cv2.flip(image, 1)
    
    if random.random() < 0.5:
        angle = random.randrange(0,360,15)
        image = rotate_bound(image, angle)

    return image

def generate_img_name_list(folder_path):
    '''
    generate .txt with training image ids
    Param:
        folder_path: path to folder
    '''
    img_list = glob.glob(folder_path + '*')

    with open('train.txt','w') as f:
        for img_path in img_list:
            img_name = img_path.split(os.sep)[-1]
            f.write(img_name + '\n')
        f.close()

def main():

    foreground_dir = os.path.join(os.getcwd(), 'train_imagev2_foreground' + os.sep)
    background_dir = os.path.join(os.getcwd(), 'train_imagev2_background' + os.sep)
    save_image_dir = os.path.join(os.getcwd(), 'train_data', 'train_image' + os.sep)
    save_mask_dir = os.path.join(os.getcwd(), 'train_data', 'train_mask' + os.sep)
    
    refresh_folder(save_image_dir); refresh_folder(save_mask_dir)

    foreground_list = glob.glob(foreground_dir + '*')
    background_list = glob.glob(background_dir + '*')
    
    for i_image in tqdm(range(len(foreground_list)), desc = 'Processing:', unit = 'img'):
        i_path = foreground_list[i_image]
        foreground, flag = load_foreground(i_path)
        if flag == False:
            continue
        foreground_name = foreground_list[i_image].split(os.sep)[-1].split('.')[0]
        
        for i in range(3):
            background, flag = load_background(background_list[random.randint(0,len(background_list)-1)])
            if flag == False:
                continue
            composite_image, composite_mask = composite_foreground2background(foreground, background,size_thresh=0.8)
            
            cv2.imwrite(os.path.join(save_image_dir, foreground_name + str(i) + '.jpg'), composite_image)
            cv2.imwrite(os.path.join(save_mask_dir, foreground_name + str(i) + '.png'), composite_mask)
            # print('save ', foreground_name+str(i), ' to folder' ' %d th foreground'%(i_image+1))
    
    print('Complete Generating Dateset \n','!!!Enjoy Coding!!!')

if __name__ == '__main__':

    foreground, flag = load_foreground('1.png')
    # background, flag = load_background('train_data/DUTS/DUTS-TR/HRSOD_train/00000.jpg')
    foreground = random_flip_rotate(foreground)
    background = generate_white_background(foreground)
    # background = random_flip(background)
    composite_image, composite_mask = composite_foreground2background(foreground, background,foreground_scale=0.8)
    cv2.imwrite(os.path.join('matte.jpg'), composite_image)
    # cv2.imwrite(os.path.join(save_mask_dir, foreground_name + str(i) + '.png'), composite_mask)
    # print('Complete Generating Dateset \n','!!Enjoy Coding!!')
    # foreground_dir = os.path.join(os.getcwd(), 'test_data', 'test_images' + os.sep)
    # generate_img_name_list(foreground_dir)
'''
flip whole dataset
'''
# data_dir = os.path.join(os.getcwd(), 'train_data' + os.sep)
# image_dir = os.path.join(data_dir, 'DUTS', 'DUTS-TR', 'HRSOD_train' + os.sep)
# tra_label_dir = os.path.join(data_dir, 'DUTS', 'DUTS-TR', 'HRSOD_train_mask' + os.sep)

# img_name_list = glob.glob(image_dir + '*') 
# mask_name_list = glob.glob(tra_label_dir + '*')

# print("---")
# print("train images: ", len(img_name_list))
# print("train labels: ", len(mask_name_list))
# print("---")

# for i_img, img_path in enumerate(img_name_list):
#     img = cv2.imread(img_path)
#     img_name = img_name_list[i_img].split(os.sep)[-1].split('.')[0]
#     print('flip image:',img_name)
#     flipped_img = cv2.flip(img, 1)
#     flipped_img_name = '1' + img_name
#     cv2.imwrite(os.path.join(image_dir, flipped_img_name + '.jpg'), flipped_img)
#     # print(flipped_img_name)


# for i_img, img_path in enumerate(mask_name_list):
#     img = cv2.imread(img_path)
#     img_name = img_name_list[i_img].split(os.sep)[-1].split('.')[0]
#     print('flip mask: ',img_name)
#     flipped_img = cv2.flip(img, 1)
#     flipped_img_name = '1' + img_name 
#     cv2.imwrite(os.path.join(tra_label_dir, flipped_img_name + '.png'),flipped_img)
#     # print(flipped_img_name)
