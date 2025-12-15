import h5py
import cv2, os
import numpy as np
from tqdm import tqdm

# resize after zero-padding
def pad(img):
    h,w = img.shape[0], img.shape[1]
    if h>w:
        padding = h-w
        padding_left = int(padding/2)
        padding_right = padding-padding_left
        img = cv2.copyMakeBorder(img,0,0,padding_left,padding_right,cv2.BORDER_CONSTANT,0) 
    elif h<w:
        padding = w-h
        padding_top = int(padding/2)
        padding_down = padding-padding_top
        img = cv2.copyMakeBorder(img,padding_top,padding_down,0,0,cv2.BORDER_CONSTANT,0) 
    return img

def crop(ori_img, mask): 
    extend_pixel = 5
    y,x = np.where(mask == 255)
    x1 = np.min(x)
    y1 = np.min(y)
    x2 = np.max(x)
    y2 = np.max(y)

    if (x1-extend_pixel) >= 0:
        x1 = x1-extend_pixel
    else:
        x1 = 0
        raise ValueError("mask out of image")
    if (y1-extend_pixel) >= 0:
        y1 = y1-extend_pixel
    else:
        y1 = 0
        raise ValueError("mask out of image")
    if (x2+extend_pixel) <= ori_img.shape[1]:
        x2 = x2+extend_pixel
    else:
        x2 = ori_img.shape[1]
        raise ValueError("mask out of image")
    if (y2+extend_pixel) <= ori_img.shape[0]:
        y2 = y2+extend_pixel
    else:
        y2 = ori_img.shape[0]
        raise ValueError("mask out of image")
    
    bbox_list = [y1,x1,y2,x2]
    crop_img = ori_img[bbox_list[0]:bbox_list[2],bbox_list[1]:bbox_list[3]]
    crop_img = pad(crop_img)
    
    mask_img = mask[bbox_list[0]:bbox_list[2],bbox_list[1]:bbox_list[3]]
    mask_img = pad(mask_img)

    if not crop_img.shape[0] == mask_img.shape[0] and crop_img.shape[1] == mask_img.shape[1] and mask_img.shape[0] == mask_img.shape[1]:
        raise ValueError
    
    crop_img = cv2.resize(crop_img,(256,256))
    mask_img = cv2.resize(mask_img,(256,256),interpolation=cv2.INTER_NEAREST)
    return crop_img, mask_img

def are_integers_arithmetic_sequence(numbers):
    differences = []
    for i in range(len(numbers) - 1):
        differences.append(numbers[i+1] - numbers[i])
    return len(set(differences)) <= 1


if __name__ == "__main__":
    hdf5_file = h5py.File("dataset.hdf5", 'r')

    # check the info
    print("Groups and datasets in HDF5 file:")
    for key in hdf5_file.keys():
        print(key)

    anno = hdf5_file["annot_id"]
    im = hdf5_file["image"]
    ma = hdf5_file["mask"]
    frame_num = hdf5_file["frame_num"]

    anno_map_frame = {}
    for i in range(len(anno)):
        if int(anno[i].decode('utf-8').rstrip('_')) not in anno_map_frame:
            anno_map_frame[int(anno[i].decode('utf-8').rstrip('_'))] = [int(frame_num[i].decode('utf-8'))]
        else:
            anno_map_frame[int(anno[i].decode('utf-8').rstrip('_'))].append(int(frame_num[i].decode('utf-8')))
            
    # check the continuity of the video frame ids
    noisy_annot = []
    for annot_id, frames in anno_map_frame.items():
        if not are_integers_arithmetic_sequence(frames):
            noisy_annot.append(annot_id)
    print("need to be double-checked", noisy_annot)

    # create folder
    patient_p_im = "TUSC/TUSC_all_images"
    os.makedirs(patient_p_im, exist_ok=True)

    frame_sizes = []
    with tqdm(total=len(anno)) as pbar:
        for i in range(len(anno)):
            # nodule id (video)
            nodule_id = int(anno[i].decode('utf-8').rstrip('_'))
            
            # our work excludes the TR1 video, you can delete the corresponding info (nodule_id = 8) in csv in advance
            if nodule_id == 8:
                continue
            
            # frame id
            frame_id = int(frame_num[i].decode('utf-8'))
            
            img, mask = im[i], ma[i]
            
            if not list(np.unique(mask)) == [0, 255]:
                _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
            assert list(np.unique(mask)) == [0, 255]
            
            out = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)
            # ensure that only single lesion is shown in one video
            if (out[0] - 1) == 2:
                assert (np.unique(out[1], return_counts=True)[1][-1] <= 150)
                mask[out[1] == 2] = 0
                # re-check
                num, _, _, _ = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)
                assert (num - 1) == 1
            else:
                assert (out[0] - 1) == 1

            frame_sizes.append(img.shape)
            
            # cropped each image using a bounding box with a buffer of 5 pixels around the lesion ROI
            crop_img, mask_img = crop(img, mask)
            
            cv2.imwrite(f"{patient_p_im}/Nodule{nodule_id}_{frame_id}.png", crop_img)
            pbar.update(1)
        pbar.close()

    
    
    
    
    