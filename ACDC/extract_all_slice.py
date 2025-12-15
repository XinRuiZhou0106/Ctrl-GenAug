import SimpleITK as sitk
import numpy as np
import cv2, os, json
import glob, cc3d
from tqdm import tqdm

def read_cfg(file_path):
    config = {}
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():
                key, value = line.split(':')
                key = key.strip()
                value = value.strip()
                if value.replace('.', '', 1).isdigit():
                    if '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                config[key] = value
    return config

def read_nii(img_path):
    # we use the mask only to remove slices without (or with very few) key anatomical structures, as such slices are not informative for clinical diagnosis
    gt_path = img_path.replace('.nii.gz', '_gt.nii.gz')
    gt_sitk = sitk.ReadImage(gt_path)
    gt_data = np.uint8(sitk.GetArrayFromImage(gt_sitk))
    # exclude the objects with less than 1000 pixels in 3D (refer to MedSAM)
    gt_data = cc3d.dust(
        gt_data, threshold=1000, connectivity=26, in_place=True
    )
    # remove small objects with less than 100 pixels in 2D slices (refer to MedSAM)
    for slice_i in range(gt_data.shape[0]):
        gt_i = gt_data[slice_i, :, :]
        gt_data[slice_i, :, :] = cc3d.dust(
            gt_i, threshold=100, connectivity=8, in_place=True
        )
    # find non-zero slices
    z_index, _, _ = np.where(gt_data > 0)
    z_index = np.unique(z_index)
    
    # load image and preprocess
    img_sitk = sitk.ReadImage(img_path)
    image_data = sitk.GetArrayFromImage(img_sitk)
    # nii preprocess start
    lower_bound, upper_bound = np.percentile(
        image_data[image_data > 0], 0.5
    ), np.percentile(image_data[image_data > 0], 99.5)
    image_data_pre = np.clip(image_data, lower_bound, upper_bound)
    image_data_pre = (
        (image_data_pre - np.min(image_data_pre))
        / (np.max(image_data_pre) - np.min(image_data_pre))
        * 255.0
    )
    image_data_pre[image_data == 0] = 0
    image_data_pre = np.uint8(image_data_pre)
    img_roi = image_data_pre[z_index, :, :]
    return img_roi

# resize after zero-padding
def pad_resize(img):
    z,h,w = img.shape[0], img.shape[1], img.shape[2]
    final_img = np.zeros((z, 256, 256), dtype=np.uint8)
    if h>w:
        padding = h-w
        padding_left = int(padding/2)
        padding_right = padding-padding_left
        # per-slice operation
        for i in range(img.shape[0]):
            padded_slice = cv2.copyMakeBorder(img[i],0,0,padding_left,padding_right,cv2.BORDER_CONSTANT,0)
            assert padded_slice.shape[0] == padded_slice.shape[1]
            final_img[i] = cv2.resize(padded_slice, (256, 256))
    elif h<w:
        padding = w-h
        padding_top = int(padding/2)
        padding_down = padding-padding_top
        # per-slice operation
        for i in range(img.shape[0]):
            padded_slice = cv2.copyMakeBorder(img[i],padding_top,padding_down,0,0,cv2.BORDER_CONSTANT,0)
            assert padded_slice.shape[0] == padded_slice.shape[1]
            final_img[i] = cv2.resize(padded_slice, (256, 256))
    else: # h=w
        for i in range(img.shape[0]):
            final_img[i] = cv2.resize(img[i], (256, 256))
    return final_img

if __name__ == "__main__":
    # load patient info
    total_stat = {"training": {}, "testing": {}}
    for conf_p in glob.glob("database/*/*/*.cfg"):
        mode, patient = conf_p.split('/')[-3], conf_p.split('/')[-2]
        config = read_cfg(conf_p)
        total_stat[mode][patient] = config
        
    # At the patient level, count the number of cases in each disease category
    train_info, test_info = total_stat['training'], total_stat['testing']
    train_cls = [i['Group'] for i in list(train_info.values())]
    test_cls = [i['Group'] for i in list(test_info.values())]
    level, count = np.unique(train_cls, return_counts=True)
    print("train cls stat:")
    for j in range(len(level)):
        print(f"{level[j]}: {count[j]}")
    print("**************")
    level, count = np.unique(test_cls, return_counts=True)
    print("test cls stat:")
    for j in range(len(level)):
        print(f"{level[j]}: {count[j]}")

    # nii -> png (save all extracted slices)
    data_im = "ACDC/ACDC_all_slices"
    os.makedirs(data_im, exist_ok=True)
    meta_file_train, meta_file_test = [], []
    
    for mode, mode_info in total_stat.items():
        with tqdm(total=len(mode_info), desc=f"{mode}:") as pbar:
            for pa, pa_info in mode_info.items():
                # ED frame
                ed_path = f"database/{mode}/{pa}/{pa}_frame{str(pa_info['ED']).zfill(2)}.nii.gz"
                assert os.path.exists(ed_path)
                ed_img = read_nii(ed_path)
                
                # Pad & resize ED frame
                ed_img_final = pad_resize(ed_img)
                
                # Save the metadata of all slices, including file names and labels
                for i in range(ed_img_final.shape[0]):
                    cv2.imwrite(f"{data_im}/{pa}_frame{str(pa_info['ED']).zfill(2)}_slice{str(i+1).zfill(2)}.png", ed_img_final[i])
                    
                    if mode == "training":
                        meta_file_train.append({"file_name": f"{pa}_frame{str(pa_info['ED']).zfill(2)}_slice{str(i+1).zfill(2)}.png",
                                                "cls": pa_info['Group']
                                            })
                    elif mode == "testing":
                        meta_file_test.append({"file_name": f"{pa}_frame{str(pa_info['ED']).zfill(2)}_slice{str(i+1).zfill(2)}.png",
                                                "cls": pa_info['Group']
                                            })
                    else:
                        raise ValueError
                
                # ES frame
                es_path = f"database/{mode}/{pa}/{pa}_frame{str(pa_info['ES']).zfill(2)}.nii.gz"
                assert os.path.exists(es_path)
                es_img = read_nii(es_path)
                
                # Pad & resize ES frame
                es_img_final = pad_resize(es_img)
                
                # Save the metadata of all slices, including file names and labels
                for j in range(es_img_final.shape[0]):
                    cv2.imwrite(f"{data_im}/{pa}_frame{str(pa_info['ES']).zfill(2)}_slice{str(j+1).zfill(2)}.png", es_img_final[j])
                    
                    if mode == "training":
                        meta_file_train.append({"file_name": f"{pa}_frame{str(pa_info['ES']).zfill(2)}_slice{str(j+1).zfill(2)}.png",
                                                "cls": pa_info['Group']
                                            })
                    elif mode == "testing":
                        meta_file_test.append({"file_name": f"{pa}_frame{str(pa_info['ES']).zfill(2)}_slice{str(j+1).zfill(2)}.png",
                                                "cls": pa_info['Group']
                                            })
                    else:
                        raise ValueError
                    
                pbar.update(1)
            pbar.close()
    
    print("train img num: ", len(meta_file_train))
    print("test img num: ", len(meta_file_test))

    with open(f"{data_im}/train_metadata.jsonl", 'w') as f_new:
        for item in meta_file_train:
            json.dump(item, f_new)
            f_new.write('\n')
    f_new.close()

    with open(f"{data_im}/test_metadata.jsonl", 'w') as f:
        for item in meta_file_test:
            json.dump(item, f)
            f.write('\n')
    f.close()