import numpy as np
import cv2, os, json
import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":
    # load labels
    train_label_csv, test_label_csv = pd.read_csv("MRNet/train-acl.csv"), pd.read_csv("MRNet/valid-acl.csv")
    train_label_dict = dict(zip(train_label_csv["PatientID"], train_label_csv["label"]))
    train_label_dict = {str(k).zfill(4): v for k, v in train_label_dict.items()}
    test_label_dict = dict(zip(test_label_csv["PatientID"], test_label_csv["label"]))
    test_label_dict = {str(k).zfill(4): v for k, v in test_label_dict.items()}

    # load patient info
    train_data, train_label, test_data, test_label = [], [], [], []
    for m in ["train", "valid"]:
        data = os.listdir(f"MRNet/{m}/sagittal")
        for d in data:
            if m == "train":
                train_data.append(d)
                train_label.append(train_label_dict[d.split(".npy")[0]])
            else:
                test_data.append(d)
                test_label.append(test_label_dict[d.split(".npy")[0]])

    total_stat = {"training": (train_data, train_label), "testing": (test_data, test_label)}

    # At the patient level, count the number of cases in each disease category
    level, count = np.unique(train_label, return_counts=True)
    print("train cls stat:")
    for j in range(len(level)):
        print(f"{level[j]}: {count[j]}")
    print("**************")
    level, count = np.unique(test_label, return_counts=True)
    print("test cls stat:")
    for j in range(len(level)):
        print(f"{level[j]}: {count[j]}")

    # nii -> png (save all extracted slices)
    data_im = "MRNet/MRNet_all_slices"
    os.makedirs(data_im, exist_ok=True)
    meta_file_train, meta_file_test = [], []
    
    slice_count = []
    for mode, mode_info in total_stat.items():
        with tqdm(total=len(mode_info[0]), desc=f"{mode}:") as pbar:
            for data_npy_p, label in zip(*mode_info):
                if mode == "training":
                    data_npy_p = "MRNet/train/sagittal/" + data_npy_p
                else:
                    data_npy_p = "MRNet/valid/sagittal/" + data_npy_p
                assert os.path.exists(data_npy_p)
                volume = np.load(data_npy_p, allow_pickle=True)[()] # slice, h, w
                assert volume.shape[1:] == (256, 256)

                slice_count.append(volume.shape[0])

                # Save the metadata of all slices, including file names and labels
                volume_name = data_npy_p.split("/")[-1].split(".")[0]
                for i in range(volume.shape[0]):
                    cv2.imwrite(f"{data_im}/{volume_name}_slice{str(i+1).zfill(2)}.png", volume[i, :, :]) 
                    
                    if mode == "training":
                        meta_file_train.append({"file_name": f"{volume_name}_slice{str(i+1).zfill(2)}.png",
                                                "cls": label
                                            })
                    elif mode == "testing":
                        meta_file_test.append({"file_name": f"{volume_name}_slice{str(i+1).zfill(2)}.png",
                                                "cls": label
                                            })
                    else:
                        raise ValueError
                
                pbar.update(1)
            pbar.close()
    
    print("train img num: ", len(meta_file_train))
    print("test img num: ", len(meta_file_test))
    print("avg slice num: ", np.mean(slice_count))
    print("min slice num: ", np.min(slice_count))
    print("max slice num: ", np.max(slice_count))

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
