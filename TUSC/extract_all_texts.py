import numpy as np
import os, json
from sklearn.model_selection import train_test_split

ACR_TI_RADS = {
    "composition":{
        1: "Mixed cystic and solid",
        2: "Solid"
    },
    "echogenicity":{
        1: "Isoechoic",
        2: "Hypoechoic",
        3: "Very hypoechoic"
    },
    "shape":{
        0: "Wider-than-tall",
        3: "Taller-than-wide"
    },
    "margin":{
        0: "Smooth",
        2: "Irregular",
        3: "Extra-thyroidal extension"
    },
    "echogenicfoci":{
        0: "None",
        1: "Macrocalcifications",
        2: "Peripheral (rim) calcifications",
        3: "Punctate echogenic foci"
    }
}

metadata_text_dict = {} # score -> attribute

metadata = np.loadtxt("metadata.csv", delimiter=',', dtype=str, encoding='utf-8') # ensure that the TR1 video is removed in csv
groups = list(metadata[0])

for data in metadata[1:]: # each nodule
    data = list(data)
    nodule_id = "Nodule" + data[0].split("_")[0]
    metadata_text_dict[nodule_id] = {}
    for i in range(1, len(groups)):
        if ("ti-rads" in groups[i]) and ("level" not in groups[i]):
            property_descrip = ACR_TI_RADS[groups[i].split("_")[1]][int(data[i])]
            metadata_text_dict[nodule_id][groups[i].split("_")[1]] = property_descrip
        elif "level" in groups[i]:
            metadata_text_dict[nodule_id][groups[i]] = data[i]
        else:
            metadata_text_dict[nodule_id][groups[i]] = data[i]
            
# split the dataset into training and testing sets (7:3)
nodule_list = list(metadata_text_dict.keys())
tirads_list = []
for v in metadata_text_dict.values():
    # merge TI-RADS2 and 3 in our work
    if v["ti-rads_level"] == "2" or v["ti-rads_level"] == "3":
        tirads_list.append("2-3")
    else:
        tirads_list.append(v["ti-rads_level"])

train_nodule_list, test_nodule_list, _, _ = train_test_split(nodule_list, tirads_list, test_size=0.3, stratify=tirads_list, random_state=3407)

# At the video level, count the number of cases in each disease category
total_stat = {}
total_stat["train_level"], total_stat["test_level"] = {}, {}
train_level = [int(metadata_text_dict[n]['ti-rads_level']) for n in train_nodule_list]
level, count = np.unique(train_level, return_counts=True)
for j in range(len(level)):
    total_stat["train_level"][f"TR{level[j]}"] = count[j]

test_level = [int(metadata_text_dict[n]['ti-rads_level']) for n in test_nodule_list]
level, count = np.unique(test_level, return_counts=True)
for j in range(len(level)):
    total_stat["test_level"][f"TR{level[j]}"] = count[j]
print("TR counts in the train & test sets")
print(total_stat)

# Save the metadata of all images, including file names, texts, and labels
meta_file_train, meta_file_test = [], []
data_p = "TUSC/TUSC_all_images_consecutive"
for im_name in os.listdir(data_p):
    nodule_name = im_name.split("_")[0]
    anno_info = metadata_text_dict[nodule_name]
    
    # text
    # note that some nodules may not include the attribute of 'echogenicfoci'
    if anno_info['echogenicfoci'] == "None":
        text = f"nodule is {anno_info['composition']}, {anno_info['echogenicity']}, {anno_info['shape']}, and {anno_info['margin']}.".lower()
    else:
        text = f"nodule is {anno_info['composition']}, {anno_info['echogenicity']}, {anno_info['shape']}, {anno_info['margin']}, and {anno_info['echogenicfoci']}.".lower()
    text = "The " + text
    
    # merge TR2 and 3
    if anno_info['ti-rads_level'] == '2' or anno_info['ti-rads_level'] == '3':
        cls = 'TR2-3'
    else:
        cls = 'TR' + anno_info['ti-rads_level']
    
    
    if nodule_name in train_nodule_list:
        meta_file_train.append({"file_name": im_name,
                                "text": text,
                                "cls": cls})
    
    if nodule_name in test_nodule_list:
        meta_file_test.append({"file_name": im_name,
                               "text": text,
                               "cls": cls})
    
with open(f"{data_p}/train_metadata.jsonl", 'w') as f_new:
    for item in meta_file_train:
        json.dump(item, f_new)
        f_new.write('\n')
f_new.close()

with open(f"{data_p}/test_metadata.jsonl", 'w') as f:
    for item in meta_file_test:
        json.dump(item, f)
        f.write('\n')
f.close()


