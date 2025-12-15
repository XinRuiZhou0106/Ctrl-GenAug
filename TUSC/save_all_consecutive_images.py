"""Delete non-consecutive frames"""

import os
import glob
import numpy as np
import shutil
from tqdm import tqdm

nodule_list = os.listdir("TUSC/TUSC_all_images")
nodule_list = list(np.unique([i.split("_")[0] for i in nodule_list]))

frames_list = os.listdir("TUSC/TUSC_all_images")

need_to_del = {}

with tqdm(total=len(nodule_list)) as pbar:
    for nodule_id in nodule_list:
        nodule_frames_path = list(glob.glob(f"TUSC/TUSC_all_images/{nodule_id}_*"))
        nodule_frames_path = sorted(nodule_frames_path, key=lambda x: int(x.split('/')[-1].split('_')[-1].split('.')[0]))
        start_id = int(nodule_frames_path[0].split('/')[-1].split('_')[-1].split('.')[0])
        for i in range(1, len(nodule_frames_path)):
            if (start_id + i) != int(nodule_frames_path[i].split('/')[-1].split('_')[-1].split('.')[0]):
                need_to_del[nodule_id] = nodule_frames_path[i:]
                break
        pbar.update(1)
pbar.close()

# save the only consecutive frames
new_data_dir = "TUSC/TUSC_all_images_consecutive"
os.makedirs(new_data_dir, exist_ok=True)
need_to_del_alllist = []
for del_frames in need_to_del.values():
    # After running this script, please manually delete Nodule4_1.png from the TUSC_all_images_consecutive folder
    if "Nodule4" in del_frames[0]:
        continue
    need_to_del_alllist += del_frames
for data_p in glob.glob("TUSC/TUSC_all_images/*"):
    if data_p not in need_to_del_alllist:
        shutil.copy(data_p, os.path.join(new_data_dir, data_p.split('/')[-1]))
        
        