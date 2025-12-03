import torch
import time
import numpy as np
import cv2
import glob
import os
import random
import imageio
from argparse import ArgumentParser
from mvextractor.videocap import VideoCap
from skimage import img_as_ubyte

def draw_motion_fields(frame, motion_fields):
    if len(motion_fields) > 0:
        num_mfs = np.shape(motion_fields)[0]
        for mf in np.split(motion_fields, num_mfs):
            start_pt = (mf[0, 3], mf[0, 4])
            end_pt = (mf[0, 5], mf[0, 6])
            cv2.arrowedLine(frame, start_pt, end_pt, (0, 255, 255), 3, cv2.LINE_AA, 0, 0.1)
            # cv2.arrowedLine(frame, start_pt, end_pt, (0, 0, 255), 2, cv2.LINE_AA, 0, 0.2)
    return frame

def extract_motion_fields(input_video, verbose=False, visual_mf=False):
    cap = VideoCap()
    ret = cap.open(input_video)
    if not ret:
        raise RuntimeError(f"Could not open {input_video}")
    
    step = 0
    times = []

    frame_types = []
    frames = []
    mfs = []
    mfs_visual = []

    while True:
        if verbose:
            print("Frame: ", step, end=" ")

        tstart = time.perf_counter()

        # read next video frame and corresponding motion fields
        ret, frame, motion_fields, frame_type, timestamp = cap.read()

        tend = time.perf_counter()
        telapsed = tend - tstart
        times.append(telapsed)

        # if there is an error reading the frame
        if not ret:
            if verbose:
                print("No frame read. Stopping.")
            break

        frame_save = np.zeros(frame.copy().shape, dtype=np.uint8) # *255
        if visual_mf:
            frame_save = draw_motion_fields(frame_save, motion_fields)

        # store motion fields, frames, etc. in output directory
        mfs_visual.append(frame_save)

        h,w = frame.shape[:2]
        mf = np.zeros((h,w,2))
        position = motion_fields[:,5:7].clip((0,0),(w-1,h-1)) # x->w, y->h 
            
        # horizontal and vertical orientations
        mf[position[:,1],position[:,0]]=motion_fields[:,0:1]*motion_fields[:,7:9]/motion_fields[:, 9:] 

        step += 1
        frame_types.append(frame_type)
        frames.append(frame)
        mfs.append(mf)
        
    if verbose:
        print("average dt: ", np.mean(times))
    cap.release()

    return frame_types, frames, mfs, mfs_visual

def keys_with_same_value(dictionary):
    result = {}
    for key, value in dictionary.items():
        if value not in result:
            result[value] = [key]
        else:
            result[value].append(key)

    conflict_points = {}
    for k in result.keys():
        if len(result[k]) > 1:
            conflict_points[k] = result[k]
    return conflict_points

def find_duplicates(input_list):
    seen = set()
    duplicates = set()

    for item in input_list:
        if item in seen:
            duplicates.add(item)
        else:
            seen.add(item)

    return list(duplicates)

def neighbors_index(point, window_size, H, W):
    """return the spatial neighbor indices"""
    t, x, y = point
    neighbors = []
    for i in range(-window_size, window_size + 1):
        for j in range(-window_size, window_size + 1):
            if i == 0 and j == 0:
                continue
            if x + i < 0 or x + i >= H or y + j < 0 or y + j >= W:
                continue
            neighbors.append((t, x + i, y + j))
    return neighbors

def sample_trajectories_based_on_motion_fields(motion_fields):     
    motion_fields = motion_fields/256
    resolutions = [32, 16, 8, 4]
    res = {}
    window_sizes = {32: 1,
                    16: 1,
                    8: 1,
                    4: 1}

    for resolution in resolutions:
        print("="*30)
        trajectories = {}
        motion_fields_resolu = torch.round(resolution*torch.nn.functional.interpolate(motion_fields, scale_factor=(resolution/256, resolution/256)))

        T = motion_fields_resolu.shape[0]+1
        H = motion_fields_resolu.shape[2]
        W = motion_fields_resolu.shape[3]

        is_activated = torch.zeros([T, H, W], dtype=torch.bool)

        for t in range(T-1):
            motion_field = motion_fields_resolu[t] # t -> t+1 mf
            for h in range(H):
                for w in range(W):
                    if not is_activated[t, h, w]:
                        is_activated[t, h, w] = True
                        x = h + int(motion_field[1, h, w])
                        y = w + int(motion_field[0, h, w])
                        if x >= 0 and x < H and y >= 0 and y < W:
                            trajectories[(t, h, w)]= (t+1, x, y) 

        conflict_points = keys_with_same_value(trajectories) # Locate patch points with the same trajectory

        for k in conflict_points:
            index_to_pop = random.randint(0, len(conflict_points[k]) - 1)
            conflict_points[k].pop(index_to_pop) # This trajectory is retained
            for point in conflict_points[k]:
                if point[0] != T-1:
                    trajectories[point]= (-1, -1, -1) # need to be masked

        active_traj = []
        all_traj = []
        for t in range(T):
            pixel_set = {(t, x//H, x%H):0 for x in range(H*W)}
            new_active_traj = []
            for traj in active_traj:
                if traj[-1] in trajectories:
                    v = trajectories[traj[-1]]
                    new_active_traj.append(traj + [v])
                    pixel_set[v] = 1
                else:
                    all_traj.append(traj) 
            active_traj = new_active_traj
            active_traj+=[[pixel] for pixel in pixel_set if pixel_set[pixel] == 0]
        all_traj += active_traj

        useful_traj = [i for i in all_traj if len(i)>1]
        for idx in range(len(useful_traj)):
            if useful_traj[idx][-1] == (-1, -1, -1): 
                useful_traj[idx] = useful_traj[idx][:-1]
        print("how many points in all trajectories for resolution{}?".format(resolution), sum([len(i) for i in useful_traj]))
        print("how many points in the video for resolution{}?".format(resolution), T*H*W)

        # validate if there are no duplicates in the trajectories
        trajs = []
        for traj in useful_traj:
            trajs = trajs + traj
        assert len(find_duplicates(trajs)) == 0, "There should not be duplicates in the useful trajectories."

        # check if non-appearing points + appearing points = all the points in the video
        all_points = set([(t, x, y) for t in range(T) for x in range(H) for y in range(W)])
        left_points = all_points- set(trajs)
        print("How many points not in the trajectories for resolution{}?".format(resolution), len(left_points))
        for p in list(left_points):
            useful_traj.append([p])
        print("how many points in all trajectories for resolution{} after pending?".format(resolution), sum([len(i) for i in useful_traj]))

        longest_length = max([len(i) for i in useful_traj])
        sequence_length = (window_sizes[resolution]*2+1)**2 + longest_length - 1

        seqs = []
        masks = []

        # create a dictionary to facilitate checking the trajectories to which each point belongs
        point_to_traj = {}
        for traj in useful_traj:
            for p in traj:
                point_to_traj[p] = traj

        for t in range(T): # if a patch does not have a corresponding trajectory, it will only attend to its spatial window
            for x in range(H):
                for y in range(W):
                    neighbours = neighbors_index((t,x,y), window_sizes[resolution], H, W)
                    sequence = [(t,x,y)]+neighbours + [(0,0,0) for i in range((window_sizes[resolution]*2+1)**2-1-len(neighbours))]
                    sequence_mask = torch.zeros(sequence_length, dtype=torch.bool)
                    sequence_mask[:len(neighbours)+1] = True

                    traj = point_to_traj[(t,x,y)].copy()
                    traj.remove((t,x,y))
                    sequence = sequence + traj + [(0,0,0) for k in range(longest_length-1-len(traj))]
                    sequence_mask[(window_sizes[resolution]*2+1)**2: (window_sizes[resolution]*2+1)**2 + len(traj)] = True

                    seqs.append(sequence)
                    masks.append(sequence_mask)

        seqs = torch.tensor(seqs)
        masks = torch.stack(masks)
        res["traj{}".format(resolution)] = seqs
        res["mask{}".format(resolution)] = masks
    return res
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--clip_data_dir', type=str, required=True, help='Directory containing the preprocessed clips')
    args = parser.parse_args()

    clip_data_dir = args.clip_data_dir

    save_mf_dir = f"{clip_data_dir.split('_')[0]}_motion_field"
    os.makedirs(save_mf_dir, exist_ok=True)

    save_traj_dir = f"{clip_data_dir.split('_')[0]}_motion_field_trajectory"
    os.makedirs(save_traj_dir, exist_ok=True)
    
    for filename in glob.glob(f"{clip_data_dir}/*.avi"):
        filename_mp4 = filename.replace("avi", "mp4")
        
        # avi -> mp4
        if not os.path.exists(filename_mp4):
            video = []
            cap_m = cv2.VideoCapture(filename)
            while True:
                ret, frame = cap_m.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis]
                    video.append(frame.repeat(3, 2))
                else:
                    break
            imageio.mimsave(filename_mp4, [img_as_ubyte(f) for f in video], fps=2.5)
        
        frame_types, frames, mfs, mfs_visual = extract_motion_fields(input_video=filename_mp4, visual_mf=False)
        assert list(np.unique(mfs[0])) == [0]
        
        # save motion fields as npy files
        mfs_npy = mfs.copy()
        mfs_npy = [m.astype(np.float32).transpose((2,0,1)) for m in mfs_npy[1:]] # delete the first frame/slice
        mfs_npy = np.stack(mfs_npy)
        frag_name = filename.split('/')[-1].split('.')[0]
        np.save(f"{save_mf_dir}/{frag_name}.npy", mfs_npy) # np.float32 [frame_num-1, 2(x&y), h, w]
        
        """prepare for trajectory sampling"""
        assert np.unique(mfs[0]).tolist() == [0.0]

        # Extract the motion field vectors (horizontal and vertical) between adjacent frames
        motion_fields = [torch.from_numpy(m.transpose((2,0,1))) for m in mfs[1:]]
        motion_fields = torch.stack(motion_fields)
        
        # motion field-based patch trajectory sampling for Motion Field Attention (MFA)
        trajectories = sample_trajectories_based_on_motion_fields(motion_fields)
        np.save(f"{save_traj_dir}/{filename.split('/')[-1].replace('.avi', '.npy')}", trajectories)
        
        # remove the temp file
        os.remove(filename_mp4)
        
    
    
