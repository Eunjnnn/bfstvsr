import argparse
import os
import os.path as osp
import warnings
import cv2
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd

from calc_metric import Metrics
from tools import Tools

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)

############ Arguments ############

parser = argparse.ArgumentParser()
parser.add_argument("--method", "-m", type=str, default="bfstvsr")
parser.add_argument("--save", "-s", type=str, default="save_gopro", help="directory for video saving")
parser.add_argument("--scale", "-scale", type=int, default=2, help="X 2/4/8 interpolation")
parser.add_argument("--eval", "-e", action="store_true", default='flolpips', help="[flolpips, vfips, tOF]")
parser.add_argument("--dataset", "-dst", type=str, default="Gopro", help="dataset name")
parser.add_argument("--dstDir", "-dstDir", type=str, default="test_img/test_bfstvsr_gopro", help="sampled frames")
parser.add_argument("--gtDir", "-gtDir", type=str, default="path/where/gt/GoPro", help="dataset name")
args = parser.parse_args()

############ Preliminary ############

dstDir = args.dstDir
gtDir = args.gtDir
saveDir = args.save
os.makedirs(saveDir, exist_ok=True)

print(dstDir, gtDir)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
SCALE = args.scale
print("Testing on Dataset: ", args.dataset)
print("Save Directory: ", saveDir)

############ build Dataset ############

dataset = args.dataset
videos = sorted(os.listdir(dstDir))
gtvideos = sorted(os.listdir(gtDir))

############ build SCORE model ############
metrics = [args.eval]
print("Building SCORE models...", metrics)


## evaluate FloLPIPS or VFIPS ##

if args.eval == "flolpips" or args.eval == "vfips":
    metric = Metrics(metrics, skip_ref_frames=SCALE)
    print("Done")

    ############ load videos, interpolate, calc metric ############

    scores = {}

    for vid_name in tqdm(videos):
        video_save_dir = osp.join(saveDir, vid_name)
        # os.makedirs(video_save_dir, exist_ok=True)
        dis_video_save_dir = osp.join(saveDir, args.method, vid_name)
        os.makedirs(dis_video_save_dir, exist_ok=True)
        gt_video_save_dir = osp.join(saveDir, 'gt/', vid_name)
        os.makedirs(gt_video_save_dir, exist_ok=True)
        print("dis_video_save_dir : ", dis_video_save_dir)
        print("gt_video_save_dir : ", gt_video_save_dir)

        sequences = [x for x in os.listdir(osp.join(dstDir, vid_name)) if ".jpg" in x or ".png" in x]
        sequences.sort(key=lambda x: int(x[:-4]))
        sequences = [osp.join(dstDir, vid_name, x) for x in sequences]

        gtsequences = [x for x in os.listdir(osp.join(gtDir, vid_name)) if ".jpg" in x or ".png" in x]
        gtsequences.sort(key=lambda x: int(x[:-4]))
        gtsequences = [osp.join(gtDir, vid_name, x) for x in gtsequences]

        # sequences = sorted([osp.join(dstDir, vid_name, x) for x in os.listdir(osp.join(dstDir, vid_name)) if x.endswith(".jpg") or x.endswith(".png")], key=lambda x: int(osp.splitext(osp.basename(x))[0]))
        # gtsequences = sorted([osp.join(gtDir, vid_name, x) for x in os.listdir(osp.join(gtDir, vid_name)) if x.endswith(".jpg") or x.endswith(".png")], key=lambda x: int(osp.splitext(osp.basename(x))[0]))
        
        height, width, _ = cv2.imread(gtsequences[0]).shape
        tot_frames = len(sequences)
        print("VIDEO: ", vid_name, " (%d x %d x %d)" % (tot_frames, height, width))

        # Process frames without using IOBuffer
        # for i, img_path in enumerate(sequences):
        #     img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        #     if img is None:
        #         print(f"Failed to load {img_path}")
        #         continue
            
        #     save_path = osp.join(video_save_dir, f"{i:07d}.png")
        #     cv2.imwrite(save_path, img)

        ############ save .mp4 files for visualization ############
        if args.save:
            print("Saving to .mp4")
            disPth = osp.join(dis_video_save_dir, f"{vid_name}-ref.mp4")
            refPth = osp.join(gt_video_save_dir, f"{vid_name}-GT.mp4")
            print("disPth : ", disPth, "refPth : ", refPth)
            Tools.frames2mp4(osp.join(dstDir, vid_name, "*.png"), disPth, fps=30)
            if not osp.isfile(refPth):
                Tools.frames2mp4(osp.join(gtDir, vid_name, "*.png"), refPth, fps=30, num_frames=tot_frames)
        # exit()

        ############ save .yuv files for calc metric ############
        disPth = osp.join(dis_video_save_dir, f"{vid_name}-ref.yuv")
        refPth = osp.join(gt_video_save_dir, f"{vid_name}-GT.yuv")
        print(disPth, refPth)
        Tools.frames2rawvideo(osp.join(dstDir, vid_name, "*.png"), disPth)
        if not osp.isfile(refPth):
            Tools.frames2rawvideo(osp.join(gtDir, vid_name, "*.png"), refPth)

        ############ calc metric ############
        meta = dict(
            disImgs=osp.join(dstDir, vid_name),
            refImgs=osp.join(gtDir, vid_name),
            disMP4=disPth,
            refMP4=refPth,
            scale=SCALE,
            hwt=(height, width, tot_frames),
        )
        print("Calculating metrics")
        s = metric.eval(meta)
        scores[vid_name] = s
        print({k: f"{v:.2f}" for k, v in s.items()})

    # Save result to txt
    avg_score = {k: np.mean([v[k] for v in scores.values()]) for k in metrics}
    print("AVG Score of %s".center(41, "=") % args.method)
    for k, v in avg_score.items():
        print("{:<10} {:<10.3f}".format(k, v))
    
    os.makedirs("scores", exist_ok=True)
    result_file = f"scores/{dataset}X{SCALE}.txt"
    need_head = not osp.isfile(result_file)
    with open(result_file, "a+") as f:
        if need_head:
            f.write("{:<10} {}\n".format("methods", " ".join(metrics)))
        f.write("{:<10} {}\n".format(args.method, " ".join(f"{avg_score[x]:<10.3f}" for x in metrics)))


## evaluate tOF ##

elif args.eval == "tOF":
    keys = metrics
    len_dict = dict.fromkeys(["TotalLength_" + _ for _ in keys], 0)
    Total_avg_dict = dict.fromkeys(["TotalAvg_" + _ for _ in keys], 0)

    total_list_dict = {}
    key_str = 'Metrics -->'
    for key_i in keys:
        total_list_dict[key_i] = []
        key_str += ' ' + str(key_i)
    key_str += ' will be measured.'
    print(key_str)

    for vid_idx, vid_name in enumerate(tqdm(videos)):
        per_scene_list_dict = {}
        for key_i in keys:
            per_scene_list_dict[key_i] = []
            
        print("Processing Video: ", vid_name)

        sequences = [x for x in os.listdir(osp.join(dstDir, vid_name)) if ".jpg" in x or ".png" in x]
        sequences.sort(key=lambda x: int(x[:-4]))
        sequences = [osp.join(dstDir, vid_name, x) for x in sequences]

        gtsequences = [x for x in os.listdir(osp.join(gtDir, vid_name)) if ".jpg" in x or ".png" in x]
        gtsequences.sort(key=lambda x: int(x[:-4]))
        gtsequences = [osp.join(gtDir, vid_name, x) for x in gtsequences]
        
        tot_frames = len(sequences)
        
        ############ calc metric ############
        for frameIndex, (pred_frame, gt_frame) in enumerate(tqdm(zip(sequences, gtsequences[:tot_frames]))):
                    
            if frameIndex == 0:
                pre_out_grey = cv2.cvtColor(cv2.imread(pred_frame).astype(np.float32),
                                            cv2.COLOR_BGR2GRAY)  #### CAUTION BRG
                pre_tar_grey = cv2.cvtColor(cv2.imread(gt_frame).astype(np.float32),
                                            cv2.COLOR_BGR2GRAY)  #### CAUTION BRG
                continue
            output_img = cv2.imread(pred_frame).astype(np.float32)  # BGR, [0,255]
            target_img = cv2.imread(gt_frame).astype(np.float32)  # BGR, [0,255]
                    
            output_grey = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
            target_grey = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)     

            target_OF = cv2.calcOpticalFlowFarneback(pre_tar_grey, target_grey, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            output_OF = cv2.calcOpticalFlowFarneback(pre_out_grey, output_grey, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            
            OF_diff = np.absolute(target_OF - output_OF)

            OF_diff_tmp = np.sqrt(np.sum(OF_diff * OF_diff, axis=-1)).mean()
            total_list_dict["tOF"].append(OF_diff_tmp)
            per_scene_list_dict["tOF"].append(OF_diff_tmp)
        
            pre_out_grey = output_grey
            pre_tar_grey = target_grey
        
        """ per scene """
        per_scene_pd_dict = {}  # per scene
        for cur_key in keys:
            per_scene_cur_list = np.float32(per_scene_list_dict[cur_key])
            per_scene_num_data_sum = per_scene_cur_list.sum()
            per_scene_num_data_len = per_scene_cur_list.shape[0]
            per_scene_num_data_mean = per_scene_num_data_sum / per_scene_num_data_len
            per_scene_pd_dict[vid_name] = pd.Series(np.float32(per_scene_num_data_mean))  # dictionary
            """ accumulation """
            cur_list = np.float32(total_list_dict[cur_key])
            num_data_sum = cur_list.sum()
            num_data_len = cur_list.shape[0]  # accum
            num_data_mean = num_data_sum / num_data_len
            print(" %s, (per scene) max %02.4f, min %02.4f, avg %02.4f" %
                    (vid_name, per_scene_cur_list.max(), per_scene_cur_list.min(), per_scene_num_data_mean))
            
            Total_avg_dict["TotalAvg_" + cur_key] = num_data_mean  # accum, update every iteration.
            len_dict["TotalLength_" + cur_key] = num_data_len  # accum, update every iteration.
        
        mode = 'w' if vid_idx == 0 else 'a'
        
        total_csv_path = os.path.join(f"total_metrics_tof_{args.method}_{args.dataset}.csv")
        pd.DataFrame(per_scene_pd_dict).to_csv(total_csv_path, mode=mode)

    """ combining all results after looping all scenes. """
    for key in keys:
        Total_avg_dict["TotalAvg_" + key] = pd.Series(
            np.float32(Total_avg_dict["TotalAvg_" + key]))  # replace key (update)
        len_dict["TotalLength_" + key] = pd.Series(
            np.float32(len_dict["TotalLength_" + key]))  # replace key (update)

    pd.DataFrame(len_dict).to_csv(total_csv_path, mode='a')
    pd.DataFrame(Total_avg_dict).to_csv(total_csv_path, mode='a')
    print("csv file of all metrics for all scenes has been saved in [%s]" %
            (total_csv_path))
    print("Finished.")
