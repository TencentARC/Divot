import hydra
from omegaconf import OmegaConf
import torch
import os
import re
import torchvision
import pyrootutils
from PIL import Image
import cv2
import numpy as np

torch.manual_seed(42)

pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

device = 'cuda:0'
dtype = torch.float32
dtype_str = 'fp32'

video_frames = 16
fps = 8
stride = 4

save_dir= 'video_recon'
os.makedirs(save_dir, exist_ok=True)

image_transform_cfg_path = 'configs/processer/dc_256_transform.yaml'
visual_encoder_cfg_path = 'configs/visual_encoder/Divot_video.yaml'
detokenizer_cfg_path = 'configs/video_detokenizer/Divot_detokenizer_stage1.yaml'

image_transform_cfg = OmegaConf.load(image_transform_cfg_path)
image_transform = hydra.utils.instantiate(image_transform_cfg)

visual_encoder_cfg = OmegaConf.load(visual_encoder_cfg_path)
visual_encoder = hydra.utils.instantiate(visual_encoder_cfg)
visual_encoder.eval().to(device, dtype=dtype)
print('Init visual encoder done')

detokenizer_cfg = OmegaConf.load(detokenizer_cfg_path)
detokenizer = hydra.utils.instantiate(detokenizer_cfg, video_tokenizer=visual_encoder)
detokenizer.eval().to(device, dtype=torch.float32)
print('Init detokenizer Done')

video_path = 'demo_videos/car.mp4'

cap = cv2.VideoCapture(video_path)

video_fps = cap.get(cv2.CAP_PROP_FPS)
video_num_frames = 0
frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    video_num_frames += 1
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append(frame)

frames = np.array(frames).astype(np.uint8)

sample_interval = video_fps / fps
max_frames = int(video_num_frames // sample_interval)

if max_frames < video_frames:
    sample_interval = int(video_num_frames / video_frames)
    max_frames = video_frames

start = 0
sampled_indices = np.linspace(
    start, start + max_frames * sample_interval, max_frames, endpoint=False
)
sampled_indices = sampled_indices[:video_frames].astype(np.int32)
sampled_indices_all = sampled_indices[0::stride]
sampled_indices_all = np.append(sampled_indices_all, sampled_indices[-1])
frames = frames[sampled_indices_all]
print(sampled_indices_all, frames.shape)

cap.release()

frames= [Image.fromarray(frame) for frame in frames]
frames = [image_transform(frame) for frame in frames]
frames = torch.stack(frames, dim=0).to(device, dtype).unsqueeze(0)

with torch.no_grad():
    video_features = visual_encoder(frames)
    decoded_video = detokenizer(video_features.to(torch.float32))
    print(decoded_video.shape)

    video = decoded_video.detach().cpu()
    video = torch.clamp(video.float(), -1., 1.)

    grid = video[0,...]
    grid = (grid + 1.0) / 2.0
    grid = (grid * 255).to(torch.uint8).permute(1, 2, 3, 0) #thwc
    path = os.path.join(save_dir, video_path.split('/')[-1])
    torchvision.io.write_video(path, grid, fps=8, video_codec='h264', options={'crf': '10'})

