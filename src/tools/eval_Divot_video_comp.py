import hydra
from omegaconf import OmegaConf
import torch
import os
import re
import cv2
import numpy as np
import json
from decord import VideoReader
from decord import cpu
import pyrootutils
import math
from PIL import Image

pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

IMG_BOI_TOKEN = "<img>"
IMG_EOI_TOKEN = "</img>"
VID_BOI_TOKEN = "<vid>"
VID_EOI_TOKEN = "</vid>"
BOI_TOKEN = "<frame>"
EOI_TOKEN = "</frame>"
FRAME_TOKEN = '<frame_{:05d}>'

device = 'cuda:0'
dtype = torch.float16
dtype_str = 'fp16'

tokenizer_cfg_path = 'configs/tokenizer/Divot_mistral_tokenizer_instruct.yaml'
image_transform_cfg_path = 'configs/processer/dc_256_transform.yaml'
visual_encoder_cfg_path = 'configs/visual_encoder/Divot_video.yaml'

##for pre-trained model
# use_instruction = False
# llm_cfg_path = 'configs/clm_models/mistral7b_merged_pretrain.yaml'
# agent_cfg_path = 'configs/clm_models/agent_7b_in64_out64_video_gmm_pretrain_comp.yaml'

##for instruction tuned model
use_instruction = True
llm_cfg_path = 'configs/clm_models/mistral7b_merged_sft_comp.yaml'
agent_cfg_path = 'configs/clm_models/agent_7b_in64_out64_video_gmm_sft_comp.yaml'

tokenizer_cfg = OmegaConf.load(tokenizer_cfg_path)
tokenizer = hydra.utils.instantiate(tokenizer_cfg)

image_transform_cfg = OmegaConf.load(image_transform_cfg_path)
image_transform = hydra.utils.instantiate(image_transform_cfg)

visual_encoder_cfg = OmegaConf.load(visual_encoder_cfg_path)
visual_encoder = hydra.utils.instantiate(visual_encoder_cfg)
visual_encoder.eval().to(device, dtype=dtype)
print('Init visual encoder done')

llm_cfg = OmegaConf.load(llm_cfg_path)
llm = hydra.utils.instantiate(llm_cfg, torch_dtype=dtype)
llm.eval().to(device, dtype=dtype)
print('Init llm done.')

agent_model_cfg = OmegaConf.load(agent_cfg_path)
agent_model = hydra.utils.instantiate(agent_model_cfg, llm=llm)
agent_model.eval().to(device, dtype=dtype)
print('Init agent mdoel Done')

fps = 8
max_vid_length = 16
stride = 4
num_clips = 20

question = 'What does this video describe?'
instruction_prompt = "[INST] {instruction} [/INST]\n"

video_path = 'demo_videos/wine.mp4'
vr = VideoReader(video_path, ctx=cpu(0))

sample_interval = vr.get_avg_fps() / fps
total_frames = len(vr) // sample_interval

sampled_indices_all = []

if total_frames >= num_clips * max_vid_length:
    clip_length = len(vr) // num_clips

    for i in range(num_clips):
        sample_interval = vr.get_avg_fps() / fps
        num_frames_per_clip = int(min(max_vid_length, clip_length // sample_interval))

        if num_frames_per_clip != max_vid_length:
            sample_interval = clip_length / max_vid_length
            num_frames_per_clip = max_vid_length

        start = i * clip_length
        sampled_indices = np.linspace(
            start, start + num_frames_per_clip * sample_interval, num_frames_per_clip, endpoint=False
        )
        sampled_indices = sampled_indices.astype(np.int32)

        if stride != 0.0:
            sampled_indices_ori = sampled_indices
            sampled_indices = sampled_indices[0::stride]
            sampled_indices = np.append(sampled_indices, sampled_indices_ori[-1])

        assert (sampled_indices < len(vr)).all(), "sample indices out of range"

        sampled_indices_all.extend(sampled_indices)

    num_clips_real = num_clips
else:
    start = 0
    num_clips_real = 0
    while start < len(vr):
        end = start + max_vid_length * sample_interval

        if end > len(vr):
            end = len(vr)

        sampled_indices = np.linspace(
            start, end, max_vid_length, endpoint=False
        )
        sampled_indices = sampled_indices.astype(np.int32)
        start += max_vid_length * sample_interval

        if len(sampled_indices) == max_vid_length:
            if stride != 0.0:
                sampled_indices_ori = sampled_indices
                sampled_indices = sampled_indices[0::stride]
                sampled_indices = np.append(sampled_indices, sampled_indices_ori[-1])
            sampled_indices_all.extend(sampled_indices)
            num_clips_real += 1

frames = vr.get_batch(sampled_indices_all).asnumpy().astype(np.uint8)

print(len(vr), vr.get_avg_fps(), frames.shape)

frames = [Image.fromarray(frame) for frame in frames]
frames = [image_transform(frame) for frame in frames]
frames = torch.stack(frames, dim=0).to(device, dtype=dtype).unsqueeze(0)

if stride != 0.0:
    real_vid_length = int(max_vid_length / stride) + 1
else:
        real_vid_length = max_vid_length

with torch.no_grad():
    bz = frames.shape[0]
    frames = frames.reshape(-1, real_vid_length, frames.shape[2], frames.shape[3], frames.shape[4])
    video_embeds = visual_encoder(frames)
    num_video_tokens = video_embeds.shape[1]
    video_embeds = video_embeds.reshape(bz, -1, num_video_tokens, video_embeds.shape[-1]).to(device, dtype=dtype)

    print(frames.shape, video_embeds.shape)

    video_tokens = VID_BOI_TOKEN + (BOI_TOKEN + ''.join([FRAME_TOKEN.format(int(item)) for item in range(num_video_tokens)]) + EOI_TOKEN) * num_clips_real + VID_EOI_TOKEN

    if use_instruction:
        instruction = instruction_prompt.format_map({'instruction': video_tokens + question})
        instruction_ids = tokenizer.encode(instruction, add_special_tokens=False)
        input_ids = [tokenizer.bos_token_id] + instruction_ids
    else:
        video_ids = tokenizer.encode(video_tokens, add_special_tokens=False)
        input_ids = [tokenizer.bos_token_id] + video_ids

    ids_cmp_mask = [False] * len(input_ids)
    attention_mask = [1] * len(input_ids)
    embeds_cmp_mask = [True] * num_clips_real

    input_ids = torch.tensor(input_ids).to(device, dtype=torch.long)
    ids_cmp_mask = torch.tensor(ids_cmp_mask).to(device, dtype=torch.bool)
    attention_mask = torch.tensor(attention_mask).to(device, dtype=torch.long)
    embeds_cmp_mask = torch.tensor(embeds_cmp_mask).to(device, dtype=torch.bool)

    boi_token_id = tokenizer.encode(BOI_TOKEN, add_special_tokens=False)[0]
    eoi_token_id = tokenizer.encode(EOI_TOKEN, add_special_tokens=False)[0]

    boi_idx = torch.where(input_ids == boi_token_id)[0].tolist()
    eoi_idx = torch.where(input_ids == eoi_token_id)[0].tolist()

    for i in range(len(boi_idx)):
        ids_cmp_mask[boi_idx[i] + 1:eoi_idx[i]] = True

    text = agent_model.generate(tokenizer=tokenizer,
                                input_ids=input_ids.unsqueeze(0),
                                video_embeds=video_embeds,
                                ids_cmp_mask=ids_cmp_mask.unsqueeze(0),
                                embeds_cmp_mask=embeds_cmp_mask.unsqueeze(0),
                                device=device,
                                for_text=True,
                                eos_token=tokenizer.eos_token)
    print(text)
