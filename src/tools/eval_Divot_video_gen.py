import hydra
from omegaconf import OmegaConf
import torch
import os
import re
import torchvision
import pyrootutils
from PIL import Image

torch.manual_seed(42)

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
is_video = True

num_vid_in_tokens = 64
num_vid_out_tokens = 64
var_scale = 0.1

save_dir= 'video_gen'
os.makedirs(save_dir, exist_ok=True)

speacial_tokens = BOI_TOKEN + ''.join([FRAME_TOKEN.format(int(item)) for item in range(num_vid_in_tokens)])

if not is_video:
    instruction_prompt = "[INST] Generate an image of {caption} [/INST]\n"
    eos_token = IMG_BOI_TOKEN
else:
    instruction_prompt = "[INST] Generate a video of {caption} [/INST]\n"
    eos_token = VID_BOI_TOKEN

tokenizer_cfg_path = 'configs/tokenizer/Divot_mistral_tokenizer_instruct.yaml'
llm_cfg_path = 'configs/clm_models/mistral7b_merged_sft_gen.yaml'
agent_cfg_path = 'configs/clm_models/agent_7b_in64_out64_video_gmm_sft_gen.yaml'
visual_encoder_cfg_path = 'configs/visual_encoder/Divot_video.yaml'
detokenizer_cfg_path = 'configs/video_detokenizer/Divot_detokenizer_stage2.yaml'

tokenizer_cfg = OmegaConf.load(tokenizer_cfg_path)
tokenizer = hydra.utils.instantiate(tokenizer_cfg)

visual_encoder_cfg = OmegaConf.load(visual_encoder_cfg_path)
visual_encoder = hydra.utils.instantiate(visual_encoder_cfg)
visual_encoder.eval().to(device, dtype=dtype)
print('Init visual encoder done')

llm_cfg = OmegaConf.load(llm_cfg_path)
llm = hydra.utils.instantiate(llm_cfg, torch_dtype=dtype)
print('Init llm done.')

agent_model_cfg = OmegaConf.load(agent_cfg_path)
agent_model = hydra.utils.instantiate(agent_model_cfg, llm=llm)
agent_model.eval().to(device, dtype=dtype)
print('Init agent mdoel Done')

detokenizer_cfg = OmegaConf.load(detokenizer_cfg_path)
detokenizer = hydra.utils.instantiate(detokenizer_cfg, video_tokenizer=visual_encoder)
detokenizer.eval().to(device, dtype=torch.float32)
print('Init detokenizer Done')

captions = [
    "A time-lapse of clouds passing over a peaceful mountain lake with reflections of the peaks",
    "A gorgeous girl is smiling",
    "People cheer at fireworks display",
    "An oil painting featuring a beach with waves",
]

boi_token_id = tokenizer.encode(BOI_TOKEN, add_special_tokens=False)[0]

for caption in captions:
    prompt = instruction_prompt.format_map({'caption': caption})
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = [tokenizer.bos_token_id] + prompt_ids
    attention_mask = [1] * len(input_ids)
    attention_mask = torch.tensor(attention_mask).to(device, dtype=torch.long).unsqueeze(0)
    input_ids = torch.tensor(input_ids).to(device, dtype=torch.long).unsqueeze(0)
    output = agent_model.generate(tokenizer=tokenizer, input_ids=input_ids, attention_mask=attention_mask, for_text=True, eos_token=eos_token, device=device)

    prompt = instruction_prompt.format_map({'caption': caption}) + output + speacial_tokens
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = [tokenizer.bos_token_id] + prompt_ids

    boi_idx = input_ids.index(boi_token_id)
    ids_gen_mask = [0] * len(input_ids)
    ids_gen_mask[boi_idx+1:] = [1] * num_vid_in_tokens
    attention_mask = [1] * len(input_ids)
    attention_mask[boi_idx+1:] = [-1] * num_vid_in_tokens

    ids_gen_mask = torch.tensor(ids_gen_mask).to(device, dtype=torch.bool).unsqueeze(0)
    attention_mask = torch.tensor(attention_mask).to(device, dtype=torch.long).unsqueeze(0)
    input_ids = torch.tensor(input_ids).to(device, dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        output = agent_model.generate(tokenizer=tokenizer, input_ids=input_ids, attention_mask=attention_mask, ids_gen_mask=ids_gen_mask, num_vid_in_tokens=num_vid_in_tokens, num_vid_out_tokens=num_vid_out_tokens, var_scale=var_scale, device=device)
        print(output.shape)

        decoded_video = detokenizer(output.to(torch.float32))
        print(decoded_video.shape)

        video = decoded_video.detach().cpu()
        video = torch.clamp(video.float(), -1., 1.)

        grid = video[0,...]
        grid = (grid + 1.0) / 2.0
        grid = (grid * 255).to(torch.uint8).permute(1, 2, 3, 0) #thwc
        path = os.path.join(save_dir, caption + '.mp4')
        torchvision.io.write_video(path, grid, fps=8, video_codec='h264', options={'crf': '10'})
