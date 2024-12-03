import hydra
from omegaconf import OmegaConf
import torch
import os
import pyrootutils
from PIL import Image

pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

device = 'cuda:0'
dtype = torch.float16
dtype_str = 'fp16'

#For pre-training
llm_cfg_path = 'configs/clm_models/mistral7b_lora.yaml'
#For instruction tuning
llm_cfg_path = 'configs/clm_models/mistral7b_lora_pretrain.yaml'

agent_cfg_path = 'configs/clm_models/agent_7b_in64_out64_video_gmm_pretrain.yaml'
save_dir = 'your_pretrained_lora_path-merged'

os.makedirs(save_dir, exist_ok=True)
llm_save_dir = os.path.join(save_dir, 'llm')
agent_save_dir = os.path.join(save_dir, 'agent')
agent_save_path = os.path.join(agent_save_dir, 'pytorch_model.bin')
os.makedirs(llm_save_dir, exist_ok=True)
os.makedirs(agent_save_dir, exist_ok=True)

llm_cfg = OmegaConf.load(llm_cfg_path)
llm = hydra.utils.instantiate(llm_cfg, torch_dtype=dtype_str)
print('Init llm done.')

agent_model_cfg = OmegaConf.load(agent_cfg_path)
agent_model = hydra.utils.instantiate(agent_model_cfg, llm=llm)

llm = agent_model.llm.merge_and_unload()
agent_model.llm = None

print('merge done')

llm.save_pretrained(llm_save_dir)

torch.save(agent_model.state_dict(), agent_save_path)

print('save_done')
