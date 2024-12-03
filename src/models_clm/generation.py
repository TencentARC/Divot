import torch
from transformers import LogitsProcessor

IMG_BOI_TOKEN = "<img>"
IMG_EOI_TOKEN = "</img>"
VID_BOI_TOKEN = "<vid>"
VID_EOI_TOKEN = "</vid>"
BOI_TOKEN = "<frame>"
EOI_TOKEN = "</frame>"
FRAME_TOKEN = '<frame_{:05d}>'

class AutoVideoTokenGenerationProcessor(LogitsProcessor):

    def __init__(self, tokenizer, num_vid_gen_tokens=100) -> None:
        super().__init__()
        vid_all_token_str = ''.join([BOI_TOKEN] + [FRAME_TOKEN.format(int(item))
                                                   for item in range(num_vid_gen_tokens)] + [EOI_TOKEN])
        self.vid_ids_list = tokenizer.encode(vid_all_token_str, add_special_tokens=False)

    def __call__(self, input_ids, scores):
        bz = input_ids.shape[0]
        for i in range(bz):
            cur_input_id = input_ids[i, -1].item()
            if cur_input_id in self.vid_ids_list[:-1]:

                output_id = self.vid_ids_list[self.vid_ids_list.index(cur_input_id) + 1]
                scores[i, ..., output_id] = scores[i, ...].max() + 10.
            else:

                scores[i, ..., torch.tensor(self.vid_ids_list[1:]).to(dtype=torch.long)] = 0.0

        return scores
