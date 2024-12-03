import torchdata.datapipes as dp
from PIL import Image
import functools
import numpy as np
import torch
import os
import av
import torch.distributed as dist
import hydra
import imageio
from braceexpand import braceexpand
import pyrootutils
import cv2
from decord import VideoReader
from decord import cpu
import random
import imageio

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

IMG_BOI_TOKEN = "<img>"
IMG_EOI_TOKEN = "</img>"
VID_BOI_TOKEN = "<vid>"
VID_EOI_TOKEN = "</vid>"
BOI_TOKEN = "<frame>"
EOI_TOKEN = "</frame>"
FRAME_TOKEN = '<frame_{:05d}>'

gen_prompt_image_all = [
    "Please show me a picture of",
    "Please design an image of",
    "Please produce a photo of",
    "Please generate an image of",
    "Please draw a painting of",
    "I'd like to see a drawing of",
    "I'd love to see an illustration of",
    "I'd like to view an image of",
    "I want to see a picture of",
    "I would like to see a photo of",
    "Show me a photo of",
    "Generate a picture of",
    "Show me a photograph of",
    "Generate an image of",
    "Generate an image:",
    "Generate a picture:",
    "Generate a painting:",
    "Generate a photograph:",
    "Show me a photograph:",
    "Draw a picture:",
    "Draw a painting:",
    "Draw an image:",
]

gen_prompt_video_all = [
    "Please show me a video of",
    "Please create a video of",
    "Please produce a video of",
    "Please generate a video of",
    "I'd like to see a video of",
    "I'd love to see a video of",
    "I'd like to view a video of",
    "I want to see a video of",
    "I would like to see a video of",
    "Show me a video of",
    "Generate a video of",
    "Create a video of",
    "Produce a video of",
    "Generate a video:",
    "Create a video:",
    "Produce a video:",
    "Show me a video:",
]


def get_accurate_frames(video_path):
    container = av.open(video_path)
    video_fps = container.streams.video[0].average_rate
    time_base = container.streams.video[0].time_base
    duration_in_seconds = container.streams.video[0].duration * time_base
    video_num_frames = int(duration_in_seconds * video_fps)
    container.close()
    return video_num_frames

def get_vid_shape(video_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height


def unwarp_data(item):
    unwarpped = {}
    for key, value in item.items():
        if isinstance(value, dict):
            unwarpped.update(value)
        elif value is not None:
            unwarpped[key] = value

    return unwarpped


def decode_videogen_img_tar_data(
    item,
    tokenizer,
    image_transform=None,
    max_length=128,
    min_aspect_ratio=0.666,
    num_img_tokens=64,
    num_frames=1,
    bilateral_attention=False,
    instruction_prompt=None,
    video_first=False,
    min_resolution=400,
    drop_text_ratio=0.0,
):
    key, value = item
    num_clips = 1

    if key.endswith(".txt"):
        caption = value.read().decode("utf-8").strip()
        if drop_text_ratio > 0.0 and np.random.rand() < drop_text_ratio:
            caption = ""

        if instruction_prompt is not None:
            caption = random.choice(gen_prompt_image_all) + ' ' + caption
            caption = instruction_prompt.format_map({'instruction': caption})

        caption_ids = tokenizer.encode(caption, add_special_tokens=False)
        if len(caption_ids) + num_img_tokens + 2 + 4 > max_length:
            # Caption is too long to fit the image
            # + 2 is saved for BOI and EOI token, +4 for video and image start and end token
            caption_ids = caption_ids[: (max_length - num_img_tokens - 2 - 4)]

        input_ids = [tokenizer.bos_token_id]
        labels = [-100]
        ids_gen_mask = [False]
        ids_cmp_mask = [False]

        video_tokens = IMG_BOI_TOKEN + (BOI_TOKEN + ''.join([FRAME_TOKEN.format(int(item)) for item in range(num_img_tokens)]) + EOI_TOKEN) * num_clips + IMG_EOI_TOKEN
        video_ids = tokenizer.encode(video_tokens, add_special_tokens=False)

        if video_first:
            input_ids = input_ids + video_ids + caption_ids
            labels = labels + [-100] * len(video_ids) + caption_ids
            ids_cmp_mask = ids_cmp_mask + [False] + ([False] + [True] * num_img_tokens + [False]) * num_clips + [False] + [False] * len(caption_ids)
            ids_gen_mask = ids_gen_mask + [False] * len(video_ids) + [False] * len(caption_ids)
            embeds_cmp_mask = True
            embeds_gen_mask = False
        else:
            input_ids = input_ids + caption_ids + video_ids
            if instruction_prompt is None:
                labels = labels + [-100] * len(caption_ids) + [-100] * len(video_ids)
            else:
                labels = labels + [-100] * len(caption_ids) + [video_ids[0]] + [-100] * (len(video_ids) - 2) + [video_ids[-1]]
            ids_gen_mask = ids_gen_mask + [False] * len(caption_ids) + [False] + ([False] + [True] * num_img_tokens + [False]) * num_clips + [False]
            ids_cmp_mask = ids_cmp_mask + [False] * len(caption_ids) + [False] * len(video_ids)
            embeds_cmp_mask = False
            embeds_gen_mask = True

        input_ids = input_ids + [tokenizer.eos_token_id]
        labels = labels + [tokenizer.eos_token_id]
        ids_gen_mask = ids_gen_mask + [False]
        ids_cmp_mask = ids_cmp_mask + [False]

        if video_first or not bilateral_attention:
            attention_mask = [1] * len(input_ids)
        else:
            ## -1 for bilateral_attention of video query
            attention_mask = [1] + [1] * len(caption_ids) + [1] + ([1] + [-1] * num_img_tokens + [1]) * num_clips + [1] + [1]

        if len(input_ids) > max_length:
            if video_first:
                print("input length exceeds", len(input_ids))
                input_ids = input_ids[:max_length]
                labels = labels[:max_length]
                ids_gen_mask = ids_gen_mask[:max_length]
                ids_cmp_mask = ids_cmp_mask[:max_length]
                attention_mask = attention_mask[:max_length]
            else:
                return {}
        else:
            padding_length = max_length - len(input_ids)
            input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
            labels = labels + [-100] * padding_length
            ids_gen_mask = ids_gen_mask + [False] * padding_length
            ids_cmp_mask = ids_cmp_mask + [False] * padding_length
            attention_mask = attention_mask + [0] * padding_length

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        ids_cmp_mask = torch.tensor(ids_cmp_mask, dtype=torch.bool)
        ids_gen_mask = torch.tensor(ids_gen_mask, dtype=torch.bool)
        embeds_cmp_mask = torch.tensor(embeds_cmp_mask)
        embeds_gen_mask = torch.tensor(embeds_gen_mask)

        return key, {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'ids_gen_mask': ids_gen_mask,
            'ids_cmp_mask': ids_cmp_mask,
            'embeds_gen_mask': embeds_gen_mask,
            'embeds_cmp_mask': embeds_cmp_mask,
        }


    if key.endswith(".jpg"):
        try:
            image = Image.open(value).convert("RGB")
            width, height = image.size
        except Exception as e:
            print("Error while decoding image: ", key)
            return key, {}

        if min(width, height) < min_resolution:
            # print("resolution is too small", key, width, height)
            return key, {}

        if min(width, height) / max(width, height) < min_aspect_ratio:
            # print("aspect ratio is too small", key, width, height)
            return key, {}

        frames = image_transform(image).unsqueeze(0).repeat(num_frames, 1, 1, 1)

        return key, {"frames": frames}

    return key, {}


def build_videogen_img_tar_datapipes(
    data_dir,
    tokenizer=None,
    max_length=77,
    batch_size=None,
    image_transform=None,
    min_aspect_ratio=0.666,
    num_img_tokens=64,
    num_frames=1,
    bilateral_attention=False,
    instruction_prompt=None,
    video_first=False,
    cycle_count=None,
    min_resolution=400,
    drop_text_ratio=0.0,
):
    decode_partial = functools.partial(
        decode_videogen_img_tar_data,
        tokenizer=tokenizer,
        image_transform=image_transform,
        max_length=max_length,
        min_aspect_ratio=min_aspect_ratio,
        num_img_tokens=num_img_tokens,
        num_frames=num_frames,
        bilateral_attention=bilateral_attention,
        instruction_prompt=instruction_prompt,
        video_first=video_first,
        min_resolution=min_resolution,
        drop_text_ratio=drop_text_ratio,
    )

    if isinstance(data_dir, str):
        data_dir = list(braceexpand(data_dir))
    datapipe = dp.iter.FileLister(root=data_dir, masks="*.tar", recursive=True)
    datapipe = datapipe.cycle(count=cycle_count)
    datapipe = datapipe.shuffle()
    datapipe = datapipe.sharding_filter()
    datapipe = datapipe.open_files(mode="b")
    datapipe = datapipe.load_from_tar_wo_exception()
    datapipe = datapipe.map(decode_partial)
    datapipe = datapipe.webdataset()
    datapipe = datapipe.map(unwarp_data)
    datapipe = datapipe.filter(lambda x: "input_ids" in x and "frames" in x)
    if batch_size is not None:
        datapipe = datapipe.batch(batch_size)
        datapipe = datapipe.collate()
    return datapipe


def build_video_caption_jsonl_datapipes(
    data_dir,
    image_dir,
    tokenizer=None,
    max_length=77,
    num_frames=4,
    num_clips=1,
    fps=4,
    stride=0,
    bilateral_attention=False,
    instruction_prompt=None,
    num_video_tokens=64,
    batch_size=None,
    image_transform=None,
    min_aspect_ratio=0.666,
    cycle_count=None,
    min_resolution=400,
    video_first=True,
):
    """
    datapipe of caption dataset (such as CC3M, LAION...) with webdataset format
    """

    decode_partial = functools.partial(
        decode_video_caption_jsonl_data_decord,
        image_dir=image_dir,
        tokenizer=tokenizer,
        image_transform=image_transform,
        max_length=max_length,
        num_frames=num_frames,
        num_clips=num_clips,
        fps=fps,
        stride=stride,
        bilateral_attention=bilateral_attention,
        instruction_prompt=instruction_prompt,
        num_video_tokens=num_video_tokens,
        min_aspect_ratio=min_aspect_ratio,
        min_resolution=min_resolution,
        video_first=video_first,
    )

    if isinstance(data_dir, str):
        data_dir = list(braceexpand(data_dir))
    datapipe = dp.iter.FileLister(root=data_dir, masks="*.jsonl", recursive=True)
    datapipe = datapipe.cycle(count=cycle_count)
    datapipe = datapipe.open_files(mode="r")
    datapipe = datapipe.parse_jsonl_files()
    datapipe = datapipe.shuffle()
    datapipe = datapipe.sharding_filter()
    datapipe = datapipe.map(decode_partial)
    datapipe = datapipe.filter(lambda item: "frames" in item and "input_ids" in item)

    if batch_size is not None:
        datapipe = datapipe.batch(batch_size)
        datapipe = datapipe.collate()

    return datapipe



def decode_video_caption_jsonl_data_decord(
    item,
    image_dir,
    tokenizer,
    image_transform=None,
    max_length=128,
    num_frames=4,
    num_clips=1,
    num_video_tokens=64,
    fps=4,
    stride=0,
    bilateral_attention=False,
    instruction_prompt=None,
    min_aspect_ratio=0.666,
    min_resolution=400,
    video_first=True
):
    key, value = item

    num_clips_require = num_clips
    video_path = value['video']
    caption = value['caption']

    if isinstance(caption, list):
        caption = np.random.choice(caption)

    if instruction_prompt is not None:
        caption = random.choice(gen_prompt_video_all) + ' ' + caption
        caption = instruction_prompt.format_map({'instruction': caption})

    max_vid_length = num_frames

    video_path = os.path.join(image_dir, video_path)

    if not os.path.exists(video_path):
        print("video file does not exists", video_path)
        return {}

    try:
        vr = VideoReader(video_path, ctx=cpu(0))

        width, height = get_vid_shape(video_path)
        if min(width, height) < min_resolution:
            print("resolution is too small", video_path, width, height)
            return {}
        if min(width, height) / max(width, height) < min_aspect_ratio:
            print("aspect ratio is too small", video_path, width, height)
            return {}

        clip_length = len(vr) // num_clips

        video_fps = vr.get_avg_fps()
        sample_interval = video_fps / fps
        total_frames_real = len(vr)
        total_frames = total_frames_real // sample_interval

        sampled_indices_all = []

        if total_frames >= num_clips * max_vid_length:
            clip_length = total_frames_real // num_clips
            sample_interval = video_fps / fps
            num_frames_per_clip = int(min(max_vid_length, clip_length // sample_interval))

            if num_frames_per_clip != max_vid_length:
                sample_interval = clip_length / max_vid_length
                num_frames_per_clip = max_vid_length

            for i in range(num_clips):
                start = np.random.uniform(i * clip_length, (i + 1) * clip_length - num_frames_per_clip * sample_interval)
                sampled_indices = np.linspace(
                    start, start + num_frames_per_clip * sample_interval, num_frames_per_clip, endpoint=False
                )
                sampled_indices = sampled_indices.astype(np.int32)

                if stride != 0:
                    sampled_indices_ori = sampled_indices
                    sampled_indices = sampled_indices_ori[0::stride]
                    sampled_indices = np.append(sampled_indices, sampled_indices_ori[-1])

                assert (sampled_indices < total_frames_real).all(), "sample indices out of range"
                sampled_indices_all.extend(sampled_indices)
        else:
            start = np.random.uniform(0, sample_interval)
            num_clips_real = 0
            while start < total_frames_real:
                end = start + max_vid_length * sample_interval

                if end > total_frames_real:
                    end = total_frames_real

                sampled_indices = np.linspace(
                    start, end, max_vid_length, endpoint=False
                )
                sampled_indices = sampled_indices.astype(np.int32)
                start += max_vid_length * sample_interval

                if len(sampled_indices) == max_vid_length:
                    if stride != 0:
                        sampled_indices_ori = sampled_indices
                        sampled_indices = sampled_indices_ori[0::stride]
                        sampled_indices = np.append(sampled_indices, sampled_indices_ori[-1])

                    sampled_indices_all.extend(sampled_indices)
                    num_clips_real += 1
            num_clips = num_clips_real

        frames = vr.get_batch(sampled_indices_all).asnumpy().astype(np.uint8)

    except:
        print("error in reading video", video_path)
        return {}

    frames = [Image.fromarray(frame) for frame in frames]
    frames = [image_transform(frame) for frame in frames]
    frames = torch.stack(frames, dim=0)

    if stride != 0.0:
        required_frames = (num_frames / stride + 1) * num_clips

    if len(frames) < required_frames:
        return {}

    input_ids = [tokenizer.bos_token_id]
    labels = [-100]
    ids_gen_mask = [False]
    ids_cmp_mask = [False]

    video_tokens = VID_BOI_TOKEN + (BOI_TOKEN + ''.join([FRAME_TOKEN.format(int(item)) for item in range(num_video_tokens)]) + EOI_TOKEN) * num_clips + VID_EOI_TOKEN

    video_ids = tokenizer.encode(video_tokens, add_special_tokens=False)
    caption_ids = tokenizer.encode(caption, add_special_tokens=False)

    if video_first:
        input_ids = input_ids + video_ids + caption_ids
        labels = labels + [-100] * len(video_ids) + caption_ids
        ids_cmp_mask = ids_cmp_mask + [False] + ([False] + [True] * num_video_tokens + [False]) * num_clips + [False] + [False] * len(caption_ids)
        ids_gen_mask = ids_gen_mask + [False] * len(video_ids) + [False] * len(caption_ids)
        embeds_cmp_mask = [True] * num_clips
        embeds_gen_mask = [False] * num_clips
    else:
        # 2 for bos and eos, 2 for video boi and video eoi
        extra_length = 2 + 2 + (2 + num_video_tokens) * num_clips
        if len(caption_ids) + extra_length > max_length:
            caption_ids = caption_ids[:(max_length-extra_length)]

        input_ids = input_ids + caption_ids + video_ids
        if instruction_prompt is None:
            labels = labels + [-100] * len(caption_ids) + [-100] * len(video_ids)
        else:
            labels = labels + [-100] * len(caption_ids) + [video_ids[0]] + [-100] * (len(video_ids) - 2) + [video_ids[-1]]
        ids_gen_mask = ids_gen_mask + [False] * len(caption_ids) + [False] + ([False] + [True] * num_video_tokens + [False]) * num_clips + [False]
        ids_cmp_mask = ids_cmp_mask + [False] * len(caption_ids) + [False] * len(video_ids)
        embeds_cmp_mask = [False] * num_clips
        embeds_gen_mask = [True] * num_clips

    input_ids = input_ids + [tokenizer.eos_token_id]
    labels = labels + [tokenizer.eos_token_id]
    ids_gen_mask = ids_gen_mask + [False]
    ids_cmp_mask = ids_cmp_mask + [False]

    if video_first or not bilateral_attention:
        attention_mask = [1] * len(input_ids)
    else:
        ## -1 for bilateral_attention of video query
        attention_mask = [1] + [1] * len(caption_ids) + [1] + ([1] + [-1] * num_video_tokens + [1]) * num_clips + [1] + [1]

    if len(input_ids) > max_length:
        if video_first:
            print("input length exceeds", len(input_ids))
            input_ids = input_ids[:max_length]
            labels = labels[:max_length]
            ids_gen_mask = ids_gen_mask[:max_length]
            ids_cmp_mask = ids_cmp_mask[:max_length]
            attention_mask = attention_mask[:max_length]
        else:
            print("input length exceeds", len(input_ids))
            return {}
    else:
        padding_length = max_length - len(input_ids)
        input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
        labels = labels + [-100] * padding_length
        ids_gen_mask = ids_gen_mask + [False] * padding_length
        ids_cmp_mask = ids_cmp_mask + [False] * padding_length
        attention_mask = attention_mask + [0] * padding_length

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    ids_cmp_mask = torch.tensor(ids_cmp_mask, dtype=torch.bool)
    ids_gen_mask = torch.tensor(ids_gen_mask, dtype=torch.bool)

    if stride != 0:
        max_vid_length = int(max_vid_length / stride + 1)

    if video_first:
        clip_padding_length = num_clips_require - num_clips
        if clip_padding_length != 0:
            frames = torch.cat(
                [frames, torch.zeros(clip_padding_length*max_vid_length, *frames.shape[1:])], dim=0
            )
            embeds_cmp_mask = embeds_cmp_mask + [False] * clip_padding_length
            embeds_gen_mask = embeds_gen_mask + [False] * clip_padding_length

    embeds_gen_mask = torch.tensor(embeds_gen_mask)
    embeds_cmp_mask = torch.tensor(embeds_cmp_mask)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'ids_gen_mask': ids_gen_mask,
        'ids_cmp_mask': ids_cmp_mask,
        'embeds_gen_mask': embeds_gen_mask,
        'embeds_cmp_mask': embeds_cmp_mask,
        "frames": frames,
    }


def build_multi_datapipes(datapipes, tokenizer=None, image_transform=None, sample_weights=None):
    if sample_weights is None:
        sample_weights = [1] * len(datapipes)
    else:
        assert len(sample_weights) == len(datapipes)

    datapipes = [
        hydra.utils.instantiate(datapipe, tokenizer=tokenizer, image_transform=image_transform) for datapipe in datapipes
    ]

    datasets_to_weights_dict = {}
    for dataset, sample_weight in zip(datapipes, sample_weights):
        datasets_to_weights_dict[dataset] = sample_weight
    datapipe = dp.iter.SampleMultiplexer(datasets_to_weights_dict, seed=142 + dist.get_rank())

    return datapipe


def build_video_qa_jsonl_datapipes(
    data_dir,
    image_dir,
    tokenizer=None,
    max_length=77,
    num_frames=4,
    num_clips=1,
    fps=4,
    stride=0,
    padding_clips=False,
    num_video_tokens=64,
    instruction_prompt=None,
    batch_size=None,
    image_transform=None,
    min_aspect_ratio=0.666,
    cycle_count=None,
    min_resolution=400,
):
    """
    datapipe of caption dataset (such as CC3M, LAION...) with webdataset format
    """

    decode_partial = functools.partial(
        decode_video_qa_jsonl_data_decord,
        image_dir=image_dir,
        tokenizer=tokenizer,
        image_transform=image_transform,
        max_length=max_length,
        num_frames=num_frames,
        num_clips=num_clips,
        fps=fps,
        stride=stride,
        padding_clips=padding_clips,
        num_video_tokens=num_video_tokens,
        instruction_prompt=instruction_prompt,
        min_aspect_ratio=min_aspect_ratio,
        min_resolution=min_resolution,
    )

    if isinstance(data_dir, str):
        data_dir = list(braceexpand(data_dir))
    datapipe = dp.iter.FileLister(root=data_dir, masks="*.jsonl", recursive=True)
    datapipe = datapipe.cycle(count=cycle_count)
    datapipe = datapipe.open_files(mode="r")
    datapipe = datapipe.parse_jsonl_files()
    datapipe = datapipe.shuffle()
    datapipe = datapipe.sharding_filter()
    datapipe = datapipe.map(decode_partial)
    datapipe = datapipe.filter(lambda item: "frames" in item and "input_ids" in item)

    if batch_size is not None:
        datapipe = datapipe.batch(batch_size)
        datapipe = datapipe.collate()

    return datapipe


def decode_video_qa_jsonl_data_decord(
    item,
    image_dir,
    tokenizer,
    image_transform=None,
    max_length=128,
    num_frames=4,
    num_clips=1,
    num_video_tokens=64,
    fps=4,
    stride=0,
    padding_clips=False,
    instruction_prompt=None,
    min_aspect_ratio=0.666,
    min_resolution=400,
):
    key, value = item

    if 'video' in value.keys():
        video_path = value['video']
        if 'questions' in value.keys():
            questions = value['questions']
            answers = value['answers']
        else:
            questions = [value['question']]
            answers = [value['answer']]
        is_video = True
    else:
        video_path = value['image']
        if 'questions' in value.keys():
            questions = value['questions']
            answers = value['answers']
        else:
            data = value['data']
            questions = data[::2]
            answers = data[1::2]

        is_video = False

    max_vid_length = num_frames
    num_clips_require = num_clips

    video_path = os.path.join(image_dir, video_path)

    if not os.path.exists(video_path):
        print("video file does not exists", video_path)
        return {}

    if is_video:
        try:
            if '.webm' in video_path:
                cap = cv2.VideoCapture(video_path)
                video_fps = cap.get(cv2.CAP_PROP_FPS)
                video_length = 0
                frames_all = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    video_length += 1
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames_all.append(frame)
                frames_all = np.array(frames_all).astype(np.uint8)

            elif '.gif' in video_path:
                gif = imageio.get_reader(video_path)
                video_fps = 24
                video_length = len(gif)

                frames_all = []
                for index, frame in enumerate(gif):
                    frames_all.append(frame[:, :, :3])
                frames_all = np.array(frames_all).astype(np.uint8)

            else:
                vr = VideoReader(video_path, ctx=cpu(0))
                video_fps = vr.get_avg_fps()
                video_length = len(vr)

            sampled_indices_all = []

            sample_interval = video_fps / fps
            total_frames = video_length // sample_interval

            if total_frames >= num_clips * max_vid_length or not padding_clips:
                clip_length = video_length // num_clips
                sample_interval = video_fps / fps
                num_frames_per_clip = int(min(max_vid_length, clip_length // sample_interval))

                if num_frames_per_clip != max_vid_length:
                    sample_interval = clip_length / max_vid_length
                    num_frames_per_clip = max_vid_length

                for i in range(num_clips):
                    start = np.random.uniform(i * clip_length, (i + 1) * clip_length - num_frames_per_clip * sample_interval)
                    sampled_indices = np.linspace(
                        start, start + num_frames_per_clip * sample_interval, num_frames_per_clip, endpoint=False
                    )
                    sampled_indices = sampled_indices.astype(np.int32)

                    if stride != 0.0:
                        sampled_indices_ori = sampled_indices
                        sampled_indices = sampled_indices[0::stride]
                        sampled_indices = np.append(sampled_indices, sampled_indices_ori[-1])

                    assert (sampled_indices < video_length).all(), "sample indices out of range"
                    sampled_indices_all.extend(sampled_indices)
            else:
                start = np.random.uniform(0, sample_interval)
                num_clips_real = 0
                while start < video_length:
                    end = start + max_vid_length * sample_interval

                    if end > video_length:
                        end = video_length

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
                num_clips = num_clips_real

            if num_clips == 0:
                start = 0
                end = video_length
                sampled_indices = np.linspace(
                    start, end, max_vid_length, endpoint=False
                )
                sampled_indices = sampled_indices.astype(np.int32)
                if stride != 0.0:
                    sampled_indices_ori = sampled_indices
                    sampled_indices = sampled_indices[0::stride]
                    sampled_indices = np.append(sampled_indices, sampled_indices_ori[-1])
                sampled_indices_all.extend(sampled_indices)
                num_clips = 1

            if '.webm' in video_path or '.gif' in video_path:
                frames = frames_all[sampled_indices_all]
            else:
                frames = vr.get_batch(sampled_indices_all).asnumpy().astype(np.uint8)


        except Exception as e:
            print("Error in reading video", video_path)
            print("Exception:", e)
            return {}

        frames = [Image.fromarray(frame) for frame in frames]
        frames = [image_transform(frame) for frame in frames]
        frames = torch.stack(frames, dim=0)

        if stride == 0.0:
            required_frames = num_frames * num_clips
        else:
            required_frames = (num_frames / stride + 1) * num_clips
        if len(frames) < required_frames:
            print("not enough frames")
            return {}
    else:
        try:
            image = Image.open(video_path).convert("RGB")
            image = image_transform(image)
            frames = image.unsqueeze(0).repeat(num_frames*num_clips, 1, 1, 1)
        except:
            print("error in reading image", video_path)
            return {}

    if is_video:
        video_tokens = VID_BOI_TOKEN + (BOI_TOKEN + ''.join([FRAME_TOKEN.format(int(item)) for item in range(num_video_tokens)]) + EOI_TOKEN) * num_clips + VID_EOI_TOKEN
    else:
        video_tokens = IMG_BOI_TOKEN + (BOI_TOKEN + ''.join([FRAME_TOKEN.format(int(item)) for item in range(num_video_tokens)]) + EOI_TOKEN) * num_clips + IMG_EOI_TOKEN

    input_ids = [tokenizer.bos_token_id]
    labels = [-100]

    for indx in range(len(questions)):
        question = questions[indx].replace("<video>\n", "")
        answer = answers[indx]

        if indx == 0:
            instruction = instruction_prompt.format_map({'instruction': video_tokens + question})
        else:
            instruction = instruction_prompt.format_map({'instruction': question})

        if indx != 0:
            instruction = '\n' + instruction

        instruction_ids = tokenizer.encode(instruction, add_special_tokens=False)
        answer_ids = tokenizer.encode(answer, add_special_tokens=False)
        input_ids = input_ids + instruction_ids + answer_ids
        labels = labels + [-100] * len(instruction_ids) + answer_ids

    input_ids = input_ids + [tokenizer.eos_token_id]
    labels = labels + [tokenizer.eos_token_id]
    ids_cmp_mask = [False] * len(input_ids)
    ids_gen_mask = [False] * len(input_ids)
    embeds_cmp_mask = [True] * num_clips
    embeds_gen_mask = [False] * num_clips
    attention_mask = [1] * len(input_ids)

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
        ids_gen_mask = ids_gen_mask[:max_length]
        ids_cmp_mask = ids_cmp_mask[:max_length]
        attention_mask = attention_mask[:max_length]
    else:
        padding_length = max_length - len(input_ids)
        input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
        labels = labels + [-100] * padding_length
        ids_gen_mask = ids_gen_mask + [False] * padding_length
        ids_cmp_mask = ids_cmp_mask + [False] * padding_length
        attention_mask = attention_mask + [0] * padding_length

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    ids_cmp_mask = torch.tensor(ids_cmp_mask, dtype=torch.bool)
    ids_gen_mask = torch.tensor(ids_gen_mask, dtype=torch.bool)

    boi_token_id = tokenizer.encode(BOI_TOKEN, add_special_tokens=False)[0]
    eoi_token_id = tokenizer.encode(EOI_TOKEN, add_special_tokens=False)[0]

    boi_idx = torch.where(input_ids == boi_token_id)[0].tolist()
    eoi_idx = torch.where(input_ids == eoi_token_id)[0].tolist()

    for i in range(len(boi_idx)):
        ids_cmp_mask[boi_idx[i] + 1:eoi_idx[i]] = True

    if stride != 0:
        max_vid_length = int(max_vid_length / stride + 1)

    clip_padding_length = num_clips_require - num_clips
    if clip_padding_length != 0:
        frames = torch.cat(
            [frames, torch.zeros(clip_padding_length*max_vid_length, *frames.shape[1:])], dim=0
        )
        embeds_cmp_mask = embeds_cmp_mask + [False] * clip_padding_length
        embeds_gen_mask = embeds_gen_mask + [False] * clip_padding_length

    embeds_gen_mask = torch.tensor(embeds_gen_mask)
    embeds_cmp_mask = torch.tensor(embeds_cmp_mask)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'ids_gen_mask': ids_gen_mask,
        'ids_cmp_mask': ids_cmp_mask,
        'embeds_gen_mask': embeds_gen_mask,
        'embeds_cmp_mask': embeds_cmp_mask,
        "frames": frames,
    }
