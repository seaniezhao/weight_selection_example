from io import BytesIO
from urllib.request import urlopen
import librosa
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration, AutoModelForCausalLM
import torch
# model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B" ,trust_remote_code=True)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B" ,trust_remote_code=True)
# print(model.config)

prompt1 = "<|audio_bos|><|AUDIO|><|audio_eos|>Generate the caption in English:"
url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Audio/glass-breaking-151256.mp3"

prompt2 = "<|audio_bos|><|AUDIO|><|audio_eos|>What's this?"
 
audio, sr = librosa.load(BytesIO(urlopen(url).read()), sr=processor.feature_extractor.sampling_rate)
inputs = processor(text=[prompt1, prompt2], audios=[audio, audio], padding=True, padding_side='right', return_tensors="pt")

for k, v in inputs.items():
    print(k, v.shape, v)


# print(inputs["feature_attention_mask"][0].sum())
# print(model.audio_tower.config)

# generated_ids = model.generate(**inputs, max_length=256)
# generated_ids = generated_ids[:, inputs.input_ids.size(1):]


# response = processor.batch_decode(generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]

# print(response)