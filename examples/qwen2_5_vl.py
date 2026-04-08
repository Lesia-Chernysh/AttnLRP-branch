import html
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import requests
import torch
from matplotlib.colors import Normalize, to_hex
from PIL import Image
from torchvision.transforms.v2.functional import pil_to_tensor
from transformers import AutoProcessor
from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl

from lxt.efficient import monkey_patch, monkey_patch_zennit
from zennit.composites import LayerMapComposite
import zennit.rules as z_rules


def colorize(tokens: list[str], weights: list[float], target: str | None = None):
    """Return a simple html heatmap for token relevance."""
    template = '<span style="color: black; background-color: {}">{}</span>'
    colors = map(to_hex, matplotlib.colormaps["bwr"](Normalize(vmin=-1, vmax=1)(weights)))
    colored_output = ""
    for token, color in zip(tokens, colors):
        colored_output += template.format(color, "&nbsp" + html.escape(token) + "&nbsp")
    if target:
        colored_output += template.format("lime", "&nbsp" + html.escape(target) + "&nbsp")
    return colored_output


device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_id = "Qwen/Qwen2.5-VL-7B-Instruct"

# patch Qwen2.5-VL for attnLRP behavior in backward.
monkey_patch(modeling_qwen2_5_vl, verbose=True)
monkey_patch_zennit(verbose=True)

dtype = torch.bfloat16 if "cuda" in device else torch.float32
model = modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    device_map=device,
    dtype=dtype,
)
processor = AutoProcessor.from_pretrained(model_id)

for param in model.parameters():
    param.requires_grad = False

# register explicit rules on the visual backbone (same pattern as notebook usage).
composite = LayerMapComposite([
    (torch.nn.Conv3d, z_rules.Gamma(100.0)),
])
composite.register(model.visual)

model.eval()

# prepare one sample.
image = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw).convert("RGB")
question = "How many cats are in the image?"

# since the qwen processor patches the image, we compute relevance to the image tensor
image_tensor = pil_to_tensor(image).float().to(device).requires_grad_(True)

prompt = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "Answer ONLY to the question.\nDo not explain.\nDo not add punctuation."}],
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image_tensor},
            {"type": "text", "text": question},
        ],
    },
]

inputs = processor.apply_chat_template(
    prompt,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
).to(device)

# we generate the full multi-token output
output_ids = model.generate(**inputs, max_new_tokens=20)

# we recompute the last forward pass to compute the attributions (generate does not return the logits)
# we compute relevance to the input embeddings because the embedding layer is non-differentiable
input_ids = output_ids[:, :-1]
input_embeds = model.get_input_embeddings()(input_ids).requires_grad_(True)
logits = model(
    inputs_embeds=input_embeds,
    pixel_values=inputs.pixel_values,
    image_grid_thw=inputs.image_grid_thw,
    attention_mask=torch.ones_like(input_ids),
).logits

# attribute union of generated token logits to the input embeddings
logit_mask = torch.full_like(logits, 0)
answer_length = input_ids.shape[-1] - inputs.input_ids.shape[-1]
for i, token_id in enumerate(output_ids[0, -answer_length:]):
    logit_mask[:, -answer_length+i-1, token_id] = 1
# logit_mask[:, -1, model.config.eos_token_id] = 1 # we could optionally attribute the eos token as well to see why the model decided to stop generating
logits.backward(logits.detach()*logit_mask)

# gradient now contains relevance due to attnLRP patching.
# efficient lxt modifies the gradient and multiplies it with the input to get the relevance
img_attention_map = (image_tensor.detach() * image_tensor.grad.detach()).sum(0)
txt_attention_map = (input_embeds.detach()[0] * input_embeds.grad.detach()[0]).sum(-1)

img_attention_map = img_attention_map / img_attention_map.abs().max().clamp_min(1e-8)
txt_attention_map = txt_attention_map / txt_attention_map.abs().max().clamp_min(1e-8)

txt_tokens = [processor.tokenizer.decode(i) for i in input_ids[0]]

plt.imshow(img_attention_map.detach().cpu().float().numpy(), cmap="bwr", vmin=-1, vmax=1)
plt.title(f"Qwen2.5-VL relevance for: {question}")
plt.axis("off")
plt.savefig("qwen2_5_vl_img_heatmap.png", bbox_inches="tight")

Path("qwen2_5_vl_txt_heatmap.html").write_text(
    colorize(txt_tokens, txt_attention_map.detach().cpu().float().numpy())
)

composite.remove()
