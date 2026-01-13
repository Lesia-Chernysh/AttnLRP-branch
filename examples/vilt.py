import torch
import requests
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from lxt.utils import pdf_heatmap

# just a helper function to draw the text attention later
import html
import matplotlib
from matplotlib.colors import Normalize, to_hex

def colorize(tokens:list[str], weights:list[float], target:str=None):
    """returns a html heatmap representation of the tokens colored by weights.
        tokens and weights should match in lenghth
        target is optionally the token the weights were calculated for
    """
    template = '<span style="color: black; background-color: {}">{}</span>'
    colors = map(to_hex, matplotlib.colormaps['bwr'](Normalize(vmin=-1, vmax=1)(weights)))
    colored_output = ''
    for token, color in zip(tokens, colors):
        colored_output += template.format(color, '&nbsp'+html.escape(token)+'&nbsp')
    if target:
        colored_output += template.format('lime', '&nbsp'+html.escape(target)+'&nbsp')
    return colored_output

# load original ViLT processor
from transformers import ViltProcessor
# load adapted ViLT model and rule composite from attnLRP (lxt library)
from lxt.explicit.models.vilt import ViltForQuestionAnswering, attnlrp as lrp_rule_composite

# init processor (combines image patching and text tokenizers)
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# still use original weights checkpoint from huggingface (from_pretrained)
adapted_model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# there is multiple ways to modify the model to compute relevance
# here part of the model was directly modified in code, part handled by automatic lrp rules, which we now apply
lrp_rule_composite.register(adapted_model)

# prepare an input sample, for illustration just an image from MSCOCO and a made up question
image = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)
question = "Are those cats or dogs?"
inputs = processor(image, question, return_tensors="pt")
txt_tokens = processor.tokenizer.batch_decode(inputs.input_ids[0])

# attnLRP modifies the backward pass gradient computation to instead compute relevance
# thus enable gradient for all inputs which should receive a relevance map
inputs.pixel_values.requires_grad = True

# the token -> embedding step is a lookup table which is non-differentiable
# thus we compute the relevance for the embeddings instead and sum it up later (all relevance on the embedding is for the token)
# huggingface allows to pass the embeddings directly instead of token indices
inputs_embeds = adapted_model.get_input_embeddings()(inputs.input_ids)
inputs_embeds.requires_grad = True

# forward pass
outputs = adapted_model(inputs_embeds=inputs_embeds, 
             pixel_values=inputs.pixel_values, 
             token_type_ids=inputs.token_type_ids, 
             attention_mask=inputs.attention_mask, 
             pixel_mask=inputs.pixel_mask)

# the max logit index marks the models' answer
logits = outputs.logits
max_logit, predicted_class_idx = logits.max(-1)

# we can look up the answer in the label list
print("Question:", txt_tokens)
print("Predicted answer:", adapted_model.config.id2label[predicted_class_idx.item()])

# to attribute the logit's value back to the inputs, we run the backward pass through it
max_logit.backward()

# since attnLRP modified the gradient computation we can now read the relevance off the .grad attribute
img_attention_map = inputs.pixel_values.grad[0].sum(0) # sum over RGB channels for per-pixel relevance
txt_attention_map = inputs_embeds.grad[0].sum(-1) # sum over embedding dim for per-token relevance

# normalize the maps to [-1, 1]
img_attention_map = img_attention_map / abs(img_attention_map).max()
txt_attention_map = txt_attention_map / abs(txt_attention_map).max()

plt.imshow(img_attention_map, cmap='bwr', vmin=-1, vmax=1)
plt.savefig('vilt_img_heatmap.png')
Path('vilt_txt_heatmap.html').write_text(colorize(txt_tokens, txt_attention_map, adapted_model.config.id2label[predicted_class_idx.item()]))