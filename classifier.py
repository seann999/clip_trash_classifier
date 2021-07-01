import numpy as np
import torch
import clip
from PIL import Image


cats = [
    ('combustible', [
        'raw garbage', 'tissue', 'paper towels', 'a plastic bag', 'a plastic egg carton', 'styrofoam',
        'a CD', 'a VCR', 'a plastic box', 'a snack bag', 'rubber', 'leather', 'a silica pack',
        'food', 'a pencil', 'rubber gloves', 'a bath towel', 'a toothbrush', 'a ball-point pen', 'toilet paper',
        'a wallet'
    ]),
    ('non-combustible', [
        'aluminum foil', 'an umbrella', 'scissors', 'cutlery', 'glass', 'fleurolecent',
        'a battery', 'a lighter', 'a USB stick', 'ceramic', 'a glass cup', 'a hair dryer',
        'a spoon', 'a light bulb', 'a remote', 'a screwdriver', 'a nail clipper',
    ]),
    ('recyclable', [
        'a newspaper', 'a magazine', 'cardboard', 'a glass bottle', 'a glass jar', 'an aluminum can', 'a plastic bottle',
        'a book', 'a plastic pill bottle', 'a cell phone', 'a nintendo switch',
    ]),
    ('large', [
        'keyboard', 'a speaker'
    ]),
    ('misc', [
        'nothing', 'a TV', 'a refrigerator', 'a human', 'a laptop', 'a monitor',
    ])
]

objects = [x for c in cats for x in c[1]]
obj2cat = {x: c[0] for c in cats for x in c[1]}
prompts = ['a photo of {}'.format(obj) for obj in objects]


class Classifier:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.text = clip.tokenize(prompts).to(self.device)

    def classify(self, image):
        image = self.preprocess(Image.fromarray(image)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits_per_image, logits_per_text = self.model(image, self.text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

        obj = objects[probs.argmax()]
        cat = obj2cat[obj]
        per = int(np.round(probs.max() * 100.0))

        if per < 30:
            return '???', ''
        else:
            return cat, obj

