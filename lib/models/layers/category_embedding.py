import torchvision.transforms as transforms

import torch
import lib.models.layers.clip.clip as clip
from PIL import Image
from torch import nn
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from lib.models.layers.state import State_encoder


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    BICUBIC = Image.BICUBIC
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


class Category_embedding(nn.Module):

    def __init__(self):
        super().__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.clip_model, self.preprocess = clip.load('/home/yuqing/test2/CiteTracker/lib/models/layers/clip/ViT-B-32.pt', self.device)
        self.clip_model, self.preprocess = clip.load('ViT-B/32', self.device)
        color = ['black object', 'blue object', 'brown object', 'gray object', 'green object', 'orange object',
                 'pink object', 'red object', 'tan object', 'violet object', 'white object', 'yellow object']
        self.color = State_encoder(color, 'color', self.clip_model, self.preprocess)
        texture = ['rough object', 'smooth object', 'soft object']
        self.texture = State_encoder(texture, 'texture', self.clip_model, self.preprocess)
        material = ['cement object', 'ceramic object', 'glass object', 'leather object', 'metal object', 'paper object',
                    'polymers object', 'stone object', 'textile object', 'wooden object']
        self.material = State_encoder(material, 'material', self.clip_model, self.preprocess)

        self.color_list = torch.cat([clip.tokenize(f"a photo of {c}") for c in color]).to(self.device)
        self.texture_list = torch.cat([clip.tokenize(f"a photo of {c}") for c in texture]).to(self.device)
        self.material_list = torch.cat([clip.tokenize(f"a photo of {c}") for c in material]).to(self.device)

        self.convert_vector = torch.nn.Linear(512, 768).to(self.device)
        self.softmax = nn.Softmax(dim=-1)
        # self.convert_vector2 = torch.nn.Linear(512, 16).to(self.device)

        self.label = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                      'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                      'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
                      'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
                      'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
                      'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
                      'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
                      'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                      'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                      'teddy bear', 'hair drier', 'toothbrush']
        self.text_inputs = torch.cat([clip.tokenize(f"a photo of {c}") for c in self.label]).to(self.device)
        with torch.no_grad():
            self.text_features = self.clip_model.encode_text(self.text_inputs)
            self.color_list = self.clip_model.encode_text(self.color_list)
            self.texture_list = self.clip_model.encode_text(self.texture_list)
            self.material_list = self.clip_model.encode_text(self.material_list)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)
            self.color_list /= self.color_list.norm(dim=-1, keepdim=True)
            self.texture_list /= self.texture_list.norm(dim=-1, keepdim=True)
            self.material_list /= self.material_list.norm(dim=-1, keepdim=True)
            self.text = self.convert_vector(self.text_features.float()).softmax(dim=-1)
            self.color_list = self.convert_vector(self.color_list.float()).softmax(dim=-1)
            self.texture_list = self.convert_vector(self.texture_list.float()).softmax(dim=-1)
            self.material_list = self.convert_vector(self.material_list.float()).softmax(dim=-1)

        self.tem_image_features = None
        self.tem_similarity = None
        self.tem_color_index = None
        self.tem_material_index = None
        self.tem_texture_index = None
        self.tem_color_sim = None
        self.tem_material_sim = None
        self.tem_texture_sim = None
        self.indices = None


    def forward(self, template, search):
        # image-encoder
        if self.training is True or type(template)==torch.Tensor:
            batch_size = len(template)
        else:
            batch_size = 1
        class_des = torch.Tensor().cuda()
        color_des = torch.Tensor().cuda()
        material_des = torch.Tensor().cuda()
        texture_des = torch.Tensor().cuda()
        attention_des = torch.Tensor().cuda()
        for i in range(batch_size):
            if self.training:
                ToPILImage = transforms.ToPILImage()
                tem_img_PIL = ToPILImage(template[i])
                tem_image_input = self.preprocess(tem_img_PIL).unsqueeze(0).to(self.device)
                sea_img_PIL = ToPILImage(search[i])
                sea_image_input = self.preprocess(sea_img_PIL).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    tem_image_features = self.clip_model.encode_image(tem_image_input)
                    tem_image_features /= tem_image_features.norm(dim=-1, keepdim=True)
                    tem_similarity = (100.0 * tem_image_features @ self.text_features.T.half()).softmax(dim=-1)
                    values, indices = tem_similarity[0].topk(1)

                    tem_color_sim, tem_color_index = self.color.forward(tem_image_features)
                    tem_material_sim, tem_material_index = self.material.forward(tem_image_features)
                    tem_texture_sim, tem_texture_index = self.texture.forward(tem_image_features)
                    sea_image_features = self.clip_model.encode_image(sea_image_input)
                    sea_image_features /= sea_image_features.norm(dim=-1, keepdim=True)
                    # sea_similarity = (100.0 * sea_image_features @ self.text_features.T.half()).softmax(dim=-1)

                    sea_color_sim = self.color.forward(sea_image_features)
                    sea_material_sim = self.material.forward(sea_image_features)
                    sea_texture_sim = self.texture.forward(sea_image_features)

                    color_attention = -torch.norm(tem_color_sim[0] - sea_color_sim[0])
                    material_attention = -torch.norm(tem_material_sim[0] - sea_material_sim[0])
                    texture_attention = -torch.norm(tem_texture_sim[0] - sea_texture_sim[0])

                    attention = self.softmax(
                        torch.Tensor([color_attention, material_attention, texture_attention])).unsqueeze(0)

                    class_label = self.text[indices]
                    color_label = self.color_list[tem_color_index].unsqueeze(0)
                    material_label = self.material_list[tem_material_index].unsqueeze(0)
                    texture_label = self.texture_list[tem_texture_index].unsqueeze(0)
            else:
                if self.tem_image_features is None:
                    with torch.no_grad():
                        tem_image_input = self.preprocess(template).unsqueeze(0).to(self.device)
                        self.tem_image_features = self.clip_model.encode_image(tem_image_input)
                        self.tem_image_features /= self.tem_image_features.norm(dim=-1, keepdim=True)
                        self.tem_similarity = (100.0 * self.tem_image_features @ self.text_features.T.half()).softmax(dim=-1)
                        values, self.indices = self.tem_similarity[0].topk(1)
                        self.tem_color_sim, self.tem_color_index = self.color.forward(self.tem_image_features)
                        self.tem_material_sim, self.tem_material_index = self.material.forward(self.tem_image_features)
                        self.tem_texture_sim, self.tem_texture_index = self.texture.forward(self.tem_image_features)

                sea_image_input = self.preprocess(search).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    sea_image_features = self.clip_model.encode_image(sea_image_input)
                    sea_image_features /= sea_image_features.norm(dim=-1, keepdim=True)
                    # sea_similarity = (100.0 * sea_image_features @ self.text_features.T.half()).softmax(dim=-1)

                    sea_color_sim =  self.color.forward(sea_image_features)
                    sea_material_sim = self.material.forward(sea_image_features)
                    sea_texture_sim =  self.texture.forward(sea_image_features)

                    color_attention = -torch.norm(self.tem_color_sim[0] - sea_color_sim[0])
                    material_attention = -torch.norm(self.tem_material_sim[0] - sea_material_sim[0])
                    texture_attention = -torch.norm(self.tem_texture_sim[0] - sea_texture_sim[0])

                    attention = self.softmax(torch.Tensor([color_attention, material_attention, texture_attention])).unsqueeze(0)

                    class_label = self.text[self.indices]
                    color_label = self.color_list[self.tem_color_index].unsqueeze(0)
                    material_label = self.material_list[self.tem_material_index].unsqueeze(0)
                    texture_label = self.texture_list[self.tem_texture_index].unsqueeze(0)


            class_des = torch.cat([class_des, class_label], dim=0)
            color_des = torch.cat([color_des, color_label], dim=0)
            material_des = torch.cat([material_des, material_label], dim=0)
            texture_des = torch.cat([texture_des, texture_label], dim=0)
            attention_des = torch.cat([attention_des, attention.cuda()], dim=0)


            
        return class_des.resize(batch_size, 1, 768, 1 ,1 ), color_des.resize(batch_size,1, 768, 1, 1), material_des.resize(
            batch_size, 1, 768, 1, 1),  texture_des.resize(batch_size, 1, 768, 1, 1), attention_des