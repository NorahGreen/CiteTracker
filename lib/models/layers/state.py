import os
import os.path as osp
from collections import OrderedDict

import torch
import torch.nn as nn
import pickle
from functools import partial

import CiteTracker.lib.models.layers.clip as clip
from CiteTracker.lib.models.layers.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


def load_checkpoint(fpath):
    r"""Load checkpoint.

    ``UnicodeDecodeError`` can be well handled, which means
    python2-saved files can be read from python3.

    Args:
        fpath (str): path to checkpoint.

    Returns:
        dict

    Examples::
        # >>> fpath = 'log/my_model/model.pth.tar-10'
        # >>> checkpoint = load_checkpoint(fpath)
    """
    if fpath is None:
        raise ValueError("File path is None")

    if not osp.exists(fpath):
        raise FileNotFoundError('File is not found at "{}"'.format(fpath))

    map_location = None if torch.cuda.is_available() else "cpu"

    try:
        checkpoint = torch.load(fpath, map_location=map_location)

    except UnicodeDecodeError:
        pickle.load = partial(pickle.load, encoding="latin1")
        pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
        checkpoint = torch.load(
            fpath, pickle_module=pickle, map_location=map_location
        )

    except Exception:
        print('Unable to load checkpoint from "{}"'.format(fpath))
        raise

    return checkpoint


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model, device):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = 4
        ctx_init = "a photo of a"
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init).to(device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)

        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
        ]))

        self.meta_net.half()

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self, im_features):
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx  # (n_ctx, ctx_dim)
        bias = self.meta_net(im_features)  # (batch, ctx_dim)
        bias = bias.unsqueeze(1)  # (batch, 1, ctx_dim)
        ctx = ctx.unsqueeze(0)  # (1, n_ctx, ctx_dim)
        ctx_shifted = ctx + bias  # (batch, n_ctx, ctx_dim)

        # Use instance-conditioned context tokens for all classes
        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(ctx_i, prefix, suffix)  # (n_cls, n_tkn, ctx_dim)
            prompts.append(pts_i)
        prompts = torch.stack(prompts)

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, classnames, clip_model, preprocess, device):
        super().__init__()
        self.prompt_learner = PromptLearner(classnames, clip_model, device)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.preprocess = preprocess
        self.device = device

    def forward(self, image_features, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        # image = image.cuda()
        # image_features = self.image_encoder(image.type(self.dtype))
        # image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        prompts = self.prompt_learner(image_features).to(self.device)

        logits = []
        for pts_i, imf_i in zip(prompts, image_features):
            text_features = self.text_encoder(pts_i, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            l_i = logit_scale * imf_i @ text_features.t()
            logits.append(l_i)
        logits = torch.stack(logits)

        # if self.prompt_learner.training:
        #     return F.cross_entropy(logits, label)
#         result = 0
#         if (logits[0][0] < logits[0][1]):
#             result = 1
#         print(result)
        return logits


class CoCoOp(nn.Module):

    def __init__(self, classnames, clip_model, preprocess):
        super().__init__()
        self._models = OrderedDict()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.build_model(classnames, clip_model, preprocess)

    def get_model_names(self, names=None):
        names_real = list(self._models.keys())
        if names is not None:
            names = tolist_if_not(names)
            for name in names:
                assert name in names_real
            return names
        else:
            return names_real

    def build_model(self, classnames, clip_model, preprocess):
#         classnames = ['is', 'is not']
#         classnames = ['black','blue','brown', 'gray', 'green', 'orange', 'pink', 'red', 'tan', 'violet', 'white', 'yellow']
#         classnames = ['black object','blue object','brown object', 'gray object', 'green object', 'orange object', 'pink object', 'red object', 'tan object', 'violet object', 'white object', 'yellow object']
        # classnames = ['multicolored object', 'single-colored object', 'two-colored object']
        # classnames = ['cement object', 'ceramic object', 'glass object', 'leather object', 'metal object', 'paper object', 'polymers object', 'stone object', 'textile object', 'wooden object']
        # classnames = ['rough object', 'smooth object', 'soft object']

        print(f"Loading CLIP backbone: ViT-B/32")
        #         clip_model = load_clip_to_cpu()
        # clip_model, preprocess = clip.load('/home/yuqing/pro/CiteTracker/lib/models/layers/clip/ViT-B-32.pt', self.device)

        print("Building custom CLIP")
        self.model = CustomCLIP(classnames, clip_model, preprocess, self.device)

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        self.model.to(self.device)

        #         self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        #         self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner)

#         device_count = torch.cuda.device_count()
#         if device_count > 1:
#             print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
#             self.model = nn.DataParallel(self.model)

    def parse_batch_test(self, batch):
        input = batch["img"]
        label = batch["label"]
        impath = batch["impath"]

        input = input.to(self.device)
        label = label.to(self.device)

        return input, label, impath

    def register_model(self, name="model", model=None, optim=None, sched=None):
        if self.__dict__.get("_models") is None:
            raise AttributeError(
                "Cannot assign model before super().__init__() call"
            )

        assert name not in self._models, "Found duplicate model names"

        self._models[name] = model

    #         self._optims[name] = optim
    #         self._scheds[name] = sched

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
            print('done')


class State_encoder():
    def __init__(self, classnames, attribute, clip_model, preprocess):
        super().__init__()
        self.trainer = CoCoOp(classnames, clip_model, preprocess)
        self.classnames = classnames
        current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
        # pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
        # directory = current_dir+'/' + attribute
        directory = os.path.join(current_dir, attribute)
        self.trainer.load_model(directory, 10)

    def forward(self, image):
        result = self.trainer.model(image)
        index = torch.argmax(result[0])

        # att = self.classnames[index]
        # att = att.split(' ')
        # return att[0]
        return result, index


# device = "cuda" if torch.cuda.is_available() else "cpu"
# clip_model, preprocess = clip.load('ViT-B/32', device)
# classnames = ['black','blue','brown', 'gray', 'green', 'orange', 'pink', 'red', 'tan', 'violet', 'white', 'yellow']
# net = State_encoder(classnames, 'color')
# image = PIL.Image.open('/home/yuqing/projects/res/CiteTracker/test.jpg')
# image_input = preprocess(image).unsqueeze(0).to(device)
# with torch.no_grad():
#     image_features = clip_model.encode_image(image_input)
#     image_features /= image_features.norm(dim=-1, keepdim=True)
#     print(net.forward(image_features))

