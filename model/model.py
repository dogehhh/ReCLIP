from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
import clip
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", nn.GELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=True, attn_mask=None)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln_1(x), self.ln_1(x), self.ln_1(x), need_weights=True, attn_mask=None)[0]
        # x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        return x

    def _initialize_weights(self, clip_model, i):
        self.ln_1 = clip_model.visual.transformer.resblocks[i].ln_1
        self.ln_1.eps = 1e-06
        self.attn = clip_model.visual.transformer.resblocks[i].attn.to(torch.float32)
        self.attn.batch_first = True
        self.mlp = clip_model.visual.transformer.resblocks[i].mlp.to(torch.float32)
        # self.mlp[1] = nn.GELU()
        self.ln_2 = clip_model.visual.transformer.resblocks[i].ln_2
        self.ln_2.eps = 1e-06
        for p in self.parameters():
            p.requires_grad = False
        return


class LastResidualAttentionBlock(nn.Module):
    def __init__(self, clip_model: clip, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", nn.GELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

        self._initialize_weights(clip_model)

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        y = self.ln_1(x)
        y = F.linear(y, self.attn.in_proj_weight, self.attn.in_proj_bias)
        N, L, C = y.shape
        y = y.view(N, L, 3, C // 3).permute(2, 0, 1, 3).reshape(3 * N, L, C // 3)
        y = F.linear(y, self.attn.out_proj.weight, self.attn.out_proj.bias)
        q, k, v = y.tensor_split(3, dim=0)
        v += x
        v = v + self.mlp(self.ln_2(v))
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        out = [x, q, k, v]
        return out

    def _initialize_weights(self, clip_model):
        self.ln_1 = clip_model.visual.transformer.resblocks[11].ln_1
        self.ln_1.eps = 1e-06
        self.attn = clip_model.visual.transformer.resblocks[11].attn.to(torch.float32)
        self.attn.batch_first = True
        self.mlp = clip_model.visual.transformer.resblocks[11].mlp.to(torch.float32)
        # self.mlp[1] = nn.GELU()
        self.ln_2 = clip_model.visual.transformer.resblocks[11].ln_2
        self.ln_2.eps = 1e-06
        for p in self.parameters():
            p.requires_grad = False
        return


class Transformer(nn.Module):
    def __init__(self, clip_model: clip, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblock = []
        for i in range(self.layers - 1):
            self.resblock.append(ResidualAttentionBlock(width, heads, attn_mask))
        self.resblock.append(LastResidualAttentionBlock(clip_model, width, heads, attn_mask))
        self._initialize_weights(clip_model)
        self.resblocks = nn.Sequential(*self.resblock)

    def forward(self, x: torch.Tensor):
        z, q, k, v = self.resblocks(x)

        return z, q, k, v

    def _initialize_weights(self, clip_model):
        for i in range(self.layers - 1):
            self.resblock[i]._initialize_weights(clip_model, i)
        return


class VisionTransformer(nn.Module):
    def __init__(self, clip_model: clip, input_resolution: int, patch_size: int, width: int, layers: int, heads: int,
                 output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.patch_size = patch_size
        self.dilation = [1, 1]
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=self.patch_size, stride=self.patch_size,
                               bias=False)

        self.cls_token = torch.load('utils/cls_token.pt')

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = nn.LayerNorm(width)

        self.transformer = Transformer(clip_model, width, layers, heads)

        self.ln_post = nn.LayerNorm(width)
        self.proj = clip_model.visual.proj.to(torch.float32)
        self._initialize_weights(clip_model)

    def forward(self, x, train=False, img_metas=None):
        B = x.shape[0]
        # PatchEmbed
        # Padding
        input_h, input_w = x.size()[-2:]
        kernel_h, kernel_w = (self.patch_size, self.patch_size)
        stride_h, stride_w = (self.patch_size, self.patch_size)
        output_h = math.ceil(input_h / stride_h)
        output_w = math.ceil(input_w / stride_w)
        pad_h = max((output_h - 1) * stride_h +
                    (kernel_h - 1) * self.dilation[0] + 1 - input_h, 0)
        pad_w = max((output_w - 1) * stride_w +
                    (kernel_w - 1) * self.dilation[1] + 1 - input_w, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [0, pad_w, 0, pad_h])
        x = x.to(device)
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        # cls_tokens = self.class_embedding.reshape(1, 1, 1024).expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Positional Embedding
        positional_embedding = self.positional_embedding
        positional_embedding = positional_embedding.unsqueeze(dim=0)
        pos_h = self.input_resolution // self.patch_size
        pos_w = self.input_resolution // self.patch_size
        cls_token_weight = positional_embedding[:, 0]
        pos_embed_weight = positional_embedding[:, (-1 * pos_h * pos_w):]
        pos_embed_weight = pos_embed_weight.reshape(
            1, pos_h, pos_w, positional_embedding.shape[2]).permute(0, 3, 1, 2)
        pos_embed_weight = F.interpolate(pos_embed_weight, size=(output_h, output_w), mode='bicubic',
                                         align_corners=False)
        cls_token_weight = cls_token_weight.unsqueeze(1)
        pos_embed_weight = torch.flatten(pos_embed_weight, 2).transpose(1, 2)
        positional_embedding = torch.cat((cls_token_weight, pos_embed_weight), dim=1)

        x = x + positional_embedding

        x = self.ln_pre(x)

        # x = x.permute(1, 0, 2)  # NLD -> LND
        # x, q, k, v = self.transformer(x)
        x, q, k, v = self.transformer(x)

        x = self.ln_post(x)
        v = self.ln_post(v)

        out = x[:, 1:]
        B, _, C = out.shape
        out = out.reshape(B, output_h, output_w,
                          C).permute(0, 3, 1, 2).contiguous()
        q = q[:, 1:]
        k = k[:, 1:]
        v = v[:, 1:]
        v = v.reshape(B, output_h, output_w, -1).permute(0, 3, 1, 2).contiguous()

        out = [out, q, k, v]
        cls_token = x[:, 0]

        if self.proj is not None:
            z_global = cls_token @ self.proj

        # return [v, (output_h, output_w), z_global, k]
        return [v, (output_h, output_w), z_global, k, positional_embedding[:, 1:, :]]

    def _initialize_weights(self, clip_model):
        self.conv1 = clip_model.visual.conv1.to(torch.float32)
        self.class_embedding = clip_model.visual.class_embedding
        self.positional_embedding = clip_model.visual.positional_embedding
        self.ln_pre = clip_model.visual.ln_pre
        self.ln_post = clip_model.visual.ln_post
        # tune CLIP
        for p in self.parameters():
            p.requires_grad = False
        return


class TextEncoder(nn.Module):
    def __init__(self, clip_model, training=False, cfg=None, device=None):
        super().__init__()
        self.transformer = clip_model.transformer.to(torch.float32)
        self.token_embedding = clip_model.token_embedding.to(torch.float32)
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection.to(torch.float32)
        self.dtype = torch.float32
        self.device = device
        token = torch.zeros((1, 73), dtype=torch.int).to(self.device)
        prompt_token = self.token_embedding(token)
        for p in self.parameters():
            p.requires_grad = False
        self.prompt_token = nn.Parameter(prompt_token)
        self.weight = False

        if not training:
            # prompt_token = torch.load(cfg.LOAD_PATH)['text_encoder.prompt_token']
            prompt_token = torch.load(cfg.LOAD_PATH)['module.text_encoder.prompt_token']
            self.prompt_token = nn.Parameter(prompt_token, requires_grad=False)

    def forward(self, cls_name_token):
        device = self.device
        prompt_token = self.prompt_token.repeat(cls_name_token.shape[0], 1, 1).to(device)
        cls_name_token = cls_name_token.to(device)

        start_token = self.token_embedding(torch.tensor(49406, dtype=torch.int, device=device)).repeat(
            cls_name_token.shape[0], 1, 1).to(device)
        cls_token = self.token_embedding(cls_name_token).to(device)
        x = torch.cat([start_token, prompt_token, cls_token], dim=1)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), 74 + cls_name_token.argmax(dim=-1)] @ self.text_projection
        return x


class ReCLIP(nn.Module):
    def __init__(self, cfg, clip_model, rank, zeroshot_weights):
        super(ReCLIP, self).__init__()
        self.vit = VisionTransformer(clip_model=clip_model,
                                     input_resolution=224,
                                     patch_size=16,
                                     width=768,
                                     layers=12,
                                     heads=12,
                                     output_dim=768)

        self.clip = clip_model
        self.k = cfg.DATASET.K
        visual_channel = cfg.MODEL.VISUAL_CHANNEL
        text_channel = cfg.MODEL.TEXT_CHANNEL
        self.proj = nn.Conv2d(visual_channel, text_channel, 1, bias=False)
        self._initialize_weights(clip_model)
        self.logit_scale = clip_model.logit_scale
        for p in self.parameters():
            p.requires_grad = False
        self.text_encoder = TextEncoder(clip_model, training=cfg.MODEL.TRAINING, cfg=cfg, device=rank)
        self.cnum = cfg.DATASET.NUM_CLASSES
        self.device = rank

        if cfg.MODEL.TRAINING:
            self.conv1 = nn.Conv2d(768, 512, kernel_size=3, stride=1, padding=1)
            self.norm1 = nn.BatchNorm2d(512)
            self.conv2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
            self.norm2 = nn.BatchNorm2d(256)
            self.conv3 = nn.Conv2d(256, self.cnum, kernel_size=3, stride=1, padding=1)
            self.norm3 = nn.BatchNorm2d(self.cnum)

            nn.init.kaiming_normal_(self.conv1.weight, a=0, mode='fan_out', nonlinearity='relu')
            nn.init.kaiming_normal_(self.conv2.weight, a=0, mode='fan_out', nonlinearity='relu')
            nn.init.kaiming_normal_(self.conv3.weight, a=0, mode='fan_out', nonlinearity='relu')

            nn.init.constant_(self.norm1.weight, 1)
            nn.init.constant_(self.norm1.bias, 0)
            nn.init.constant_(self.norm2.weight, 1)
            nn.init.constant_(self.norm2.bias, 0)
            nn.init.constant_(self.norm3.weight, 1)
            nn.init.constant_(self.norm3.bias, 0)

            self.proj_net = nn.Sequential(self.conv1,
                                        self.norm1,
                                        nn.ReLU(inplace=True),
                                        self.conv2,
                                        self.norm2,
                                        nn.ReLU(inplace=True),
                                        self.conv3,
                                        self.norm3,
                                        nn.ReLU(inplace=True)).to(self.device)

        else:
            self.conv1 = nn.Conv2d(768, 512, kernel_size=3, stride=1, padding=1)
            self.norm1 = nn.BatchNorm2d(512)
            self.conv2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
            self.norm2 = nn.BatchNorm2d(256)
            self.conv3 = nn.Conv2d(256, self.cnum, kernel_size=3, stride=1, padding=1)
            self.norm3 = nn.BatchNorm2d(self.cnum)

            self.proj_net = nn.Sequential(self.conv1,
                                        self.norm1,
                                        nn.ReLU(inplace=True),
                                        self.conv2,
                                        self.norm2,
                                        nn.ReLU(inplace=True),
                                        self.conv3,
                                        self.norm3,
                                        nn.ReLU(inplace=True)).to(self.device)

    def forward(self, image, gt_cls, zeroshot_weights, cls_name_token, training=False, img_metas=None,
                return_feat=False):
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cnum = zeroshot_weights.shape[0]
        device = self.device
        gt_cls_text_embeddings = zeroshot_weights.to(device)

        batch_size = image.shape[0]
        image = image.to(device)
        v, shape, z_global, k, positional_embedding = self.vit(image, train=False, img_metas=img_metas)
        positional_embedding = positional_embedding.reshape(1, shape[0], shape[1], -1).permute(0, 3, 1, 2)

        feat = self.proj(v)
        feat = feat / feat.norm(dim=1, keepdim=True)

        logit_scale = self.logit_scale.exp()

        # ori
        output_q = F.conv2d(feat, gt_cls_text_embeddings[:, :, None, None]).permute(0, 2, 3, 1).reshape(batch_size, -1,
                                                                                                        cnum)

        # reference prompt
        prompt = self.text_encoder(cls_name_token)
        prompt = prompt / prompt.norm()

        pe = self.proj_net(positional_embedding).expand(batch_size, cnum, positional_embedding.shape[2], positional_embedding.shape[3]).permute(0, 2, 3, 1).reshape(batch_size, -1, cnum)
        output_r = F.conv2d(feat, prompt[:, :, None, None]).permute(0, 2, 3, 1).reshape(batch_size, -1, cnum)
        output_r = torch.add(output_r, pe)
        output = torch.sub(output_q, output_r).permute(0, 2, 1).reshape(batch_size, -1, shape[0], shape[1])

        if return_feat:
            return output[0], feat[0], shape

        if training:
            # gumbel softmax
            output_scale = torch.mul(output.reshape(batch_size, cnum, -1).permute(0, 2, 1), 1000)
            output_gumbel = F.gumbel_softmax(output_scale, tau=1, hard=True, dim=2).reshape(batch_size, shape[0], shape[1], -1)

            loss = 0

            for j in range(batch_size):
                masked_image_features = []
                if len(gt_cls[j]) == 0:
                    continue
                for i in gt_cls[j]:
                    mask = output_gumbel[j, :, :, i].unsqueeze(dim=0)

                    masked_image_feature = torch.mul(feat[j].unsqueeze(dim=0), mask)
                    feature_pool = nn.AdaptiveAvgPool2d((1, 1))(masked_image_feature).reshape(1, 512)
                    masked_image_features.append(feature_pool)
                masked_image_features = torch.stack(masked_image_features, dim=0).squeeze(dim=1)

                similarity_img = logit_scale * masked_image_features @ gt_cls_text_embeddings.t()
                labels = torch.tensor(gt_cls[j]).to(device)
                loss += F.cross_entropy(similarity_img, labels)
            return output, loss / batch_size

        return output

    def _initialize_weights(self, clip_model):
        # tune CLIP
        self.proj.weight = nn.Parameter(clip_model.visual.proj[:, :, None, None].permute(1, 0, 2, 3).to(torch.float32),
                                        requires_grad=False)


class RECLIPPP(nn.Module):
    def __init__(self, cfg, clip_model, rank):
        super(RECLIPPP, self).__init__()
        self.vit = VisionTransformer(clip_model=clip_model,
                                     input_resolution=224,
                                     patch_size=16,
                                     width=768,
                                     layers=12,
                                     heads=12,
                                     output_dim=768)

        self.clip = clip_model
        self.k = cfg.DATASET.K
        visual_channel = cfg.MODEL.VISUAL_CHANNEL
        text_channel = cfg.MODEL.TEXT_CHANNEL
        self.proj = nn.Conv2d(visual_channel, text_channel, 1, bias=False)
        self._initialize_weights(clip_model)
        self.logit_scale = clip_model.logit_scale
        for p in self.parameters():
            p.requires_grad = False
        self.text_encoder = TextEncoder(clip_model, training=cfg.MODEL.TRAINING, cfg=cfg, device=rank)
        self.cnum = cfg.DATASET.NUM_CLASSES
        self.device = rank

        if cfg.MODEL.TRAINING:
            self.pe_proj = nn.Conv2d(768, 512, kernel_size=1)

            # decoder
            self.decoder_conv2 = nn.Conv2d(512 + self.cnum, self.cnum, kernel_size=5, padding=2, stride=1)
            nn.init.kaiming_normal_(self.decoder_conv2.weight, a=0, mode='fan_out', nonlinearity='relu')
            self.decoder_norm2 = nn.BatchNorm2d(self.cnum)
            nn.init.constant_(self.decoder_norm2.weight, 1)
            nn.init.constant_(self.decoder_norm2.bias, 0)

        else:
            self.pe_proj = nn.Conv2d(768, 512, kernel_size=1)
            self.decoder_conv2 = nn.Conv2d(self.cnum + 512, self.cnum, kernel_size=5, padding=2, stride=1)
            self.decoder_norm2 = nn.BatchNorm2d(self.cnum)

    def forward(self, image, gt_cls, zeroshot_weights, cls_name_token, training=False, img_metas=None,
                return_feat=False):
        cnum = zeroshot_weights.shape[0]
        device = self.device
        gt_cls_text_embeddings = zeroshot_weights.to(device)

        batch_size = image.shape[0]
        image = image.to(device)
        v, shape, z_global, k, positional_embedding = self.vit(image, train=False, img_metas=img_metas)
        positional_embedding = positional_embedding.reshape(1, shape[0], shape[1], -1).permute(0, 3, 1, 2)

        feat = self.proj(v)
        feat = feat / feat.norm(dim=1, keepdim=True)

        logit_scale = self.logit_scale.exp()

        # ori
        output_q = F.conv2d(feat, gt_cls_text_embeddings[:, :, None, None]).permute(0, 2, 3, 1).reshape(batch_size, -1,
                                                                                                        cnum)

        # reference prompt
        prompt = self.text_encoder(cls_name_token)
        prompt = prompt / prompt.norm()

        pe = self.pe_proj(positional_embedding).permute(0, 2, 3, 1).reshape(1, shape[0] * shape[1], -1)
        bias_logits = pe @ prompt.t()
        output = torch.sub(output_q, bias_logits).permute(0, 2, 1).reshape(batch_size, -1, shape[0], shape[1])

        feature = torch.cat((feat, output), dim=1)
        feature = self.decoder_conv2(feature)
        feature = self.decoder_norm2(feature)
        output = feature

        if return_feat:
            return output[0], feat[0], shape

        if training:
            # gumbel softmax
            output_scale = torch.mul(output.reshape(batch_size, cnum, -1).permute(0, 2, 1), 100)
            output_gumbel = F.gumbel_softmax(output_scale, tau=1, hard=True, dim=2).reshape(batch_size, shape[0], shape[1], -1)

            loss = 0

            for j in range(batch_size):
                masked_image_features = []
                if len(gt_cls[j]) == 0:
                    continue
                for i in gt_cls[j]:
                    mask = output_gumbel[j, :, :, i].unsqueeze(dim=0)
                    masked_image_feature = torch.mul(feat[j].unsqueeze(dim=0), mask)
                    feature_pool = nn.AdaptiveAvgPool2d((1, 1))(masked_image_feature).reshape(1, 512)
                    masked_image_features.append(feature_pool)
                masked_image_features = torch.stack(masked_image_features, dim=0).squeeze(dim=1)

                similarity_img = logit_scale * masked_image_features @ gt_cls_text_embeddings.t()
                labels = torch.tensor(gt_cls[j]).to(device)
                loss += F.cross_entropy(similarity_img, labels)

            return output, loss / batch_size

        return output

    def _initialize_weights(self, clip_model):
        self.proj.weight = nn.Parameter(clip_model.visual.proj[:, :, None, None].permute(1, 0, 2, 3).to(torch.float32),
                                        requires_grad=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes=64,
                 planes=64,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 plugins=None,
                 init_cfg=None):
        super(Bottleneck, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.dcn = dcn
        self.with_dcn = dcn is not None
        self.plugins = plugins
        self.with_plugins = plugins is not None

        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False).to(device)
        self.norm1 = nn.BatchNorm2d(planes).to(device)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=self.conv2_stride, padding=dilation,
                               dilation=dilation, bias=False).to(device)
        self.norm2 = nn.BatchNorm2d(planes).to(device)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False).to(device)
        self.norm3 = nn.BatchNorm2d(planes * self.expansion).to(device)

        # TODO:
        # for param in norm_layer.parameters():
        # param.requires_grad = requires_grad

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward_plugin(self, x, plugin_names):
        """Forward function for plugins."""
        out = x
        for name in plugin_names:
            out = getattr(self, name)(x)
        return out

    def forward(self, x):

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv1_plugin_names)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)
            if self.style == 'clip':
                out = self.avgpool(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv2_plugin_names)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv3_plugin_names)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


class ResLayer(nn.Sequential):
    def __init__(self,
                 block,
                 inplanes,
                 planes,
                 num_blocks,
                 stride=1,
                 dilation=1,
                 avg_down=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 multi_grid=None,
                 contract_dilation=False,
                 style='pytorch',
                 **kwargs):
        self.block = block

        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = []
            conv_stride = stride
            if avg_down:
                conv_stride = 1
                if style == 'clip':
                    downsample.append(
                        nn.AvgPool2d(kernel_size=stride))
                else:
                    downsample.append(
                        nn.AvgPool2d(
                            kernel_size=stride,
                            stride=stride,
                            ceil_mode=True,
                            count_include_pad=False))
            downsample.extend([
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=conv_stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            ])
            downsample = nn.Sequential(*downsample)

        layers = []
        if multi_grid is None:
            if dilation > 1 and contract_dilation:
                first_dilation = dilation // 2
            else:
                first_dilation = dilation
        else:
            first_dilation = multi_grid[0]
        layers.append(
            block(
                inplanes=inplanes,
                planes=planes,
                stride=stride,
                dilation=first_dilation,
                downsample=downsample,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                style=style,
                **kwargs))
        inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=1,
                    dilation=dilation if multi_grid is None else multi_grid[i],
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    **kwargs))
        self.layers = layers
        super(ResLayer, self).__init__(*layers)


class ResNet(nn.Module):
    arch_settings = {
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3)),
        '50x4': (Bottleneck, (4, 6, 10, 6)),
        '50x16': (Bottleneck, (6, 8, 18, 8)),
    }

    def __init__(self,
                 depth=50,
                 in_channels=3,
                 stem_channels=64,
                 base_channels=64,
                 num_stages=4,
                 strides=(1, 2, 1, 1),
                 dilations=(1, 1, 2, 4),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 deep_stem=False,
                 avg_down=False,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='SyncBN', requires_grad=True),
                 norm_eval=False,
                 dcn=None,
                 stage_with_dcn=(False, False, False, False),
                 plugins=None,
                 multi_grid=None,
                 contract_dilation=True,
                 with_cp=False,
                 zero_init_residual=True,
                 pretrained='open-mmlab://resnet50_v1c',
                 init_cfg=None,
                 cfg=None):
        super(ResNet, self).__init__()
        self.pretrained = pretrained
        self.zero_init_residual = zero_init_residual
        block_init_cfg = None
        self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)

        self.depth = depth
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        self.strides = strides
        self.dilations = dilations
        self.out_indices = out_indices
        self.style = style
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.dcn = dcn
        self.stage_with_dcn = stage_with_dcn
        self.plugins = plugins
        self.multi_grid = multi_grid
        self.contract_dilation = contract_dilation
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = stem_channels
        resnet50_v1c = torch.load('pretrain/resnet50_v1c-2cccc1ad.pth')

        self._make_stem_layer(in_channels, stem_channels, resnet50_v1c)
        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            dcn = self.dcn if self.stage_with_dcn[i] else None
            if plugins is not None:
                stage_plugins = self.make_stage_plugins(plugins, i)
            else:
                stage_plugins = None
            # multi grid is applied to last layer only
            stage_multi_grid = multi_grid if i == len(
                self.stage_blocks) - 1 else None
            planes = base_channels * 2 ** i
            res_layer = self.make_res_layer(
                block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                avg_down=self.avg_down,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                dcn=dcn,
                plugins=stage_plugins,
                multi_grid=stage_multi_grid,
                contract_dilation=contract_dilation,
                init_cfg=block_init_cfg)
            self.res_layers.append(res_layer)
            self.inplanes = planes * self.block.expansion
        for i in range(len(self.res_layers)):
            for j in range(self.stage_blocks[i]):
                layer = i + 1
                block = j
                prefix = 'layer' + str(layer) + '.' + str(block)
                self.res_layers[i][j].conv1.weight = nn.Parameter(resnet50_v1c['state_dict'][prefix + '.conv1.weight'])
                self.res_layers[i][j].norm1.weight = nn.Parameter(resnet50_v1c['state_dict'][prefix + '.bn1.weight'])
                self.res_layers[i][j].norm1.bias = nn.Parameter(resnet50_v1c['state_dict'][prefix + '.bn1.bias'])
                self.res_layers[i][j].conv2.weight = nn.Parameter(resnet50_v1c['state_dict'][prefix + '.conv2.weight'])
                self.res_layers[i][j].norm2.weight = nn.Parameter(resnet50_v1c['state_dict'][prefix + '.bn2.weight'])
                self.res_layers[i][j].norm2.bias = nn.Parameter(resnet50_v1c['state_dict'][prefix + '.bn2.bias'])
                self.res_layers[i][j].conv3.weight = nn.Parameter(resnet50_v1c['state_dict'][prefix + '.conv3.weight'])
                self.res_layers[i][j].norm3.weight = nn.Parameter(resnet50_v1c['state_dict'][prefix + '.bn3.weight'])
                self.res_layers[i][j].norm3.bias = nn.Parameter(resnet50_v1c['state_dict'][prefix + '.bn3.bias'])
                if j == 0:
                    self.res_layers[i][0].downsample[0].weight = nn.Parameter(
                        resnet50_v1c['state_dict'][prefix + '.downsample.0.weight'])
                    self.res_layers[i][0].downsample[1].weight = nn.Parameter(
                        resnet50_v1c['state_dict'][prefix + '.downsample.1.weight'])
                    self.res_layers[i][0].downsample[1].bias = nn.Parameter(
                        resnet50_v1c['state_dict'][prefix + '.downsample.1.bias'])
            self.res_layers[i] = self.res_layers[i].to(device)
        self.res_layers = nn.Sequential(*self.res_layers)
        self._freeze_stages()

        self.feat_dim = self.block.expansion * base_channels * 2 ** (
                len(self.stage_blocks) - 1)

    def forward(self, x):
        if self.deep_stem:
            x = x.to(device)
            x = self.stem(x)
        x = self.stem_pool(x)
        outs = []
        for i in range(len(self.res_layers)):
            res_layer = self.res_layers[i]
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def _make_stem_layer(self, in_channels, stem_channels, resnet50_v1c):
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels // 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels // 2),
            # nn.SyncBatchNorm(stem_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(stem_channels // 2, stem_channels // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels // 2),
            # nn.SyncBatchNorm(stem_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(stem_channels // 2, stem_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            # nn.SyncBatchNorm(stem_channels),
            nn.ReLU(inplace=True),
        )
        self.stem[0].weight = nn.Parameter(resnet50_v1c['state_dict']['stem.0.weight'])
        self.stem[1].weight = nn.Parameter(resnet50_v1c['state_dict']['stem.1.weight'])
        self.stem[1].bias = nn.Parameter(resnet50_v1c['state_dict']['stem.1.bias'])
        self.stem[3].weight = nn.Parameter(resnet50_v1c['state_dict']['stem.3.weight'])
        self.stem[4].weight = nn.Parameter(resnet50_v1c['state_dict']['stem.4.weight'])
        self.stem[4].bias = nn.Parameter(resnet50_v1c['state_dict']['stem.4.bias'])
        self.stem[6].weight = nn.Parameter(resnet50_v1c['state_dict']['stem.6.weight'])
        self.stem[7].weight = nn.Parameter(resnet50_v1c['state_dict']['stem.7.weight'])
        self.stem[7].bias = nn.Parameter(resnet50_v1c['state_dict']['stem.7.bias'])

        self.stem = self.stem.to(device)

        self.stem_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stem_pool = self.stem_pool.to(device)

    def make_res_layer(self, **kwargs):
        """Pack all blocks in a stage into a ``ResLayer``."""
        return ResLayer(**kwargs)

    def _freeze_stages(self):
        """Freeze stages param and norm stats."""
        if self.frozen_stages >= 0:
            if self.deep_stem:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            else:
                self.norm1.eval()
                for m in [self.conv1, self.norm1]:
                    for param in m.parameters():
                        param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False


class ResNetV1c(ResNet):
    def __init__(self, cfg=None, **kwargs):
        super(ResNetV1c, self).__init__(
            cfg=cfg, deep_stem=True, avg_down=False, **kwargs)


class ConvModule(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias='auto',
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 inplace=True,
                 with_spectral_norm=False,
                 padding_mode='zeros',
                 order=('conv', 'norm', 'act')):
        super(ConvModule, self).__init__()
        official_padding_mode = ['zeros', 'circular']
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.inplace = inplace
        self.with_spectral_norm = with_spectral_norm
        self.with_explicit_padding = padding_mode not in official_padding_mode
        self.order = order

        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None
        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == 'auto':
            bias = not self.with_norm
        self.with_bias = bias

        # if self.with_explicit_padding:
        #     pad_cfg = dict(type=padding_mode)
        #     self.padding_layer = build_padding_layer(pad_cfg, padding)

        # reset padding to 0 for conv module
        conv_padding = 0 if self.with_explicit_padding else padding
        # build convolution layer
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=conv_padding,
                              dilation=dilation, groups=groups, bias=bias)
        # export the attributes of self.conv to a higher level for convenience
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = padding
        self.dilation = self.conv.dilation
        self.transposed = self.conv.transposed
        self.output_padding = self.conv.output_padding
        self.groups = self.conv.groups

        if self.with_spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv)

        # build normalization layers
        if self.with_norm:
            # norm layer is after conv layer
            if order.index('norm') > order.index('conv'):
                norm_channels = out_channels
            else:
                norm_channels = in_channels
            self.norm = nn.BatchNorm2d(norm_channels)
        else:
            self.norm_name = None

        # build activation layer
        if self.with_activation:
            self.activate = nn.ReLU(inplace=True)

        # Use msra init by default
        self.init_weights()

    def init_weights(self):
        if not hasattr(self.conv, 'init_weights'):
            if self.with_activation and self.act_cfg['type'] == 'LeakyReLU':
                nonlinearity = 'leaky_relu'
                a = self.act_cfg.get('negative_slope', 0.01)
            else:
                nonlinearity = 'relu'
                a = 0
            nn.init.kaiming_normal_(
                self.conv.weight, a=a, mode='fan_out', nonlinearity=nonlinearity)
        if self.with_norm:
            nn.init.constant_(self.norm.weight, 1)
            nn.init.constant_(self.norm.bias, 0)

    def forward(self, x, activate=True, norm=True):
        for layer in self.order:
            if layer == 'conv':
                if self.with_explicit_padding:
                    x = self.padding_layer(x)
                x = x.to(device)
                x = self.conv(x)
            elif layer == 'norm' and norm and self.with_norm:
                x = self.norm(x)
            elif layer == 'act' and activate and self.with_activation:
                x = self.activate(x)
        return x


class ASPPModuleV2(nn.ModuleList):
    """Atrous Spatial Pyramid Pooling (ASPP) Module.

    Args:
        dilations (tuple[int]): Dilation rate of each layer.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
    """

    def __init__(self, dilations, in_channels, channels, conv_cfg, norm_cfg,
                 act_cfg):
        super(ASPPModuleV2, self).__init__()
        self.dilations = dilations
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        for dilation in dilations:
            self.append(
                ConvModule(
                    self.in_channels,
                    self.channels,
                    1 if dilation == 1 else 3,
                    dilation=dilation,
                    padding=0 if dilation == 1 else dilation,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

    def forward(self, x):
        """Forward function."""
        aspp_outs = []
        for aspp_module in self:
            aspp_outs.append(aspp_module(x))

        return aspp_outs


class ASPPHeadV2(nn.Module):
    def __init__(self, num_classes, dilations=(6, 12, 18, 24), in_channels=2048, channels=512):
        super(ASPPHeadV2, self).__init__()
        self.in_index = -1
        self.in_channels = in_channels
        self.channels = channels
        self.num_classes = num_classes
        self.dropout_ratio = 0
        self.conv_cfg = None
        self.norm_cfg = {'type': 'SyncBN', 'requires_grad': True}
        self.act_cfg = {'type': 'ReLU'}
        self.ignore_index = 255
        self.align_corners = False
        self.loss = F.cross_entropy
        self.sampler = None
        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        self.dropout = None
        self.fp16_enabled = False
        self.freeze = False

        assert isinstance(dilations, (list, tuple))
        self.dilations = dilations
        self.aspp_modules = ASPPModuleV2(
            dilations,
            self.in_channels,
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.aspp_modules = self.aspp_modules.to(device)

    def forward(self, inputs):
        x = inputs[self.in_index]
        aspp_outs = self.aspp_modules(x)
        output = sum(aspp_outs)
        return output


class ReCLIP_DISTILL_HEAD(nn.Module):
    def __init__(self, cfg, cls_name_token, clip_model, text_categories,
                 text_embeddings, text_channels=512, cls_bg=False, norm_feat=True,
                 unlabeled_cats=[], clip_unlabeled_cats=[],
                 reset_counter=False, clip_channels=768,
                 vit=True, ks_thresh=0., pd_thresh=0., conf_thresh=0.,
                 distill=False, distill_labeled=True, **kwargs):
        super().__init__()

        # decode_head
        init_cfg = dict(type='Normal', std=0.01, override=dict(name='conv_seg')),
        self._init_inputs(in_channels=2048, in_index=-1, input_transform=None)
        self.channels = 512
        self.num_classes = text_categories
        self.dropout_ratio = 0
        self.conv_cfg = None
        self.norm_cfg = {'type': 'SyncBN', 'requires_grad': True}
        self.act_cfg = {'type': 'ReLU'}
        self.in_index = -1
        self.ignore_index = 255
        self.align_corners = False
        self.sampler = None
        self.conv_seg = nn.Conv2d(self.channels, self.num_classes, kernel_size=1)
        self.dropout = None
        self.fp16_enabled = False

        self.text_categories = text_categories
        self.text_channels = text_channels
        self.cnum = text_categories
        self.norm_feat = norm_feat
        self.unlabeled_cats = unlabeled_cats
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.clip_unlabeled_cats = torch.arange(text_categories).to(device)
        self.self_train = False
        self.clip_guided = True
        self.train_unlabeled = self.self_train or self.clip_guided
        self.cls_bg = cls_bg
        self.reset_counter = reset_counter
        self.distill = False
        self.distill_labeled = True

        del self.conv_seg
        self.init_cfg = None

        # decode module
        self.decode_module = ASPPHeadV2(num_classes=text_categories, in_channels=2048, channels=512)

        self.text_embeddings = text_embeddings.to(device)

        text_encoder = TextEncoder(clip_model, training=False, cfg=cfg, device=device)
        prompt = text_encoder(cls_name_token)
        prompt = prompt / prompt.norm()
        self.reference_prompt = prompt
        self.logit_scale = clip_model.logit_scale

        self.cls_name_token = cls_name_token

        self.k = cfg.DATASET.K
        self.threshold = cfg.DATASET.THRESHOLD

        self.vit = vit

        ori_weight = torch.load(cfg.LOAD_PATH)

        # weight = torch.load(cfg.LOAD_PATH)
        weight = {}
        for key, value in ori_weight.items():
            new_key = key[7:]
            weight[new_key] = value

        self.conv1 = nn.Conv2d(768, 512, kernel_size=3, stride=1, padding=1)
        self.conv1.weight = nn.Parameter(weight['proj_net.0.weight'], requires_grad=False)
        self.conv1.bias = nn.Parameter(weight['proj_net.0.bias'], requires_grad=False)

        self.norm1 = nn.BatchNorm2d(512)
        self.norm1.weight = nn.Parameter(weight['proj_net.1.weight'], requires_grad=False)
        self.norm1.bias = nn.Parameter(weight['proj_net.1.bias'], requires_grad=False)
        # self.norm1.running_mean = nn.Parameter(weight['proj_net.1.running_mean'], requires_grad=False)
        # self.norm1.running_var = nn.Parameter(weight['proj_net.1.running_var'], requires_grad=False)
        # self.norm1.num_batches_tracked = nn.Parameter(weight['proj_net.1.num_batches_tracked'], requires_grad=False)

        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.conv2.weight = nn.Parameter(weight['proj_net.3.weight'], requires_grad=False)
        self.conv2.bias = nn.Parameter(weight['proj_net.3.bias'], requires_grad=False)

        self.norm2 = nn.BatchNorm2d(256)
        self.norm2.weight = nn.Parameter(weight['proj_net.4.weight'], requires_grad=False)
        self.norm2.bias = nn.Parameter(weight['proj_net.4.bias'], requires_grad=False)
        # self.norm2.running_mean = nn.Parameter(weight['proj_net.4.running_mean'], requires_grad=False)
        # self.norm2.running_var = nn.Parameter(weight['proj_net.4.running_var'], requires_grad=False)
        # self.norm2.num_batches_tracked = nn.Parameter(weight['proj_net.4.num_batches_tracked'], requires_grad=False)

        self.conv3 = nn.Conv2d(256, self.cnum, kernel_size=3, stride=1, padding=1)
        self.conv3.weight = nn.Parameter(weight['proj_net.6.weight'], requires_grad=False)
        self.conv3.bias = nn.Parameter(weight['proj_net.6.bias'], requires_grad=False)

        self.norm3 = nn.BatchNorm2d(self.cnum)
        self.norm3.weight = nn.Parameter(weight['proj_net.7.weight'], requires_grad=False)
        self.norm3.bias = nn.Parameter(weight['proj_net.7.bias'], requires_grad=False)
        # self.norm3.running_mean = nn.Parameter(weight['proj_net.7.running_mean'], requires_grad=False)
        # self.norm3.running_var = nn.Parameter(weight['proj_net.7.running_var'], requires_grad=False)
        # self.norm3.num_batches_tracked = nn.Parameter(weight['proj_net.7.num_batches_tracked'], requires_grad=False)

        self.proj_net = nn.Sequential(self.conv1,
                                      self.norm1,
                                      nn.ReLU(inplace=True),
                                      self.conv2,
                                      self.norm2,
                                      nn.ReLU(inplace=True),
                                      self.conv3,
                                      self.norm3,
                                      nn.ReLU(inplace=True)).to(device)

        if self.clip_guided:
            self.clip = VisionTransformer(clip_model=clip_model,
                                          input_resolution=224,
                                          patch_size=16,
                                          width=768,
                                          layers=12,
                                          heads=12,
                                          output_dim=768)
            self.ks_thresh = ks_thresh
            self.pd_thresh = pd_thresh
            self.conf_thresh = conf_thresh
            if vit:
                self.proj = nn.Conv2d(clip_channels, text_channels, 1, bias=False)
                self.proj.weight = nn.Parameter(
                    clip_model.visual.proj[:, :, None, None].permute(1, 0, 2, 3).to(torch.float32),
                    requires_grad=False)
            for param in self.clip.parameters():
                param.requires_grad = False

    def _init_inputs(self, in_channels, in_index, input_transform):
        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def decode(self, x, pe=None, shape=None, feature=None):
        output = self.decode_module(x)
        feat = output.detach()
        if feature:
            return feat
        output, logits = self.cls_seg(output, pe=pe, shape=shape)

        if self.train_unlabeled:
            return [output, feat, logits]

        return [output]

    def cls_seg(self, feat, pe=None, shape=None):
        if self.dropout is not None:
            feat = self.dropout(feat)
        if self.norm_feat:
            feat = feat / feat.norm(dim=1, keepdim=True)

        # output = self.cls_seg_head(feat)
        # output_q = F.conv2d(feat, self.cls_text[:, :, None, None])
        output_q = F.conv2d(feat, self.text_embeddings[:, :, None, None])
        # batch_size = output_q.shape[0]

        output_r = F.conv2d(feat, self.reference_prompt[:, :, None, None])
        output = torch.sub(output_q, output_r)

        return output, output

    def assign_label(self, gt_semantic_seg, feat, norm=True, unlabeled_cats=None,
                     clip=False, k=None, cls_token=None, pe=None):
        if norm:
            feat = feat / feat.norm(dim=1, keepdim=True)

        gt_semantic_seg = gt_semantic_seg.squeeze(1).to(device)
        text_embeddings = self.text_embeddings
        unlabeled_idx = (gt_semantic_seg < 0)

        output_q = torch.einsum('nchw,lc->nlhw', [feat, text_embeddings])
        # output_mask = output_q
        batch_size = output_q.shape[0]
        shape = output_q.shape[2:]

        output_r = torch.einsum('nchw,lc->nlhw', [feat, self.reference_prompt])
        pe = pe.permute(0, 2, 1).reshape(output_r.shape)
        output_r = torch.add(output_r, pe)
        output_mask = torch.sub(output_q, output_r)

        # pd
        N, C, H, W = output_mask.shape
        _output = F.softmax(output_mask * 10, dim=1)
        max_cls_conf = _output.view(N, C, -1).max(dim=-1)[0]
        selected_cls = (max_cls_conf < 0.5)[:, :, None, None].expand(N, C, H, W)
        output_mask[selected_cls] = -100

        output_mask = F.interpolate(input=output_mask, size=gt_semantic_seg.shape[1:], mode='bilinear',
                                    align_corners=self.align_corners)
        output_mask = output_mask.permute(0, 2, 3, 1)
        match_matrix = output_mask[unlabeled_idx]
        gt_semantic_seg[unlabeled_idx] = unlabeled_cats[match_matrix.argmax(dim=1)]

        return gt_semantic_seg[:, None, :, :], output_mask, output_q

    def forward(self, filenames=None, text=None, train=None, self_train=False, inputs=None, img=None,
                gt_semantic_seg=None, feat=False, cls=None):
        if feat:
            return self.decode(inputs, feature=True)
        if train:
            batch_size = img.shape[0]

            gt_self, gt_clip, gt_weight = None, None, None
            with torch.no_grad():
                # clip cannot deal with background
                gt = gt_semantic_seg.clone()
                x = self.clip(img, train=False)
                q, k, v, cls_token = None, None, None, None
                if self.vit:
                    if isinstance(x, list) and len(x) == 5:
                        v, shape, z_global, k, positional_embedding = x
                        dim = positional_embedding.shape[2]
                        positional_embedding = positional_embedding.expand(batch_size, shape[0] * shape[1], dim)
                        positional_embedding = positional_embedding.reshape(batch_size, shape[0], shape[1], -1).permute(0, 3, 1, 2)
                        pe = self.proj_net(positional_embedding).permute(0, 2, 3, 1).reshape(batch_size, -1, self.cnum)
                        pe_res = None
                    if v is not None:
                        clip_feat = self.proj(v)

                gt_clip, logits_output, logits_clip = self.assign_label(gt, clip_feat,
                                                                        True, self.clip_unlabeled_cats,
                                                                        k=k, cls_token=cls_token, clip=True, pe=pe)
            if gt_clip is not None:
                gt_semantic_seg = gt_clip
                gt_semantic_seg[gt_semantic_seg < 0] = 255
            # import pdb;pdb.set_trace()
            # gt_semantic_seg = gt_semantic_seg.to(device)
            # gt_semantic_seg = gt_semantic_seg.unsqueeze(1)
            seg_logits, feat, logits = self.decode(inputs, pe=pe_res, shape=shape)
            seg_logits = F.interpolate(seg_logits, size=gt_semantic_seg.shape[2:], mode='bilinear', align_corners=False)
            feat = F.interpolate(feat, size=gt_semantic_seg.shape[2:], mode='bilinear', align_corners=False)
            clip_feat = F.interpolate(clip_feat, size=gt_semantic_seg.shape[2:], mode='bilinear', align_corners=False)
            gt_semantic_seg = gt_semantic_seg.squeeze(1)

            # feature
            mask = (gt_semantic_seg != 255)
            l1_loss = F.l1_loss(feat.permute(0, 2, 3, 1)[mask], clip_feat.permute(0, 2, 3, 1)[mask])

            # contrastive loss
            ce_loss = 0
            for j in range(batch_size):
                masked_image_features = []
                for i in cls[j]:
                    mask = (gt_semantic_seg[j] == i)
                    masked_image_feature = torch.mul(feat[j].unsqueeze(dim=0), mask)
                    feature_pool = nn.AdaptiveAvgPool2d((1, 1))(masked_image_feature).reshape(1, 512)
                    masked_image_features.append(feature_pool)
                masked_image_features = torch.stack(masked_image_features, dim=0).squeeze(dim=1)
                similarity_img = self.logit_scale * masked_image_features @ self.text_embeddings.t()
                labels = torch.tensor(cls[j]).to(self.device)
                ce_loss += F.cross_entropy(similarity_img, labels)

            # mask-guided loss
            losses = F.cross_entropy(seg_logits, gt_semantic_seg, weight=None, reduction='none', ignore_index=255)
            return losses.mean() + 0.5 * l1_loss + 0.5 * ce_loss / batch_size

        else:
            with torch.no_grad():
                batch_size = img.shape[0]
                x = self.clip(img, train=False)
                v, shape, z_global, k, positional_embedding = x
                dim = positional_embedding.shape[2]

                positional_embedding = positional_embedding.expand(batch_size, shape[0] * shape[1], dim)
                positional_embedding = positional_embedding.reshape(batch_size, shape[0], shape[1], -1).permute(0, 3, 1, 2)
                pe = self.proj_net(positional_embedding).permute(0, 2, 3, 1).reshape(batch_size, -1, self.cnum)
                pe_res=None

                seg_logits, _, _ = self.decode(inputs, shape=shape, pe=pe_res)

            return seg_logits


class ReCLIP_DISTILL(nn.Module):
    def __init__(self, clip_model, cfg, cls_name_token, text_categories, text_channels, text_embeddings):
        super().__init__()
        self.backbone = ResNetV1c(cfg)
        self.decode_head = ReCLIP_DISTILL_HEAD(clip_model=clip_model, cfg=cfg, cls_name_token=cls_name_token,
                                            text_categories=text_categories, text_embeddings=text_embeddings,
                                            text_channels=text_channels)

    def forward(self, img, label, filenames, text, cls=None, self_train=False, train=False):
        x = self.backbone(img)
        result = self.decode_head(filenames, text=text, train=train, inputs=x, img=img, gt_semantic_seg=label, cls=cls,
                                  self_train=self_train)
        return result

    def get_feat(self, img):
        x = self.backbone(img)
        feat = self.decode_head(inputs=x, feat=True)
        return feat

