from model.fam import ConditionedMaskDecoder,TwoWayTransformer, FilterAnything, ImageEncoderViT, T5Encoder
from timm.models.registry import register_model
from functools import partial
import torch



@register_model
def sam_backbone_base(pretrained_ckpt = "/mnt/lustre/zhengjinliang/.cache/torch/hub/checkpoints/sam_vit_b_01ec64.pth",**kwargs):
    model = ImageEncoderViT(
            depth=12,
            embed_dim=768,
            img_size=1024,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=12,
            patch_size=16,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=[2, 5, 8, 11],
            window_size=14,
            out_chans=256,
        )
    ckpt = torch.load(pretrained_ckpt, map_location="cpu")
    backbone_ckpt = {}
    for key, value in ckpt.items():
        if 'image_encoder.' in key:
            backbone_ckpt[key.replace('image_encoder.', '')] = value
    msg = model.load_state_dict(backbone_ckpt)
    print(msg)
    return model
    
@register_model
def t5_base(name = 'google-t5/t5-base', device = 'cuda', **kwargs):
    return  T5Encoder(name, device)


@register_model
def decoder_base(**kwargs):
    return ConditionedMaskDecoder(
        transformer_dim=256,
        num_task=2,
        transformer=TwoWayTransformer(
            depth=3,
            embedding_dim=256,
            mlp_dim=2048,
            num_heads=8,
        )
    )

@register_model
def decoder_large(**kwargs):
    return ConditionedMaskDecoder(
        transformer_dim=256,
        num_task=2,
        transformer=TwoWayTransformer(
            depth=6,
            embedding_dim=256,
            mlp_dim=2048,
            num_heads=8,
        )
    )



