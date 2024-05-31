import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch
from peft import get_peft_model, LoraConfig
from timm.models.layers import Mlp
from PIL import Image

def apply_mask_to_rgb(image, mask, fill_color=(255, 255, 255)):
    """
    Applies a binary mask to an RGB PIL Image and fills non-masked areas with a given color.

    Parameters:
    - image: PIL Image in RGB format to which the mask will be applied.
    - mask: Tensor of shape (height, width) consisting of 0s and 1s.
    - fill_color: Tuple indicating the RGB values to use for filling non-masked areas.

    Returns:
    - PIL Image with the mask applied.
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    mask = mask.cpu()
    image_data = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
    image_data = image_data.view(image.size[1], image.size[0], 3)
    mask = mask.unsqueeze(-1).repeat(1, 1, 3)
    fill_tensor = torch.tensor(fill_color, dtype=torch.uint8).repeat(image.size[1], image.size[0], 1)
    masked_data = image_data * mask + fill_tensor * (1 - mask)
    result_image = Image.frombytes('RGB', image.size, masked_data.numpy().tobytes())
    return result_image


def llavaTextPreprocess(ins: str, role  = "policy"):
    if not isinstance(ins, str): raise RuntimeError
    if role == "policy":
        return f"USER: <image>\n{ins} \n ASSISTANT:"
    elif role == "discriminator":
        return f"USER: <image>\n{ins} \n Can you follow instruction above based on the given image? \n ASSISTANT:"
    else:
        raise NotImplementedError
    

class LLaVAEncoder(torch.nn.Module):
    def __init__(self, 
                llava_model = "llava-hf/llava-1.5-7b-hf",
                out_dim = 256 ,lora_r = 8, lora_alpha = 16, lora_dropout = 0.05, lora_target_modules = ("q_proj", "v_proj")):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained(llava_model)
        model = LlavaForConditionalGeneration.from_pretrained(llava_model, 
                    torch_dtype=torch.float16, 
                    local_files_only = True,
                    low_cpu_mem_usage=True,
                    attn_implementation="flash_attention_2") 
        lora_module_names = set()
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and 'language_model' in name and any([x in name for x in lora_target_modules]):
                lora_module_names.add(name)
        lora_module_names = sorted(list(lora_module_names))
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_module_names,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(
            model, lora_config
        )
        self.model.print_trainable_parameters()
        self.text_proj = Mlp(4096, 2048, out_dim)

    def forward(self, text_input, image, **kwargs):
        
        prompt = [llavaTextPreprocess(t) for t in text_input]
        inputs = self.processor(text=prompt, images=image, return_tensors="pt", padding=True).to("cuda", torch.float16)
        output = self.model(**inputs, output_hidden_states = True, return_dict = True)
        return self.text_proj(output.hidden_states[-1][:,-2:-1])




# class LLaVAEncoderDiscriminator(torch.nn.Module):
#     def __init__(self, 
#                 llava_model = "llava-hf/llava-1.5-7b-hf",
#                 out_dim = 256 ,lora_r = 8, lora_alpha = 16, lora_dropout = 0.05, lora_target_modules = ("q_proj", "v_proj")):
#         super().__init__()
#         self.processor = AutoProcessor.from_pretrained(llava_model)
#         model = LlavaForConditionalGeneration.from_pretrained(llava_model, 
#                     torch_dtype=torch.float16, 
#                     local_files_only = True,
#                     low_cpu_mem_usage=True,
#                     attn_implementation="flash_attention_2") 
#         lora_module_names = set()
#         for name, module in model.named_modules():
#             if isinstance(module, torch.nn.Linear) and 'language_model' in name and any([x in name for x in lora_target_modules]):
#                 lora_module_names.add(name)
#         lora_module_names = sorted(list(lora_module_names))
#         lora_config = LoraConfig(
#             r=lora_r,
#             lora_alpha=lora_alpha,
#             target_modules=lora_module_names,
#             lora_dropout=lora_dropout,
#             bias="none",
#             task_type="CAUSAL_LM",
#         )
#         self.model = get_peft_model(
#             model, lora_config
#         )
#         self.model.print_trainable_parameters()
#         self.text_proj = Mlp(4096, 2048, out_dim)
#         self.discriminator_head = Mlp(4096, 1024, 1)

#     def forward(self, text_input, image, mask, **kwargs):
#         masked_image = [apply_mask_to_rgb(img, m) for img, m in zip(image, mask)]
#         prompt = [llavaTextPreprocess(t, role = 'complex_policy') for t in text_input]
#         masked_prompt = [llavaTextPreprocess(t, role = 'discriminator') for t in text_input]
#         inputs = self.processor(text=prompt + masked_prompt, images=image + masked_image, return_tensors="pt", padding=True).to("cuda", torch.float16)
#         output = self.model(**inputs, output_hidden_states = True, return_dict = True)
#         generate_output, discriminator_output = torch.chunk(output.hidden_states[-1],2,0)
#         return self.text_proj(generate_output[:,-2:-1]), self.discriminator_head(discriminator_output[:,-1])



