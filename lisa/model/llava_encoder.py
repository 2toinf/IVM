
import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from peft import get_peft_model, LoraConfig
from timm.models.layers import Mlp

class LLaVAEncoder(torch.nn.Module):
    def __init__(self, out_dim = 256 ,lora_r = 8, lora_alpha = 16, lora_dropout = 0.05, lora_target_modules = ("q_proj", "v_proj")):
        super().__init__()
        self.processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
        model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", 
                    local_files_only = True,
                    torch_dtype=torch.float16, 
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

    def forward(self, text_input, image_input, **kwargs):
        text_input = [f"[INST] <image>\n{t} [/INST]" for t in text_input]
        inputs = self.processor(text_input, image_input, return_tensors="pt", padding=True).to("cuda")
        # autoregressively complete prompt
        output = self.model(**inputs, output_hidden_states = True, return_dict = True)
        return self.text_proj(output.hidden_states[-1][:,-2:-1])





if __name__ == "__main__":
    from torchvision.transforms import functional as F
    from PIL import Image
    image = Image.open("/mnt/lustre/zhengjinliang/0.jpg").convert('RGB')

    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
    model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", 
                local_files_only = True,
                torch_dtype=torch.float16, 
                low_cpu_mem_usage=True,
                attn_implementation="flash_attention_2").to("cuda:0")
    text_input = ["Disribe this image", "is the red cup in the image? disribe its location"]
    image_input = F.to_tensor(image).unsqueeze(0).repeat(2, 1, 1, 1)

    
    prompt = [f"[INST] <image>\n{t} [/INST]" for t in text_input]

    inputs = processor(prompt, image_input, return_tensors="pt", padding=True).to("cuda:0")

    # autoregressively complete prompt
    output = model.generate(**inputs, max_new_tokens=100)

    print(processor.decode(output[0][0], skip_special_tokens=True))
    print(processor.decode(output[1][0], skip_special_tokens=True))
