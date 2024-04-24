
import torch
import transformers
from transformers import T5Tokenizer, T5EncoderModel, T5Model

T5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]

class T5Encoder(torch.nn.Module):
    def __init__(self, 
            model_path = '/mnt/lustre/zhengjinliang/.cache/huggingface/hub/models--google--t5-v1_1-base/snapshots/b5fc947a416ea3cb079532cb3c2bbadeb7f800fc',
            device = "cuda",
            max_length = 64) -> None:
        super().__init__()
        self.device = device
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5EncoderModel.from_pretrained(model_path)
        self.model.to(device)
        self.max_length = max_length

    @torch.no_grad()
    def forward(self, text):
        encoded = self.tokenizer.batch_encode_plus(
            text,
            return_tensors = "pt",
            padding = 'longest',
            max_length = self.max_length,
            truncation = True
        )

        input_ids = encoded.input_ids.to(self.device)
        attn_mask = encoded.attention_mask.to(self.device)
        output = self.model(input_ids = input_ids, attention_mask = attn_mask)
        encoded_text = output.last_hidden_state.detach()
        # attn_mask = attn_mask.bool()
        # encoded_text.masked_fill_(~attn_mask[..., None], 0.)
        # numer = encoded_text.sum(dim = -2)
        # denom = attn_mask.sum(dim = -1)[..., None]
        # numer.masked_fill_(denom == 0, 0.)
        # mean_encodings = numer / denom.clamp(min = 1e-3)

        return encoded_text


if __name__ == "__main__":
    model = T5Encoder(device="cpu")
    emb = model.embed_text(["test our models  and and and"])
    print(emb.shape)


