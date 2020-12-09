import torch
import torch.nn as nn
from transformers import BartModel

class ParameterGenerator(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.encoder = BartModel.from_pretrained('facebook/bart-base')
        self.encoder.eval()

        self.decoders = nn.ModuleList([
            nn.Linear(768, 768*32*2+768+32) for _ in range(12)
        ])

    def encode(self, input_ids, attention_mask=None, encoder_outputs=None,
            decoder_input_ids=None, decoder_attention_mask=None, decoder_cached_states=None,
            use_cache=False, is_training=False):

        with torch.no_grad():
            outputs = self.encoder(
                input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                encoder_outputs=encoder_outputs,
            )

        x = outputs[0]  # last hidden state
        eos_mask = input_ids.eq(self.config.eos_token_id)
        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        sentence_representation = x[eos_mask, :].view(x.size(0), -1, x.size(-1))[:, -1, :]
        
        print(sentence_representation.size())
        return sentence_representation

    def decode(self, sr):
        return [one_decoder(sr) for one_decoder in self.decoders]

    def forward(self, input_ids, attention_mask=None, encoder_outputs=None,
            decoder_input_ids=None, decoder_attention_mask=None, decoder_cached_states=None,
            use_cache=False, is_training=False):
        
        h = self.encode(input_ids, attention_mask, encoder_outputs, decoder_input_ids,
                        decoder_attention_mask, decoder_cached_states, use_cache, is_training)

        params = self.decode(h)

        return params

class GrowingBart(nn.Module):
    def __init__(self, model, meta_model, config):
        super().__init__()

        self.config = config
        self.model = model
        self.meta_model = meta_model

    def forward(self, rel_ids, rel_masks, input_ids, input_masks, output_ids, output_masks, is_training=False):
        # generate adapter parameters using task descriptions
        generated_params = self.meta_model(rel_ids, attention_mask=rel_masks)

        # apply the parameters to the adapters
        self.apply_params_to_adapters(generated_params)
        
        # use the adapted model to make zero-shot inference
        ret = self.model(input_ids, attention_mask=input_masks,
                    decoder_input_ids=output_ids,
                    decoder_attention_mask=output_masks,
                    is_training=is_training
        )

        return ret

    def apply_params_to_adapters(self, generated_params):
        encoder_params, decoder_params = generated_params[:6], generated_params[6:] 
        for p, encoder_layer in zip(encoder_params, self.model.encoders()):
            dw, uw, db, ub = p[0][0:768*32], p[0][768*32:768*32*2], p[0][768*32*2:768*32*2+32], p[0][768*32*2+32:]
            encoder_layer.adapter_down_weight = dw.view(768, 32)
            encoder_layer.adapter_down_bias = db.view(32)
            encoder_layer.adapter_up_weight = uw.view(32, 768)
            encoder_layer.adapter_up_bias = ub.view(768)

        for p, decoder_layer in zip(decoder_params, self.model.decoders()):
            dw, uw, db, ub = p[0][0:768*32], p[0][768*32:768*32*2], p[0][768*32*2:768*32*2+32], p[0][768*32*2+32:]
            decoder_layer.adapter_down_weight = dw.view(768, 32)
            decoder_layer.adapter_down_bias = db.view(32)
            decoder_layer.adapter_up_weight = uw.view(32, 768)
            decoder_layer.adapter_up_bias = ub.view(768)
        
