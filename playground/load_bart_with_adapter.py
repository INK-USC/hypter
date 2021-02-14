# PYTHONPATH='.' python playground/load_bart_with_adapter.py 
import torch
from transformers import BartModel
from bart_with_adapter import BartWithAdapterConfig, MyBartWithAdapter
from growing_bart import ParameterGenerator, GrowingBart

def main():
    config = BartWithAdapterConfig.from_pretrained('facebook/bart-base')
    bart = MyBartWithAdapter(config)

    bart_old = BartModel.from_pretrained("facebook/bart-base")
    ret = bart.model.load_state_dict(bart_old.state_dict(), strict=False)

    print(ret)

def main2():
    config = BartWithAdapterConfig.from_pretrained('facebook/bart-base')
    bart = MyBartWithAdapter(config)

    bart_old = BartModel.from_pretrained("facebook/bart-base")
    bart.model.load_state_dict(bart_old.state_dict(), strict=False)

    config = BartWithAdapterConfig.from_pretrained('facebook/bart-base')
    # config.adapt_layer_norm = True
    generator = ParameterGenerator(config)

    output = generator(torch.tensor([[1,2,3]]))
    print(output)
    print(output[0].size())

    growingbart = GrowingBart(bart, generator, config)

    output = growingbart(torch.tensor([[4,1,3,4,3,5,6,3,2]]), torch.tensor([[1,1,1,1,1,1,1,1,1]]),
        torch.tensor([[4,1,3,4,3,5,6,3,2]]), torch.tensor([[1,1,1,1,1,1,1,1,1]]),
        torch.tensor([[4,1,3,4,3,5,6,3,2]]), torch.tensor([[1,1,1,1,1,1,1,1,1]]))

    print(output)
    
    loss = output[0].sum(-1).sum(-1).sum(-1)
    print(loss)
    loss.backward()

    # for n, p in generator.decoders.named_parameters():
    #     if p.requires_grad:
    #         print(n)
    #         print(p.grad)


if __name__ == "__main__":
    main2()