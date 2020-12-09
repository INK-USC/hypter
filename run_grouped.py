import os
import numpy as np
import torch

from transformers import BartTokenizer, BartConfig
from transformers import AdamW, get_linear_schedule_with_warmup

from bart_with_adapter import BartWithAdapterConfig, MyBartWithAdapter
from growing_bart import ParameterGenerator, GrowingBart

from dataset import MyDatasetCollection
from bart import MyBart
from utils import freeze_embeds

from tqdm import tqdm

def run(args, logger):
    assert args.dataset.endswith("grouped")
    
    tokenizer = BartTokenizer.from_pretrained(args.model)

    train_data = MyDatasetCollection(logger, args, args.train_file, True)
    dev_data = MyDatasetCollection(logger, args, args.predict_file, False)

    train_data.load_dataset(tokenizer)
    train_data.load_dataloader()

    dev_data.load_dataset(tokenizer)
    dev_data.load_dataloader()

    if args.do_train:
        config = BartWithAdapterConfig.from_pretrained('facebook/bart-base')
        bart = MyBartWithAdapter(config)

        if args.checkpoint is not None:
            def convert_to_single_gpu(state_dict):
                def _convert(key):
                    if key.startswith('module.'):
                        return key[7:]
                    return key
                return {_convert(key):value for key, value in state_dict.items()}
            bart_old = MyBart.from_pretrained(args.model,
                                           state_dict=convert_to_single_gpu(torch.load(args.checkpoint)))
            bart.model.load_state_dict(bart_old.model.state_dict(), strict=True)

        else:
            bart_old = MyBart.from_pretrained(args.model)
            bart.model.load_state_dict(bart_old.model.state_dict(), strict=True)

        generator = ParameterGenerator(config)
        model = GrowingBart(bart, generator, config)

        if args.freeze_embeds:
            logger.info("Freezing embeddings")
            freeze_embeds(model)

        if args.n_gpu>1:
            model = torch.nn.DataParallel(model)

        if torch.cuda.is_available():
            model.to(torch.device("cuda"))

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.meta_model.decoders.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.meta_model.decoders.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler =  get_linear_schedule_with_warmup(optimizer,
                                        num_warmup_steps=args.warmup_steps,
                                        num_training_steps=args.total_steps)

        train(args, logger, model, train_data, dev_data, optimizer, scheduler)



def train(args, logger, model, train_data, dev_data, optimizer, scheduler):

    model.train()
    global_step = 0
    train_losses = []
    best_accuracy = -1
    stop_training = False

    logger.info("Starting training!")

    for batch in tqdm(train_data.dataloader, desc="Epoch {}".format(0)):
        global_step += 1

        if torch.cuda.is_available():
            batch = [b.to(torch.device("cuda")) for b in batch[0]]

        rel_ids, rel_masks = batch[0].unsqueeze(0), batch[1].unsqueeze(0)
        input_ids, input_masks = batch[2], batch[3]
        output_ids, output_masks = batch[4], batch[5]

        loss = model.forward(rel_ids=rel_ids,
                            rel_masks=rel_masks,
                            input_ids=input_ids,
                            input_masks=input_masks,
                            output_ids=output_ids,
                            output_masks=output_masks,
                            is_training=True)

        train_losses.append(loss.detach().cpu())
        loss.backward()

        if global_step % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()    # We have accumulated enough gradients
            scheduler.step()
            model.zero_grad()

        print(loss)
        # break

def inference():
    pass