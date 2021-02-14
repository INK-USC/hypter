import os
import numpy as np
import torch

from transformers import BartTokenizer, BartConfig
from transformers import AdamW, get_linear_schedule_with_warmup

from bart_with_adapter import BartWithAdapterConfig, MyBartWithAdapter
from growing_bart import ParameterGenerator, GrowingBart

from dataset import MyDatasetCollection
from bart import MyBart
from utils import freeze_embeds, trim_batch

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
        config = BartWithAdapterConfig.from_pretrained(args.model)
        config.adapter_dim = args.adapter_dim
        config.adapt_layer_norm = args.adapt_layer_norm
        config.unfreeze_hyper_encoder = args.unfreeze_hyper_encoder
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
            logger.info("Loading checkpoint from {}".format(args.checkpoint))

        else:
            bart_old = MyBart.from_pretrained(args.model)
            bart.model.load_state_dict(bart_old.model.state_dict(), strict=True)

        generator = ParameterGenerator(config)
        model = GrowingBart(bart, generator, config)

        if args.freeze_embeds:
            logger.info("Freezing embeddings")
            freeze_embeds(bart)

        if args.n_gpu>1:
            model = torch.nn.DataParallel(model)

        if torch.cuda.is_available():
            model.to(torch.device("cuda"))

        no_decay = ['bias', 'LayerNorm.weight']
        if args.unfreeze_hyper_encoder:
            optimizer_grouped_parameters = [
                {'params': [p for n, p in model.meta_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
                {'params': [p for n, p in model.meta_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

            num_parameters = sum(p.numel() for p in model.meta_model.parameters() if p.requires_grad)
            logger.info("#Params: {}".format(num_parameters))
        else:
            optimizer_grouped_parameters = [
                {'params': [p for n, p in model.meta_model.decoders.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
                {'params': [p for n, p in model.meta_model.decoders.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler =  get_linear_schedule_with_warmup(optimizer,
                                        num_warmup_steps=args.warmup_steps,
                                        num_training_steps=args.total_steps)

        train(args, logger, model, train_data, dev_data, optimizer, scheduler)

    if args.do_predict:
        checkpoint = os.path.join(args.output_dir, args.predict_checkpoint)
        def convert_to_single_gpu(state_dict):
            def _convert(key):
                if key.startswith('module.'):
                    return key[7:]
                return key
            return {_convert(key):value for key, value in state_dict.items()}
        
        config = BartWithAdapterConfig.from_pretrained(args.model)
        config.adapter_dim = args.adapter_dim
        bart = MyBartWithAdapter(config)
        generator = ParameterGenerator(config)
        model = GrowingBart(bart, generator, config)
        
        model.load_state_dict(convert_to_single_gpu(torch.load(checkpoint)), strict=False)

        logger.info("Loading checkpoint from {}".format(checkpoint))
        if torch.cuda.is_available():
            model.to(torch.device("cuda"))

        model.eval()
        score = inference(model, dev_data, save_predictions=True, verbose=True)
        logger.info("%s on %s data: %s" % (dev_data.metric, dev_data.data_type, score))

def train(args, logger, model, train_data, dev_data, optimizer, scheduler):

    model.train()
    global_step = 0
    train_losses = []
    best_accuracy = (-1.0, -1.0, -1.0) if args.dataset == "zest_grouped" else -1.0
    stop_training = False


    # curr_em = inference(model if args.n_gpu==1 else model.module, dev_data)
    # logger.info("[Before Training] %s %s" % (
    #         dev_data.metric,
    #         curr_em))

    logger.info("Starting training!")

    model.model.backup_layer_norm_parameters()

    for epoch in range(int(args.num_train_epochs)):
        for batch in tqdm(train_data.dataloader, desc="Epoch {}".format(epoch)):
            global_step += 1

            if torch.cuda.is_available():
                batch = [b.to(torch.device("cuda")) for b in batch[0]]

            rel_ids, rel_masks = batch[0].unsqueeze(0), batch[1].unsqueeze(0)
            input_ids, input_masks = batch[2], batch[3]
            output_ids, output_masks = batch[4], batch[5]

            pad_token_id = train_data.tokenizer.pad_token_id
            rel_ids, rel_masks = trim_batch(rel_ids, pad_token_id, rel_masks)
            input_ids, input_masks = trim_batch(input_ids, pad_token_id, input_masks)
            output_ids, output_masks = trim_batch(output_ids, pad_token_id, output_masks)

            loss = model.forward(rel_ids=rel_ids,
                                rel_masks=rel_masks,
                                input_ids=input_ids,
                                input_masks=input_masks,
                                output_ids=output_ids,
                                output_masks=output_masks,
                                is_training=True)

            train_losses.append(loss.detach().cpu())
            loss.backward()

            model.model.restore_layer_norm_parameters()

            if global_step % args.gradient_accumulation_steps == 0:
                # for p in model.meta_model.decoders.parameters():
                    # print(p)
                    # print(p.grad)
                    # break
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
                # for p in model.meta_model.decoders.parameters():
                #     print()
                #     print(p)
                #     print(p.grad)
                #     break
                
                optimizer.step()    # We have accumulated enough gradients
                scheduler.step()
                model.zero_grad()

                

                # print(model.meta_model.decoders[-1].linear1.weight)

            if global_step % args.eval_period == 0:
                model.eval()
                # curr_em = 0.0
                curr_em = inference(model if args.n_gpu==1 else model.module, dev_data, save_predictions=True)
                logger.info("Step %d Train loss %.2f %s %s on epoch=%d" % (
                        global_step,
                        np.mean(train_losses),
                        dev_data.metric,
                        curr_em,
                        epoch))
                train_losses = []
                if best_accuracy < curr_em:
                    model_state_dict = {k:v.cpu() for (k, v) in model.state_dict().items()}
                    torch.save(model_state_dict, os.path.join(args.output_dir, "best-model.pt"))
                    logger.info("Saving model with best %s: %s -> %s on epoch=%d, global_step=%d" % \
                            (dev_data.metric, best_accuracy, curr_em, epoch, global_step))
                    best_accuracy = curr_em
                    wait_step = 0
                    stop_training = False
                else:
                    wait_step += 1
                    if wait_step >= args.wait_step:
                        stop_training = True
                        break
                model.train()

        if stop_training:
            break

    model_state_dict = {k:v.cpu() for (k, v) in model.state_dict().items()}
    torch.save(model_state_dict, os.path.join(args.output_dir, "last-model.pt"))


def inference(model, dev_data, save_predictions=False, verbose=False):
    predictions = []
    bos_token_id = dev_data.tokenizer.bos_token_id
    for idx, batch in enumerate(dev_data.dataloader.inference_dataloader()):

        if torch.cuda.is_available():
            batch = [b.to(torch.device("cuda")) for b in batch]

        pad_token_id = dev_data.tokenizer.pad_token_id
        batch[0], batch[1] = trim_batch(batch[0].unsqueeze(0), pad_token_id, batch[1].unsqueeze(0))
        batch[2], batch[3] = trim_batch(batch[2], pad_token_id, batch[3])

        with torch.no_grad():
            model.set_relation(batch[0], batch[1])

            outputs = model.model.generate(input_ids=batch[2],
                                    attention_mask=batch[3],
                                    num_beams=dev_data.args.num_beams,
                                    max_length=dev_data.args.max_output_length,
                                    decoder_start_token_id=model.config.bos_token_id,
                                    early_stopping=dev_data.gen_early_stop,)
        for input_, output in zip(batch[2], outputs):
            pred = dev_data.decode(output)
            predictions.append(pred)

    if save_predictions:
        dev_data.save_predictions(predictions)

    return dev_data.evaluate(predictions, verbose=verbose)