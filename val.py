import torch
from tqdm import tqdm
from .beam_search import beam_search
from .utils import calc_bleu_score, create_src_mask, calc_f_beta, calc_recall, calc_precision
from torch.nn.utils.rnn import pad_sequence

def validation(model, config, tokenizer_src, tokenizer_tgt, validation_dataloader, epoch, beam_size, num_example=5):
    with torch.no_grad():
        device = config["device"]

        source_texts = []
        expected = []
        predicted = []

        count = 0

        tgt_vocab_size = tokenizer_tgt.get_vocab_size()
        pad_index = tokenizer_tgt.token_to_id("[PAD]")

        labels = []
        preds = []

        batch_iterator = tqdm(validation_dataloader, desc=f"Validation Bleu Epoch {epoch:02d}")
        for batch in batch_iterator:
            #test
            # print(f"{batch = }")
            src = batch['encoder_input'].to(device) # (b, seq_len)
            src_mask = create_src_mask(src, tokenizer_src.token_to_id("[PAD]"), device) # (B, 1, 1, seq_len)

            src_text = batch['src_text'][0]
            tgt_text = batch['tgt_text'][0]

            pred_ids = beam_search(model=model,
                                    config=config,
                                    beam_size=beam_size,
                                    tokenizer_src=tokenizer_src,
                                    tokenizer_tgt=tokenizer_tgt,
                                    src=src,
                                    src_mask=src_mask)
            
            # print(f"{pred_ids = }")
            # print(f"{label_ids = }")
            pred_text = tokenizer_tgt.decode(pred_ids.detach().cpu().numpy())
            pred_ids = torch.tensor(tokenizer_tgt.encode(pred_text).ids, dtype=torch.int64).to(device)
            label_ids = torch.tensor(tokenizer_tgt.encode(tgt_text).ids, dtype=torch.int64).to(device)
            

            padding = pad_sequence([label_ids, pred_ids], padding_value=pad_index, batch_first=True)
            label_ids = padding[0]
            pred_ids = padding[1]
            
            labels.append(label_ids)
            preds.append(pred_ids)

            source_texts.append(tokenizer_src.encode(src_text).tokens)
            expected.append([tokenizer_tgt.encode(tgt_text).tokens])
            predicted.append(tokenizer_tgt.encode(pred_text).tokens)

            count += 1

            print_step = len(validation_dataloader) // num_example
            
            if count % print_step == 0:
                print()
                print(f"{f'SOURCE: ':>12}{src_text}")
                print(f"{f'TARGET: ':>12}{tgt_text}")
                print(f"{f'PREDICTED: ':>12}{pred_text}")
                print(f"{f'TOKENS TARGET: ':>12}{[tokenizer_tgt.encode(tgt_text).tokens]}")
                print(f"{f'TOKENS PREDICTED: ':>12}{tokenizer_tgt.encode(pred_text).tokens}")
                scores = calc_bleu_score(refs=[[tokenizer_tgt.encode(tgt_text).tokens]],
                                        cands=[tokenizer_tgt.encode(pred_text).tokens])
                print(f'BLEU OF SENTENCE {count}')
                for i in range(0, len(scores)):
                    print(f'BLEU_{i + 1}: {scores[i]}')
                
                print(f"{label_ids = }")
                print(f"{pred_ids = }")

                recall = calc_recall(preds=pred_ids, target=label_ids, tgt_vocab_size=tgt_vocab_size, pad_index=pad_index, device=device)
                precision = calc_precision(preds=pred_ids, target=label_ids, tgt_vocab_size=tgt_vocab_size, pad_index=pad_index, device=device)
                f_05 = calc_f_beta(preds=pred_ids, target=label_ids, beta=config["f_beta"], tgt_vocab_size=tgt_vocab_size, pad_index=pad_index, device=device)
            
                recall = recall.item()
                precision = precision.item()
                f_05 = f_05.item()
                print(f"{recall = }")
                print(f"{precision = }")
                print(f"{f_05 = }")

        labels = torch.cat(labels, dim=0)
        preds = torch.cat(preds, dim=0)

        recall = calc_recall(preds=preds, target=labels, tgt_vocab_size=tgt_vocab_size, pad_index=pad_index, device=device)
        precision = calc_precision(preds=preds, target=labels, tgt_vocab_size=tgt_vocab_size, pad_index=pad_index, device=device)
        f_05 = calc_f_beta(preds=preds, target=labels, beta=config["f_beta"], tgt_vocab_size=tgt_vocab_size, pad_index=pad_index, device=device)

        scores_corpus = calc_bleu_score(refs=expected,
                                    cands=predicted)
        
        recall = recall.item()
        precision = precision.item()
        f_05 = f_05.item()
        print(f"{recall = }")
        print(f"{precision = }")
        print(f"{f_05 = }")
        
        return scores_corpus, recall, precision, f_05