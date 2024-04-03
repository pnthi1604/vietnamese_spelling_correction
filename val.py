import torch
from tqdm import tqdm
from .beam_search import beam_search
from .utils import calc_bleu_score, create_src_mask, calc_f_beta, calc_recall, calc_precision
from torch.nn.utils.rnn import pad_sequence

def validation(model, config, tokenizer_src, tokenizer_tgt, validation_dataloader, epoch, beam_size, have_test=False):
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
        src = batch['encoder_input'].to(device) # (b, seq_len)
        src_mask = create_src_mask(src, tokenizer_src.token_to_id("[PAD]"), device) # (B, 1, 1, seq_len)
        
        label_ids = batch["label"][0].to(device)

        src_text = batch['src_text'][0]
        tgt_text = batch['tgt_text'][0]

        model_out = beam_search(model=model,
                                config=config,
                                beam_size=beam_size,
                                tokenizer_src=tokenizer_src,
                                tokenizer_tgt=tokenizer_tgt,
                                src=src,
                                src_mask=src_mask)
        
        pred_ids = model_out.detach().cpu().numpy()
        pred_text = tokenizer_tgt.decode(pred_ids)

        padding = pad_sequence([torch.tensor(label_ids).to(device), torch.tensor(pred_ids).to(device)], padding_value=pad_index, batch_first=True)
        label_ids = padding[0]
        pred_ids = padding[1]
        
        labels.append(label_ids)
        preds.append(pred_ids)

        source_texts.append(tokenizer_src.encode(src_text).tokens)
        expected.append([tokenizer_tgt.encode(tgt_text).tokens])
        predicted.append(tokenizer_tgt.encode(pred_text).tokens)

        count += 1
        
        if count % 20 == 0 and not have_test:
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
            # print(f"{recall = }")
            # print(f"{precision = }")
            # print(f"{f_05 = }")

    labels = torch.cat(labels, dim=0)
    preds = torch.cat(preds, dim=0)

    recall = calc_recall(preds=preds, target=labels, tgt_vocab_size=tgt_vocab_size, pad_index=pad_index, device=device)
    precision = calc_precision(preds=preds, target=labels, tgt_vocab_size=tgt_vocab_size, pad_index=pad_index, device=device)
    f_05 = calc_f_beta(preds=preds, target=labels, beta=0.5, tgt_vocab_size=tgt_vocab_size, pad_index=pad_index, device=device)

    scores_corpus = calc_bleu_score(refs=expected,
                                cands=predicted)
    
    return scores_corpus, recall.item(), precision.item(), f_05.item()