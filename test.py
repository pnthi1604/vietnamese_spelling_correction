from .train import get_model
from .val import validation
from .config import weights_file_path
import torch
from torch.utils.tensorboard import SummaryWriter
from .utils import get_tokenizer, set_seed
from .pre_dataset import load_data, get_dataloader_test, get_dataloader_custom_test_dataset

def test_model(config):
    set_seed()
    device = config["device"]
    device = torch.device(device)

    # load data and tokenizer
    dataset = load_data(config)
    tokenizer_src, tokenizer_tgt = get_tokenizer(config=config,
                                                 dataset=dataset)
    
    src_vocab_size = tokenizer_src.get_vocab_size()
    tgt_vocab_size = tokenizer_tgt.get_vocab_size()
    pad_id_token = tokenizer_tgt.token_to_id("[PAD]")

    # get dataloader
    test_dataloader = get_dataloader_test(config=config,
                                          dataset=dataset,
                                          tokenizer_src=tokenizer_src,
                                          tokenizer_tgt=tokenizer_tgt
    )

    model = get_model(config=config,
                        device=device,
                        src_vocab_size=src_vocab_size,
                        tgt_vocab_size=tgt_vocab_size,
                        pad_id_token=pad_id_token)
    
    writer = SummaryWriter(config["experiment_name"])
    model_filenames = weights_file_path(config=config)
    model_filename = model_filenames[-1]

    state = torch.load(model_filename)
    model.load_state_dict(state['model_state_dict'])

    bleu_results = {} #[bleu <1, 2, 3, 4>][beam_size]
    for i in range(0, 4):
        bleu_results[f"Test_model_Bleu_{i + 1}"] = {}

    max_beam = config["beam_test"]

    for beam_size in range(1, max_beam + 1):
        scores_corpus = validation(model=model,
                config=config,
                tokenizer_src=tokenizer_src,
                tokenizer_tgt=tokenizer_tgt,
                validation_dataloader=test_dataloader,
                epoch=0,
                beam_size=beam_size)
        
        for i in range(len(scores_corpus)):
            bleu_results[f"Test_model_Bleu_{i + 1}"][f"Beam_size={beam_size}"] = scores_corpus[i]

    print()
    print(bleu_results)
    for i in range(0, 4):
        writer.add_scalars(f"Test_model_Bleu_{i + 1}", bleu_results[f"Test_model_Bleu_{i + 1}"], 0)
        writer.close()

def test_model_with_beam_size(config, beam_size):
    set_seed()
    device = config["device"]
    device = torch.device(device)

    # load data and tokenizer
    dataset = None
    if not config["custom_test_dataset"]:
        dataset = load_data(config)
    tokenizer_src, tokenizer_tgt = get_tokenizer(config=config,
                                                 dataset=dataset)
    
    src_vocab_size = tokenizer_src.get_vocab_size()
    tgt_vocab_size = tokenizer_tgt.get_vocab_size()
    pad_id_token = tokenizer_tgt.token_to_id("[PAD]")

    # get dataloader
    if not config["custom_test_dataset"]:
        test_dataloader = get_dataloader_test(config=config,
                                            tokenizer_src=tokenizer_src,
                                            tokenizer_tgt=tokenizer_tgt
        )
    else:
        test_dataloader = get_dataloader_custom_test_dataset(
            config=config,
            tokenizer_src=tokenizer_src,
            tokenizer_tgt=tokenizer_tgt
        )

    # print(f"{test_dataloader = }")

    model = get_model(config=config,
                        device=device,
                        src_vocab_size=src_vocab_size,
                        tgt_vocab_size=tgt_vocab_size,
                        pad_id_token=pad_id_token)
    
    writer = SummaryWriter(config["experiment_name"])
    model_filenames = weights_file_path(config=config)
    model_filename = model_filenames[-1]

    state = torch.load(model_filename)
    model.load_state_dict(state['model_state_dict'])

    bleu_results = {} #[bleu <1, 2, 3, 4>][beam_size]
    for i in range(0, 4):
        bleu_results[f"Test_model_Bleu_{i + 1}"] = {}

    scores_corpus, sum_recall_val, sum_precision_val, sum_f_05_val, sum_accuracy_val = validation(model=model,
            config=config,
            tokenizer_src=tokenizer_src,
            tokenizer_tgt=tokenizer_tgt,
            validation_dataloader=test_dataloader,
            epoch=0,
            beam_size=beam_size)
    
    for i in range(len(scores_corpus)):
        bleu_results[f"Test_model_Bleu_{i + 1}"][f"Beam_size={beam_size}"] = scores_corpus[i]

    print()
    print(bleu_results)
    for i in range(0, 4):
        writer.add_scalars(f"Test_model_Bleu_{i + 1}", bleu_results[f"Test_model_Bleu_{i + 1}"], 0)
        writer.close()

    writer.add_scalar("Test_model_Recall", sum_recall_val, 0)
    writer.add_scalar("Test_model_Precision", sum_precision_val, 0)
    writer.add_scalar("Test_model_Accuracy", sum_accuracy_val, 0)
    writer.add_scalar("Test_model_f_05_score", sum_f_05_val, 0)

    print(f"{sum_recall_val = }")
    print(f"{sum_precision_val = }")
    print(f"{sum_accuracy_val = }")
    print(f"{sum_f_05_val = }")