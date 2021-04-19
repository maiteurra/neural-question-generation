from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge


# BART input shift
def shift_tokens_right(input_ids, pad_token_id):
    """ Shift input ids one token to the right, and wrap the last non pad token (usually <eos>).
        This is taken directly from modeling_bart.py
    """
    prev_output_tokens = input_ids.clone()
    index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = input_ids[:, :-1]
    return prev_output_tokens


# Print predictions in files
def update_files(conf, predictions, targets, inputs):
    with open(conf.testing.predictions_path, 'a', encoding='utf-8') as prediction_file:
        prediction_file.writelines(predictions)
    with open(conf.testing.target_path, 'a', encoding='utf-8') as target_file:
        target_file.writelines(targets)
    with open(conf.testing.results_path, 'a', encoding='utf-8') as results_file:
        for i, p, t in zip(inputs, predictions, targets):
            results_file.write(i + p + t + "\n")


def evaluation_metrics(conf):
    with open(conf.testing.predictions_path, 'r', encoding='utf-8') as file:
        predictions = {i: [line] for i, line in enumerate(file.readlines())}
    with open(conf.testing.target_path, 'r', encoding='utf-8') as file:
        references = {i: [line] for i, line in enumerate(file.readlines())}

    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Rouge(), "ROUGE_L"),
        # (Meteor(), "METEOR"),
    ]

    results = {}
    with open(conf.testing.evaluation_path, 'w') as file:
        for scorer, method in scorers:
            score, scores = scorer.compute_score(references, predictions)
            print(method, file=file)
            print(score, file=file)
            if isinstance(method, list):
                for idx, m in enumerate(method):
                    results[m] = score[idx]
            else:
                results[method] = score
    return results
