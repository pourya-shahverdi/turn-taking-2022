from turngpt.eval import TurnGPTEval, turns_to_turngpt_tensors, load
from turngpt.turngpt_utils import remove_punctuation_capitalization
import torch
import csv
from pathlib import Path

def turns_to_turngpt_tensors_and_reverse(turns, tokenizer, explicit_turn_shift=True):
    assert isinstance(turns, list), "turns must be a list of strings"
    turns = [remove_punctuation_capitalization(text) for text in turns]
    sp1_idx, sp2_idx = (
        tokenizer.convert_tokens_to_ids("<speaker1>"),
        tokenizer.convert_tokens_to_ids("<speaker2>"),
    )
    input_ids, speaker_ids = [], []
    word_id = []
    word_count = 0
    for i, t in enumerate(turns):
        toks = []
        ids = []
        for word in t.split(' '):    
            word_tokens = tokenizer.encode(' ' + word)
            for tok in word_tokens:
                toks.append(tok)
                ids.append(word_count)
            if len(word_tokens) > 0:
                word_count += 1
            
        print(toks)
        print(word_id)
        if i % 2 == 0:
            cur_speaker = sp1_idx
        else:
            cur_speaker = sp2_idx
        if explicit_turn_shift:
            input_ids.append(cur_speaker)
            speaker_ids.append(cur_speaker)
            word_id.append(None)
        input_ids += toks
        speaker_ids += [cur_speaker] * len(toks)
        word_id += ids
    assert len(speaker_ids) == len(input_ids)
    return (
        torch.tensor(input_ids).unsqueeze(0),
        torch.tensor(speaker_ids).unsqueeze(0),
        word_id
    )

def main():
    model, dm, args = load()
    evaluation_model = TurnGPTEval(model, dm.tokenizer)

    if torch.cuda.is_available():
        evaluation_model = evaluation_model.to("cuda")

    # turns = [
    #     " yesterday we met in the park",
    #     " okay when will you meet again",
    #     " tomorrow",
    #     "",
    # ]

    # input_ids, speaker_ids, word_ids = turns_to_turngpt_tensors_and_reverse(
    #     turns, dm.tokenizer, explicit_turn_shift=True
    # )

    src_dir = Path('/home/alex/turn-taking/data/merged_text')
    dst_dir = Path('/home/alex/turn-taking/data/scored_text')
    for src_path in src_dir.glob('**/*'):
        dst_path = dst_dir / src_path.name
        with open(src_path) as src, open(dst_path, 'w') as dst:
            in_data = csv.reader(src)
            out_data = csv.writer(dst)
            current_speaker = 0
            turns = []
            word_scores = []
            lines = []
            for line in in_data:
                lines.append(line)
                if line[3] != '':
                    speaker = 1
                    word = line[3]
                else:
                    speaker = 2
                    word = line[4]
                if current_speaker != speaker:
                    current_speaker = speaker
                    turns.append('')
                turns[-1] += (' ' + word)
                word_scores.append(0)
            turns.append('')
            input_ids, speaker_ids, word_ids = turns_to_turngpt_tensors_and_reverse(
                turns, dm.tokenizer, explicit_turn_shift=True
            )
            out = evaluation_model.trp(input_ids.to(evaluation_model.device), speaker_ids.to(evaluation_model.device))
            trp = out["trp"]
            for i, score in enumerate(trp[0]):
                word_idx = word_ids[i]
                if word_idx is not None:
                    word_scores[word_idx] = score.item()
            for i, line in enumerate(lines):
                line.append(word_scores[i])
                out_data.writerow(line)



if __name__ == '__main__':
    main()