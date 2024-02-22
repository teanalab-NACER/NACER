# for pytorch/fairseq
import argparse
import json
import os

import jsonlines
from tqdm import tqdm

from genre.fairseq_model import GENRE

model = GENRE.from_pretrained("/home/hi9115/New-baselines/GENRE/scripts_genre/models").eval()

# for huggingface/transformers
# from genre.hf_model import GENRE
# model = GENRE.from_pretrained("../models/hf_wikipage_retrieval").eval()


# sentences = ["Einstein was a German physicist."]
#
# print(model.sample(sentences))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_filename",
        default="/home/hi9115/New-baselines/GENRE/scripts_genre/qblink/kilt_format_test.json",
        type=str,
        help="Filename of the QBLINK test dataset",
    )
    parser.add_argument(
        "--output_path",
        default="/home/hi9115/New-baselines/GENRE/scripts_genre/qblink/",
        type=str,
        help="Path where to save the results",
    )

    args = parser.parse_args()

    # read the test dataset and store it in a list
    with jsonlines.open(args.input_filename) as f:
        dataset = [e for e in f]

    total = len(dataset)
    correct_1 = 0
    correct_10 = 0
    miss = 0
    mrr = 0

    with jsonlines.open('/home/hi9115/New-baselines/GENRE/scripts_genre/qblink/mapped_question_test.json') as f:
        mapped_question_test = [e for e in f]


    with open(os.path.join(args.output_path, "results_for_fine_tuned_new.json", ), "w", ) as output:
        pbar = tqdm(dataset, total=total)
        for _, dialog in enumerate(pbar):
            for map_id in mapped_question_test:
                if map_id['id'] == dialog['id']:

                    q_id = dialog["id"]
                    answer = dialog["output"][0]["provenance"][0]['title']
                    question = dialog["input"]
                    predicted_answers = model.sample(question, beam=10)
                    result = [item["text"] for item in predicted_answers]


                    # if answer in result:
                    #     result.split('\n')
                    if answer in result:
                        i = result.index(answer)
                        if i == 0:
                            correct_1 += 1
                            dialog["pos_pred"] = [i + 1, result]
                            print(f"First match for question: {q_id}, in dialog id: {dialog['id']}")
                        else:
                            correct_10 += 1
                            mrr += 1 / (i + 1)
                            dialog["pos_pred"] = [i + 1, result]
                            print(f"{i + 1}th match for question: {q_id}, Position of hit is: {i + 1}")
                    else:
                        miss += 1
                        dialog["pos_pred"] = [-1, result]

                    json.dump(dialog, output)
                    output.write("\n")

        output.close()

    print(f"Hits@1: {correct_1}")
    print(f"Recall@1: {correct_1 / total:.4f}")
    print(f"Hits@10: {correct_10}")
    print(f"Recall@10: {correct_10 / total:.4f}")
    print(f"MRR: {mrr / total:.4f}")
    print(f"Total: {total}")
    print(f"Missed: {miss}")

