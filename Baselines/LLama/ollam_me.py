from llama_index.llms import Ollama
import json
import argparse
import jsonlines
from tqdm import tqdm
import os

def process_answers(file):
    # with open(question_file_path, 'r') as file:
    lines = file.readlines()
    answers_list = []

    for line in lines:
        # Split the line into answers using newline character
        answers = line.strip().split('\n')
        # Append the answers to the list
        answers_list.append(answers)

    return answers_list


def save_to_json(answers_list, json_file_path):
    with open(json_file_path, 'w') as json_file:
        json.dump(answers_list, json_file)



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_filename",
        default="/home/hi9115/New-baselines/LLAMA/QBLINK/kilt_format_test.json",
        type=str,
        help="Dataset to be used to evaluate the in-context learning llama model."
    )
    parser.add_argument(
        "--output_path",
        default="/home/hi9115/New-baselines/LLAMA/QBLINK",
        type=str,
        help="Path where to save the results",
    )
    args = parser.parse_args()
    # Open the JSON file
    # with open(args.input_filename, 'r') as file:
    #     data = json.load(file)

    ## initialize the model
    llama2 = Ollama(model="new-prompt")

    ## loop over the dataset and get the predictions
    # llm_answers = []
    # for i in range(0, len(data), 1):
    #     print(i)
    #     llm_says = llama2.complete(data[i]['question']).text
    #     item = {
    #         'id': i,
    #         'question': data[i]['question'],
    #         'answers': [e for e in llm_says.split('\n')[e]],
    #     }
    #     llm_answers.append(item)
    with jsonlines.open(args.input_filename) as f:
        dataset = [e for e in f]

    total = len(dataset)
    correct_1 = 0
    correct_10 = 0
    miss = 0
    mrr = 0

    with jsonlines.open('/home/hi9115/New-baselines/LLAMA/QBLINK/mapped_question_test.json') as f:
        mapped_question_test = [e for e in f]


    with open(os.path.join(args.output_path, "results_for_fine_tuned_new.json", ), "w", ) as output:
        pbar = tqdm(dataset, total=total)
        for _, dialog in enumerate(pbar):
            for map_id in mapped_question_test:
                if map_id['id'] == dialog['id']:

                    q_id = dialog["id"]
                    answer = dialog["output"][0]["provenance"][0]['title']
                    question = dialog["input"]
                    # predicted_answers = model.sample(question, beam=10)
                    predicted_answers = llama2.complete(question).text
                    # result = [item for item in predicted_answers.split('\n', 9)]
                    # result = [item["text"] for item in predicted_answers]
                    result = predicted_answers.split('\n', 9)
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

        # answers = process_answers(llm_says)
        # output_json_path = "/home/hi9115/New-baselines/LLAMA/QBLINK/"
        # # Save the answers list to a JSON file
        # save_to_json(answers, output_json_path)

        # print(llm_says)
    # sentence = "Name this philosophical book about epistemology which discusses the \u201cmissing shade of blue\u201d in its section \u201cOf the Origin of Ideas.\u201d It is a revision of its author\u2019s earlier work, A Treatise of Human Nature.",
    # llm_says = llama2.complete(data[0]['question'])
    # print(llm_says)

# str= "Mona \n Mozhdeh \n Marjan \n Mozhgan"
# print(str)