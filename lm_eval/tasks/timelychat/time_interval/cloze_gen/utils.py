import random

import datasets


def process_docs_instruct_without_gold_response(dataset: datasets.Dataset):
    def _helper(doc):
        doc["choices"] = [doc["time_elapsed"].lower(), *list(map(lambda x: x.lower(), doc["negative_answers"]))]
        random.shuffle(doc["choices"])
        answer_idx = doc["choices"].index(doc["time_elapsed"].lower())
        answer_map = {0: "A", 1: "B", 2: "C", 3: "D"}
        doc["gold"] = answer_map[answer_idx]

        context = "\n".join([f"{speaker}: {utt}" for speaker, utt in zip(doc["speaker_list"], doc["context"])])
        query = (
            "### Dialogue context ###\n"
            f"{context}\n\n"
            f"Question: How much time is needed for {doc['target_speaker']} to respond?\n"
            f"A. {doc['choices'][0]}\n"
            f"B. {doc['choices'][1]}\n"
            f"C. {doc['choices'][2]}\n"
            f"D. {doc['choices'][3]}\n"
            "Answer:"
        )
        doc["query"] = query

        return doc

    return dataset.map(_helper)


def process_docs_instruct_without_gold_response_cot(dataset: datasets.Dataset):
    def _helper(doc):
        doc["choices"] = [doc["time_elapsed"].lower(), *list(map(lambda x: x.lower(), doc["negative_answers"]))]
        random.shuffle(doc["choices"])
        answer_idx = doc["choices"].index(doc["time_elapsed"].lower())
        answer_map = {0: "A", 1: "B", 2: "C", 3: "D"}
        doc["gold"] = answer_map[answer_idx]

        context = "\n".join([f"{speaker}: {utt}" for speaker, utt in zip(doc["speaker_list"], doc["context"])])
        query = (
            "### Dialogue context ###\n"
            f"{context}\n\n"
            f"Question: How much time is needed for {doc['target_speaker']} to respond?\n"
            f"A. {doc['choices'][0]}\n"
            f"B. {doc['choices'][1]}\n"
            f"C. {doc['choices'][2]}\n"
            f"D. {doc['choices'][3]}\n"
            "Answer: Let's think step by step."
        )
        doc["query"] = query

        return doc

    return dataset.map(_helper)


def process_docs_instruct_with_gold_response(dataset: datasets.Dataset):
    def _helper(doc):
        doc["choices"] = [doc["time_elapsed"].lower(), *list(map(lambda x: x.lower(), doc["negative_answers"]))]
        random.shuffle(doc["choices"])
        answer_idx = doc["choices"].index(doc["time_elapsed"].lower())
        answer_map = {0: "A", 1: "B", 2: "C", 3: "D"}
        doc["gold"] = answer_map[answer_idx]

        context = "\n".join([f"{speaker}: {utt}" for speaker, utt in zip(doc["speaker_list"], doc["context"])])
        query = (
            "### Dialogue context ###\n"
            f"{context}\n\n"
            "### Target response ###\n"
            f"{doc['target_speaker']}: {doc['timely_response']}\n\n"
            f"Question: How much time is needed for {doc['target_speaker']} to respond with the target response?\n"
            f"A. {doc['choices'][0]}\n"
            f"B. {doc['choices'][1]}\n"
            f"C. {doc['choices'][2]}\n"
            f"D. {doc['choices'][3]}\n"
            "Answer:"
        )
        doc["query"] = query

        return doc

    return dataset.map(_helper)


def process_docs_instruct_with_gold_response_cot(dataset: datasets.Dataset):
    def _helper(doc):
        doc["choices"] = [doc["time_elapsed"].lower(), *list(map(lambda x: x.lower(), doc["negative_answers"]))]
        random.shuffle(doc["choices"])
        answer_idx = doc["choices"].index(doc["time_elapsed"].lower())
        answer_map = {0: "A", 1: "B", 2: "C", 3: "D"}
        doc["gold"] = answer_map[answer_idx]

        context = "\n".join([f"{speaker}: {utt}" for speaker, utt in zip(doc["speaker_list"], doc["context"])])
        query = (
            "### Dialogue context ###\n"
            f"{context}\n\n"
            "### Target response ###\n"
            f"{doc['target_speaker']}: {doc['timely_response']}\n\n"
            f"Question: How much time is needed for {doc['target_speaker']} to respond with the target response?\n"
            f"A. {doc['choices'][0]}\n"
            f"B. {doc['choices'][1]}\n"
            f"C. {doc['choices'][2]}\n"
            f"D. {doc['choices'][3]}\n"
            "Answer: Let's think step by step."
        )
        doc["query"] = query

        return doc

    return dataset.map(_helper)