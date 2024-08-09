import random
from typing import Dict, List, Union

import datasets


def process_docs_instruct(dataset: datasets.Dataset):
    def _helper(doc):
        doc["choices"] = [doc["timely_response"], doc["untimely_response"]]
        random.shuffle(doc["choices"])
        doc["gold"] = doc["choices"].index(doc["timely_response"]) + 1

        context = "\n".join([f"{speaker}: {utt}" for speaker, utt in zip(doc["speaker_list"], doc["context"])])
        query = (
            "### Dialogue context ###\n"
            f"{context}\n\n"
            "### Time elapsed ###\n"
            f"{doc['time_elapsed']}\n\n"
            f"### Next response ###\n"
            f"1. {doc['target_speaker']}: {doc['choices'][0]}\n"
            f"2. {doc['target_speaker']}: {doc['choices'][1]}\n\n"
            "Answer:"
        )
        doc["query"] = query

        return doc

    return dataset.map(_helper)