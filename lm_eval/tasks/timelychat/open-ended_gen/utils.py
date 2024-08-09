from typing import Dict, List, Union


def doc_to_text_instruct(doc: Dict[str, Union[str, List[str]]]) -> str:
    context = "\n".join([f"{speaker}: {utt}" for speaker, utt in zip(doc["speaker_list"], doc["context"])])

    # Refer to Table 12 of https://arxiv.org/abs/2402.13211
    query = (
        "### Dialogue context ###\n"
        f"{context}\n\n"
        "### Time elapsed ###\n"
        f"{doc['time_elapsed']}\n\n"
        "### Next response ###\n"
        f"{doc['target_speaker']}:"
    )

    return query


def doc_to_text_rebot(doc: Dict[str, Union[str, List[str]]]) -> str:
    """
    The dialogue format used in ReBot (https://arxiv.org/pdf/2310.13420).
    ex) <relationship> r <time interval> summary <user> u1 <bot> c1 <user> ... <bot> cn
    """
    query = " ".join([f"<{speaker}> {utt}" for speaker, utt in zip(doc["speaker_list"], doc["context"])])
    query += f" <{doc['time_elapsed'].capitalize()} after> <{doc['target_speaker']}>"

    return query