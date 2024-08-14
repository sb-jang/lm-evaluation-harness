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


def doc_to_text_msc(doc: Dict[str, Union[str, List[str]]]) -> str:
    """
    The dialogue format used in MSC (https://aclanthology.org/2022.acl-long.356/).
    ex) p1: u1\np2: u2\n ... \ntime: t
    """
    speaker_mapping = {"A": "p1", "B": "p2"}
    query = ""
    for speaker, utt in zip(doc["speaker_list"], doc["context"]):
        if speaker in speaker_mapping.keys():
            speaker = speaker_mapping[speaker]
        query += f"{speaker}: {utt}\n"

    target_speaker = speaker_mapping[doc["target_speaker"]] if doc["target_speaker"] in speaker_mapping.keys() else doc["target_speaker"]
    query += f"time: {doc['time_elapsed']}\n{target_speaker}:"

    return query