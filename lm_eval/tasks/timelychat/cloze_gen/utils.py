from typing import Dict, List, Union


def doc_to_text_instruct(doc: Dict[str, Union[str, List[str]]]) -> str:
    context = "\n".join([f"{speaker}: {utt}" for speaker, utt in zip(doc["speaker_list"], doc["context"])])

    query = (
        "### Dialogue context ###\n"
        f"{context}\n\n"
        "### Time elapsed ###\n"
        f"{doc['time_elapsed']}\n\n"
        f"### Next response ###\n"
        f"1. {doc['target_speaker']}: {doc['timely_response']}\n"
        f"2. {doc['target_speaker']}: {doc['untimely_response']}\n\n"
        "Answer:"
    )

    return query