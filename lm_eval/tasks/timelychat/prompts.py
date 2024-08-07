from typing import Dict, List, Union


SPEAKER_TOKEN = "<spk>"
TIME_TOKEN = "<time>"
UTTERANCE_TOKEN = "<utt>"


def doc_to_text_vanilla(doc: Dict[str, Union[str, List[str]]]) -> str:
    """
    Convert an example to a query prompt.
    Prompt is in a natural language form.
    ex)
    A: Hi, how are you?\n
    B: I'm good, how about you?\n
    (2 hours later)\n
    A:

    :param doc: example from the dataset
    structure: {
        "context": list of utterances (str),
        "speaker_list": list of speakers (str),
        "time_elapsed": time_elapsed (str),
        "target_speaker": str,
        "timely_response": str,
        "untimely_response": str,
        "narrative": str
    }
    :return: query prompt
    """
    query = "\n".join([f"{speaker}: {utt}" for speaker, utt in zip(doc["speaker_list"], doc["context"])])
    query += f"\n({doc['time_elapsed']} later)\n{doc['target_speaker']}:"

    return query


def doc_to_text_with_indicator_tokens(doc: Dict[str, Union[str, List[str]]]) -> str:
    """
    Prompt is in a fine-tuning form, where special tokens are prepended to each component (speaker id, time, utterance).
    ex)
    <spk> A: <utt> Hi, how are you?
    <spk> B: <time> 0 minutes later <utt> I'm good, how about you?
    <spk> A: <time> 2 hours later <utt>
    """
    query = f"{SPEAKER_TOKEN} {doc['speaker_list'][0]}: {UTTERANCE_TOKEN} {doc['context'][0]}"
    for spk, utt in zip(doc["speaker_list"][1:], doc["context"][1:]):
        query += f" {SPEAKER_TOKEN} {spk}: {TIME_TOKEN} 0 minutes later {UTTERANCE_TOKEN} {utt}"
    query += f" {SPEAKER_TOKEN} {doc['target_speaker']}: {TIME_TOKEN} {doc['time_elapsed']} later {UTTERANCE_TOKEN}"

    return query


def doc_to_text_instruction(doc: Dict[str, Union[str, List[str]]]) -> str:
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