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


def doc_to_text_rebot_summ(doc: Dict[str, Union[str, List[str]]]) -> str:
    """
    Summary-appended prompt for ReBot evaluation.
    """
    # query = f"{doc['summary']} <{doc['speaker_list'][-1]}> {doc['context'][-1]}"
    query = " ".join([f"<{speaker}> {utt}" for speaker, utt in zip(doc["speaker_list"], doc["context"])])
    query += f" <summary> {doc['summary']}"
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


def doc_to_text_msc_summ(doc: Dict[str, Union[str, List[str]]]) -> str:
    """
    Summary-appended prompt for MSC evaluation.
    """
    speaker_mapping = {"A": "p1", "B": "p2"}
    query = ""
    for speaker, utt in zip(doc["speaker_list"], doc["context"]):
        if speaker in speaker_mapping.keys():
            speaker = speaker_mapping[speaker]
        query += f"{speaker}: {utt}\n"

    target_speaker = speaker_mapping[doc["target_speaker"]] if doc["target_speaker"] in speaker_mapping.keys() else doc["target_speaker"]
    query += f"summary: {doc['summary']}\ntime: {doc['time_elapsed']}\n{target_speaker}:"

    return query


def doc_to_text_gapchat_both(doc: Dict[str, Union[str, List[str]]]) -> str:
    """
    The dialogue format used in GapChat (https://aclanthology.org/2023.findings-emnlp.720).
    ex)
    Text:<spk> speaker_1: <utt> Hi, how are you? <spk> speaker_2: <utt> I'm good, thank you. <spk> ...\n
    Progress: <spk> speaker_1: You just started preparing for a presentation.
    Schedule: <spk> speaker_1: finished: to-do: You just started preparing for a presentation.
    label: <spk> speaker_2: <utt>
    """
    last_turn = doc["text"].split("<spk>")[-1].strip()
    target_speaker = "speaker_1" if last_turn.startswith("speaker_2") else "speaker_2"
    query = f"Text:{doc['text']}\nProgress:{doc['Progress']}\nSchedule:{doc['Schedule']}\nlabel:<spk> {target_speaker}: <utt>"

    return query


def doc_to_text_gapchat_progress(doc: Dict[str, Union[str, List[str]]]) -> str:
    last_turn = doc["text"].split("<spk>")[-1].strip()
    target_speaker = "speaker_1" if last_turn.startswith("speaker_2") else "speaker_2"
    query = f"Text:{doc['text']}\nProgress:{doc['Progress']}\nlabel:<spk> {target_speaker}: <utt>"
    
    return query


def doc_to_text_gapchat_schedule(doc: Dict[str, Union[str, List[str]]]) -> str:
    last_turn = doc["text"].split("<spk>")[-1].strip()
    target_speaker = "speaker_1" if last_turn.startswith("speaker_2") else "speaker_2"
    query = f"Text:{doc['text']}\nSchedule:{doc['Schedule']}\nlabel:<spk> {target_speaker}: <utt>"
    
    return query


def doc_to_text_gapchat_unaware(doc: Dict[str, Union[str, List[str]]]) -> str:
    """
    <spk> speaker_1: <time> 0 minutes later <utt> Hi, how are you? <spk> speaker_2: <time> 0 minutes later <utt> I'm good, thank you. <spk> speaker_1: <time> {time_elapsed} later <utt>
    """
    last_turn = doc["text"].split("<spk>")[-1].strip()
    target_speaker = "speaker_1" if last_turn.startswith("speaker_2") else "speaker_2"
    turns = doc["text"].split("<spk>")[1:]
    query = turns[0].strip() + " "
    for turn in turns[1:]:
        split = turn.split("<utt>")
        speaker = split[0].strip()
        utt = split[1].strip()
        query += f"<spk> {speaker}: <time> 0 minutes later <utt> {utt} "
    query += f"<spk> {target_speaker}: <time> {doc['time_elapsed']} later <utt>"
    
    return query
