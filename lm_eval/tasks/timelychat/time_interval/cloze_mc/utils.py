import datasets


def process_docs_instruct_without_gold_response(dataset: datasets.Dataset):
    def _helper(doc):
        doc["choices"] = [doc["time_elapsed"].lower(), *list(map(lambda x: x.lower(), doc["negative_answers"]))]
        # Refer to Table 12 of https://arxiv.org/abs/2402.13211
        context = "\n".join([f"{speaker}: {utt}" for speaker, utt in zip(doc["speaker_list"], doc["context"])])
        query = (
            "### Dialogue context ###\n"
            f"{context}\n\n"
            f"Question: How much time is needed for {doc['target_speaker']} to respond?\n"
            "Answer:"
        )
        doc["query"] = query

        return doc

    return dataset.map(_helper)


def process_docs_instruct_with_gold_response(dataset: datasets.Dataset):
    def _helper(doc):
        doc["choices"] = [doc["time_elapsed"].lower(), *list(map(lambda x: x.lower(), doc["negative_answers"]))]

        context = "\n".join([f"{speaker}: {utt}" for speaker, utt in zip(doc["speaker_list"], doc["context"])])
        query = (
            "### Dialogue context ###\n"
            f"{context}\n\n"
            "### Target response ###\n"
            f"{doc['target_speaker']}: {doc['timely_response']}\n\n"
            f"Question: How much time is needed for {doc['target_speaker']} to respond with the target response?\n"
            "Answer:"
        )
        doc["query"] = query

        return doc

    return dataset.map(_helper)


def process_docs_rebot_without_gold_response(dataset: datasets.Dataset):
    """
    The dialogue format used in ReBot (https://arxiv.org/pdf/2310.13420).
    ex) <relationship> r <time interval> summary <user> u1 <bot> c1 <user> ... <bot> cn
    """
    def _helper(doc):
        doc["choices"] = [doc["time_elapsed"].lower(), *list(map(lambda x: x.lower(), doc["negative_answers"]))]

        query = " ".join([f"<{speaker}> {utt}" for speaker, utt in zip(doc["speaker_list"], doc["context"])])
        query += " <"
        # NOTE: `target_delimiter` must be set to "" in yaml file
        doc["choices"] = [c + " after>" for c in doc["choices"]]
        doc["query"] = query

        return doc

    return dataset.map(_helper)


def process_docs_rebot_with_gold_response(dataset: datasets.Dataset):
    def _helper(doc):
        doc["choices"] = [doc["time_elapsed"].lower(), *list(map(lambda x: x.lower(), doc["negative_answers"]))]

        query = " ".join([f"<{speaker}> {utt}" for speaker, utt in zip(doc["speaker_list"], doc["context"])])
        query += " <"
        # NOTE: `target_delimiter` must be set to "" in yaml file
        doc["choices"] = [c + f" after> <{doc['target_speaker']}> {doc['timely_response']}" for c in doc["choices"]]
        doc["query"] = query

        return doc

    return dataset.map(_helper)


def process_docs_msc_without_gold_response(dataset: datasets.Dataset):
    """
    The dialogue format used in MSC (https://aclanthology.org/2022.acl-long.356/).
    ex) p1: u1\np2: u2\n ... \ntime: t
    """
    def _helper(doc):
        doc["choices"] = [doc["time_elapsed"].lower(), *list(map(lambda x: x.lower(), doc["negative_answers"]))]

        speaker_mapping = {"A": "p1", "B": "p2"}
        query = ""
        for speaker, utt in zip(doc["speaker_list"], doc["context"]):
            if speaker in speaker_mapping.keys():
                speaker = speaker_mapping[speaker]
            query += f"{speaker}: {utt}\n"

        query += "time:"
        doc["query"] = query
        return doc

    return dataset.map(_helper)


def process_docs_msc_with_gold_response(dataset: datasets.Dataset):
    def _helper(doc):
        doc["choices"] = [doc["time_elapsed"].lower(), *list(map(lambda x: x.lower(), doc["negative_answers"]))]

        speaker_mapping = {"A": "p1", "B": "p2"}
        query = ""
        for speaker, utt in zip(doc["speaker_list"], doc["context"]):
            if speaker in speaker_mapping.keys():
                speaker = speaker_mapping[speaker]
            query += f"{speaker}: {utt}\n"

        query += "time:"
        doc["query"] = query

        target_speaker = speaker_mapping[doc["target_speaker"]] if doc["target_speaker"] in speaker_mapping.keys() else doc["target_speaker"]
        doc["choices"] = [c + f"\n{target_speaker}: {doc['timely_response']}" for c in doc["choices"]]
        return doc

    return dataset.map(_helper)


def process_docs_sft_without_gold_response(dataset: datasets.Dataset):
    """
    ex)
    <spk> A: <utt> Hi, how are you?
    <spk> B: <time> 0 minutes later <utt> I'm good, how about you?
    <spk> A: <time> 2 hours later <utt>
    """
    def _helper(doc):
        doc["choices"] = [doc["time_elapsed"].lower(), *list(map(lambda x: x.lower(), doc["negative_answers"]))]

        query = f"<spk> {doc['speaker_list'][0]}: {doc['context'][0]}"
        for spk, utt in zip(doc["speaker_list"][1:], doc["context"][1:]):
            query += f" <spk> {spk}: <time> 0 minutes later <utt> {utt}"
        query += " <spk> A: <time>"
        doc["query"] = query

        doc["choices"] = [c + " later" for c in doc["choices"]]
        return doc

    return dataset.map(_helper)


def process_docs_sft_with_gold_response(dataset: datasets.Dataset):
    def _helper(doc):
        doc["choices"] = [doc["time_elapsed"].lower(), *list(map(lambda x: x.lower(), doc["negative_answers"]))]

        query = f"<spk> {doc['speaker_list'][0]}: {doc['context'][0]}"
        for spk, utt in zip(doc["speaker_list"][1:], doc["context"][1:]):
            query += f" <spk> {spk}: <time> 0 minutes later <utt> {utt}"
        query += " <spk> A: <time>"
        doc["query"] = query

        doc["choices"] = [c + f" later <utt> {doc['timely_response']}" for c in doc["choices"]]
        return doc

    return dataset.map(_helper)