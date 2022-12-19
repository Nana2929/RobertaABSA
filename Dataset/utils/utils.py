
def dependencies2format(doc):  # doc.sentences[i]
    """
    Format annotation: sentence of keys
                                - tokens
                                - tags
                                - predicted_dependencies
                                - predicted_heads
                                - dependencies
    RETURN token,pos,deprel,head,dependencies
    """
    token = doc["words"]
    pos = doc["pos"]
    # sentence['energy'] = doc['energy']
    predicted_dependencies = doc["predicted_dependencies"]
    predicted_heads = doc["predicted_heads"]
    deprel = doc["predicted_dependencies"]
    head = doc["predicted_heads"]
    dependencies = []
    for idx, item in enumerate(predicted_dependencies):
        dep_tag = item
        frm = predicted_heads[idx]
        to = idx + 1
        dependencies.append(
            [dep_tag, frm, to]
        )

    return token, pos, deprel, head, dependencies
