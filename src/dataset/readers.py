def read_triples(path):
    """Read tab-separated ``(head, relation, tail)`` triples."""
    triples = []
    with open(path, 'r', encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.strip().split('\t')
            if len(fields) != 3:
                raise ValueError(
                    "Expected three tab-separated fields in {} at line {}.".format(
                        path, line_number
                    )
                )
            triples.append(tuple(fields))
    return triples

def read_entity_ids(path):
    """Read and validate the numeric entity ids from an ``ent_ids_*`` file."""
    entity_ids = []
    seen = set()
    with open(path, 'r', encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.rstrip('\n').split('\t', 1)
            if len(fields) != 2:
                raise ValueError(
                    "Expected two tab-separated fields in {} at line {}.".format(
                        path, line_number
                    )
                )
            entity_id = int(fields[0])
            if entity_id in seen:
                raise ValueError(
                    "Duplicate entity id {} in {} at line {}.".format(
                        entity_id, path, line_number
                    )
                )
            seen.add(entity_id)
            entity_ids.append(entity_id)
    return entity_ids
