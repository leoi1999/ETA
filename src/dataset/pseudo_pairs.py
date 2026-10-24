def find_pseudo_pairs(
    scores,
    source_entity_ids,
    target_entity_ids,
    threshold,
):
    """Return mutual top-1 pseudo pairs with their matching scores."""
    new_pairs = set()
    argmax_cols = scores.argmax(axis=0)
    argmax_rows = scores.argmax(axis=1)
    for col_index, row_index in enumerate(argmax_cols):
        if argmax_rows[row_index] != col_index:
            continue

        matching_score = float(scores[row_index][col_index])
        if matching_score <= threshold:
            continue

        new_pairs.add(
            (
                source_entity_ids[row_index],
                target_entity_ids[col_index],
                matching_score,
            )
        )
    return new_pairs


def remove_conflicting_pairs(scored_pairs):
    left_to_right = {}
    right_to_left = {}
    for left_id, right_id, _ in scored_pairs:
        left_to_right.setdefault(left_id, []).append(right_id)
        right_to_left.setdefault(right_id, []).append(left_id)

    conflicting_pairs = set()
    for left_id, right_ids in left_to_right.items():
        if len(right_ids) > 1:
            conflicting_pairs.update((left_id, right_id) for right_id in right_ids)
    for right_id, left_ids in right_to_left.items():
        if len(left_ids) > 1:
            conflicting_pairs.update((left_id, right_id) for left_id in left_ids)

    return {
        scored_pair
        for scored_pair in scored_pairs
        if scored_pair[:2] not in conflicting_pairs
    }


def choose_best_pairs(scored_pairs):
    """Keep pairs that are the highest-scoring choice for both entities."""
    best_by_left = {}
    best_by_right = {}
    for scored_pair in scored_pairs:
        left_entity, right_entity, matching_score = scored_pair
        if (
            left_entity not in best_by_left
            or best_by_left[left_entity][2] < matching_score
        ):
            best_by_left[left_entity] = scored_pair
        if (
            right_entity not in best_by_right
            or best_by_right[right_entity][2] < matching_score
        ):
            best_by_right[right_entity] = scored_pair

    return {
        scored_pair
        for scored_pair in scored_pairs
        if best_by_left[scored_pair[0]] == scored_pair
        and best_by_right[scored_pair[1]] == scored_pair
    }
