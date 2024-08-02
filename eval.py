from .utils import save_json


def calculate_ir_metrics(result_ids, relevant_ids):
    relevant_set = set(relevant_ids)
    num_relevant = len(relevant_set)

    if num_relevant == 0:
        return {"ap": 0.0, "p3": 0.0, "p5": 0.0, "recall": 0.0}

    hits = 0
    sum_precisions = 0.0
    p3, p5 = 0.0, 0.0

    for i, result in enumerate(result_ids, 1):
        if result in relevant_set:
            hits += 1
        precision = hits / i
        sum_precisions += precision

        if i == 3:
            p3 = precision
        elif i == 5:
            p5 = precision

    ap = sum_precisions / num_relevant
    recall = hits / num_relevant

    return {
        "ap": ap,
        "p3": p3,
        "p5": p5,
        "recall": recall
    }


def get_query_result(query, model_manager, model):
    query_embedding = model.encode(query)
    results = model_manager.search(query_embedding)
    return results


def get_queries_results(queries, model_manager, output_path, model):
    result_dict = {}
    for query_id, query in queries.items():
        results = get_query_result(query, model_manager, model)
        result_dict[query_id] = results
    save_json(result_dict, output_path)
