import logging
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from sentence_transformers import SentenceTransformer
from pathlib import Path


from .eval import get_queries_results
from .index_manager import SingleIndexManager, index_corpus
from .utils import save_json, load_json, load_yaml
from .configs import Config
from .settings import (CONFIG_FILE, DATASETS_FOLDER, RESULTS_FOLDER, INDEX_FOLDER, LOGS_FOLDER,
                       QUERIES_RESULTS_FILE)

logger = logging.getLogger('tinies_vs_giant')


def download_dataset(dataset_name, out_dir):
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(
        dataset_name)
    data_path = util.download_and_unzip(url, out_dir)
    return data_path


def get_dataset(dataset_name, out_dir):
    data_path = download_dataset(dataset_name, out_dir)
    corpus, queries, qrels = GenericDataLoader(
        data_folder=data_path).load(split="test")
    return corpus, queries, qrels


def get_model_query_result_on_dataset(model, queries, model_manager, output_path):
    queries_result_path = output_path / QUERIES_RESULTS_FILE
    results = {}
    for query_id, query in queries.items():
        query_embedding = model.encode(query)
        results[query_id] = model_manager.search(query_embedding)
    save_json(results, queries_result_path)


def calculate_dataset_performance(dataset_name, configs):
    dataset_output_folder = configs.output_folder / dataset_name
    dataset_output_folder.mkdir(parents=True, exist_ok=True)
    corpus, queries, qrels = get_dataset(
        dataset_name, configs.output_folder)
    # first get the results for all slm models
    for model_name in configs.slm_models_names:
        model_eval(configs, corpus, queries, model_name)
    # then get the results for the llm model
    model_eval(configs, corpus, queries, configs.llm_model_name)
    # finally calculate the metrics
    calculate_metrics(configs, dataset_output_folder, qrels)


def model_eval(configs, corpus, queries, model_name):
    logger.info("Calculating results for %s", model_name)
    model_output_folder = configs.output_folder / model_name
    model_output_folder.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(model_name)
    model_manager = SingleIndexManager(
        model.get_sentence_embedding_dimension())
    indexes_path = configs.output_folder / \
        INDEX_FOLDER if configs.save_index else None
    index_corpus(corpus, model, model_manager, indexes_path)
    get_model_query_result_on_dataset(
        model, queries, model_manager, model_output_folder)


def main():
    logger.info("Initializing the experiment")
    configs_data = load_yaml(CONFIG_FILE)
    configs = Config(**configs_data)
    configs.output_folder = Path(configs.output_folder)
    for dataset_name in configs.datasets:
        logger.info("Running the experiment on %s", dataset_name)
        calculate_dataset_performance(dataset_name, configs)
        logger.info("Finished the experiment on %s", dataset_name)
