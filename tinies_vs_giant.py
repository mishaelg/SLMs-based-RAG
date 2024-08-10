from beir import util
from beir.datasets.data_loader import GenericDataLoader
from sentence_transformers import SentenceTransformer
from pathlib import Path

from .fusion import comb_results
from .eval import calculate_metrics
from .index_manager import SingleIndexManager, index_corpus
from .utils import save_json, load_yaml
from .configs import Config
from .settings import (CONFIG_FILE, INDEX_FOLDER, DATASET_FOLDER
                       QUERIES_RESULTS_FILE)
from .logger_config import logger


def download_dataset(dataset_name, out_dir):
    """
    Downloads a dataset from a given URL and unzips it to the specified output directory.

    Args:
        dataset_name (str): The name of the dataset to download.
        out_dir (str): The output directory where the dataset will be saved.

    Returns:
        data_path (str): The path to the downloaded and unzipped dataset.

    """
    datset_path = out_dir / DATASET_FOLDER
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(
        dataset_name)
    data_path = util.download_and_unzip(url, out_dir)
    return data_path


def get_dataset(dataset_name, out_dir):
    """
    Downloads the specified dataset and returns the corpus, queries, and qrels.

    Args:
        dataset_name (str): The name of the dataset to download.
        out_dir (str): The output directory where the dataset will be saved.

    Returns:
        tuple: A tuple containing the corpus, queries, and qrels.
    """
    data_path = download_dataset(dataset_name, out_dir)
    corpus, queries, qrels = GenericDataLoader(
        data_folder=data_path).load(split="test")
    return corpus, queries, qrels


def get_model_query_result_on_dataset(model, queries, model_manager, output_path):
    """
    Retrieves the query results for a given model on a dataset.

    Args:
        model (object): The model used for encoding queries.
        queries (dict): A dictionary of query IDs and their corresponding queries.
        model_manager (object): The manager object for the model.
        output_path (str): The path to save the query results.

    Returns:
        None
    """
    queries_result_path = output_path / QUERIES_RESULTS_FILE
    results = {}
    for query_id, query in queries.items():
        query_embedding = model.encode(query)
        results[query_id] = model_manager.search(query_embedding)
    save_json(results, queries_result_path)


def calculate_dataset_performance(dataset_name, configs):
    """
    Calculate the performance of a dataset on all models.

    Args:
        dataset_name (str): The name of the dataset.
        configs (object): The configuration object.

    Returns:
        None
    """
    dataset_output_folder = configs.output_folder / dataset_name
    dataset_output_folder.mkdir(parents=True, exist_ok=True)
    corpus, queries, qrels = get_dataset(
        dataset_name, dataset_output_folder)
    # first get the results for all slm models
    for model_name in configs.slm_models_names:
        logger.info("Calculating results for %s", model_name)
        model_eval(configs, corpus, queries, model_name, qrels)
    # then get the results for the llm model
    logger.info("Calculating results for %s", configs.llm_model_name)
    model_eval(configs, corpus, queries, configs.llm_model_name, qrels)
    # Create the fusion results
    for fusion_method in configs.fusion_methods:
        logger.info("Fusing the results using %s", fusion_method)
        fusion_output_folder = dataset_output_folder / fusion_method
        fusion_output_folder.mkdir(parents=True, exist_ok=True)
        comb_results(
            dataset_output_folder, fusion_output_folder, fusion_method)
    # aggregate the results
    # TODO: implement the aggregation of results


def model_eval(configs, corpus, queries, model_name, qrels):
    """
    Evaluates a given model on a corpus and queries.

    Args:
        configs (object): Configuration object containing settings.
        corpus (list): List of sentences in the corpus.
        queries (list): List of query sentences.
        model_name (str): Name of the model to be evaluated.

    Returns:
        None
    """
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
    # calculate the metrics
    calculate_metrics(configs, model_output_folder, qrels)


def main():
    logger.info("Initializing the experiment")
    configs_data = load_yaml(CONFIG_FILE)
    configs = Config(**configs_data)
    configs.output_folder = Path(configs.output_folder)
    for dataset_name in configs.datasets:
        logger.info("Running the experiment on %s", dataset_name)
        calculate_dataset_performance(dataset_name, configs)
        logger.info("Finished the experiment on %s", dataset_name)
