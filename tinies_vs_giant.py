from beir import util
from beir.datasets.data_loader import GenericDataLoader


from .utils import save_json, load_json, load_yaml


CONFIG_FILE = "config.yaml"
DATASETS_FOLDER = "datasets"
CONFIGS = load_yaml(CONFIG_FILE)


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



def main():
    for dataset_name in CONFIGS["datasets"]:
        corpus, queries, qrels = get_dataset(
            dataset_name, CONFIGS["data_path"])
        print("Corpus size:", len(corpus))
        print("Queries size:", len(queries))
        print("Qrels size:", len(qrels))
    datasets_folder = CONFIGS["datasets_folder"]    
    corpus, queries, qrels = get_dataset(
        CONFIGS["dataset_name"], CONFIGS["data_path"])

    # Load the model
    model_manager = ModelManager()
    model = model_manager.get_model(CONFIGS["model_name"])

    # Index the corpus
    index_manager = IndexManager()
    index_manager.index_corpus(corpus, model, model_manager)

    # Get the results for the queries
    get_queries_results(queries, index_manager, CONFIGS["output_path"], model)