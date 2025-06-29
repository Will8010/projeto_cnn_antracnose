import os
from kaggle.api.kaggle_api_extended import KaggleApi

def baixar_dataset_kaggle(usuario_kaggle, api_kaggle, dataset_id, force_download):
    os.environ['KAGGLE_USERNAME'] = usuario_kaggle
    os.environ['KAGGLE_KEY'] = api_kaggle

    api = KaggleApi()
    api.authenticate()

    dataset_dir = os.path.join(os.getcwd(), dataset_id.split('/')[-1])
    if not os.path.exists(dataset_dir) or force_download:
        api.dataset_download_files(dataset_id, path=dataset_dir, unzip=True)
    return dataset_dir
        