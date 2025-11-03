import os
import urllib.request
def download_data(url, save_path):
    if not os.path.exists(save_path):
        print(f"Downloading data from {url}...")
        urllib.request.urlretrieve(url, save_path)
        print(f"Data downloaded and saved to {save_path}.")
    else:
        print(f"Data already exists at {save_path}.")

if __name__ == "__main__":
    data_url = "https://datasets.cellxgene.cziscience.com/412352dd-a919-4d8e-9f74-e210627328b5.h5ad"
    save_directory = "./data"
    os.makedirs(save_directory, exist_ok=True)
    save_file_path = os.path.join(save_directory, "dataset.h5ad")
    download_data(data_url, save_file_path)