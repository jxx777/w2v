from tqdm import tqdm
import urllib.request

class DownloadProgressBar(tqdm):
    def update_to(self, block_num=1, block_size=1, total_size=None):
        if total_size is not None:
            self.total = total_size
        self.update(block_num * block_size - self.n)

def download_with_progress(url, destination):
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc="Downloading") as t:
        urllib.request.urlretrieve(url, filename=destination, reporthook=t.update_to)
    print()
