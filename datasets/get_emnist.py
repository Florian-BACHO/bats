import zipfile
import shutil
import os
from download_file import download

if __name__ == "__main__":
    print("Downloading EMNIST...")
    url = 'http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/matlab.zip'
    filename = 'emnist.zip'

    download(url, filename)

    print("Done.")

    print("Extracting EMNIST...")

    with zipfile.ZipFile('emnist.zip', 'r') as zip_ref:
        zip_ref.extractall(".")

    print("Done.")

    shutil.move("matlab/emnist-balanced.mat", "emnist-balanced.mat")

    print("Cleaning...")

    shutil.rmtree("matlab")
    os.remove(filename)

    print("Done.")