from download_file import download

if __name__ == "__main__":
    print("Downloading MNIST...")
    url = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'
    filename = 'mnist.npz'

    download(url, filename)

    print("Done.")