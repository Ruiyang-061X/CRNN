import six

from torch.utils.data import Dataset
import lmdb
from PIL import Image


class LMDBDataset(Dataset):

    def __init__(self, path, transform=None):
        self.environment = lmdb.open(path, max_readers=1)
        self.transform = transform

    def __getitem__(self, index):
        index += 1
        with self.environment.begin() as txn:
            image_key = ('image-%09d' % (index)).encode()
            image = txn.get(image_key)
            buffer = six.BytesIO()
            buffer.write(image)
            buffer.seek(0)
            image = Image.open(buffer).convert('L')
            label_key = ('label-%09d' % (index)).encode()
            label = txn.get(label_key).decode()
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        with self.environment.begin() as txn:
            n = int(txn.get('num-samples'.encode()))

        return n


if __name__ == '__main__':
    trainset = LMDBDataset(path='../crnn.pytorch_dataset/trainset')
    image, label =  trainset[0]
    print(image)
    print(label)
    print(len(trainset))
    image.show()