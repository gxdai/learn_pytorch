from __future__ import print_function
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")



plt.ion()   # interactive mode

# read cvs
landmarks_frame = pd.read_csv('faces/face_landmarks.csv')

n = 65

img_name = landmarks_frame.iloc[n, 0]
landmarks = landmarks_frame.iloc[n, 1:].as_matrix()
landmarks = landmarks.astype('float').reshape(-1, 2)

print('Image name: {}'.format(img_name)) 
print('Landmarks shape: {}'.format(landmarks.shape)) 
print('The first 4 landmarks: {}'.format(landmarks[:4])) 
# torch.utils.data.Dataset is an abstract class representing a dataset.
# You can customize your own dataset by inheriting Dataset and overwrite the following methods.

# __len__: so that len(dataset) returns the size of dataset
# __getitem__: to support the indexing that dataset[i] can be used to get i-th sample.


# customize dataset for face landmarks
# read csv in __init__
# read images in __getitem
def show_landmarks(image, landmarks):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated



class FaceLandmarksDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __getitem__(self, idx):
        image_name = os.path.join(self.root_dir, 
                                  self.landmarks_frame.iloc[idx, 0])
        image = io.imread(image_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}
        
        if self.transform:
            sample = self.transform(sample)

        return sample
    def __len__(self):
        return len(self.landmarks_frame)

face_dataset = FaceLandmarksDataset(csv_file='faces/face_landmarks.csv',
                                    root_dir='./faces')



fig = plt.figure()
print(len(face_dataset))
for i in range(len(face_dataset)):
    sample = face_dataset[i]
    print(i, sample['image'].shape, sample['landmarks'].shape)
    ax = plt.subplot(1, 4, i+1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    show_landmarks(**sample)
    if i == 3:
        break
    
plt.savefig('facemark.png')



# transformation of images

# * rescale
# RandomCrop
# ToTensor: convert images to torch images


"""
Ony need to implement __call__ and __init__ if required.
"""

# __call__ implements a function call operator


class Rescale:
    """Rescale images.
    
    Args:
        output_size (tuple or int): Desired output size.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))   # check input data type
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = h / w * self.output_size, self.output_size
            else:
                new_h, new_w = self.output_size, w / h * self.output_size
        else:
            new_h, new_w = self.output_size
        
        new_h, new_w = int(new_h), int(new_w)
        
        img = transform.resize(image, (new_h, new_w))
        print("Rescale", new_h, new_w)
        landmarks = (landmarks * [new_w / float(w), new_h / float(h)]).astype('int')
        
        return {'image': img, 'landmarks': landmarks}
         
class RandomCrop:
    """Random crop images>
    
    Args:
        output_size (int or tuple): Desired output size. If int, square crop
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        elif isinstance(output_size, tuple):
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
       
        print("Crop", h, w, new_h, new_w)
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        img = image[top: top + new_h, left: left+new_w]
        # landmarks swap axis for height and width as y, x
        landmarks = landmarks - [left, top]
        
        return {'image': img, 'landmarks': landmarks}



class ToTensor:
    """construct ndarrays in sample to tensors."""
    
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        # swap axis from (H, W, C) to (C, H, W)
        image = image.transpose(2, 0, 1)
        
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}


# torchvision.transforms.Compose is a simple callable class which allows us to do multiple operations.
scale = Rescale(256)
crop = RandomCrop(224)
composed = transforms.Compose([scale, crop])

# Apply each of the above operation to samples.
fig = plt.figure()
sample = face_dataset[65]
for i, tsfrm in enumerate([scale, composed]):
    print('{}-th operation'.format(i))
    transformed_sample = tsfrm(sample)
    ax = plt.subplot(1, 3, i + 1)
    plt.tight_layout()
    ax.set_title(type(tsfrm).__name__)
    show_landmarks(**transformed_sample)

plt.show()

# iterating over the dataset
# An image is read from the file on the fly.
# Transforms are applied on the read image.
# Since one of the transforms is random, data is augmented on sampling.

transformed_dataset = FaceLandmarksDataset(csv_file='faces/face_landmarks.csv',
                     root_dir='./faces/',
                     transform=transforms.Compose([
                         Rescale(256),
                         RandomCrop(224),
                         ToTensor()
                     ]))
for i in range(len(transformed_dataset)):
    sample = transformed_dataset[i]
    print(i, sample['image'].size(), sample['landmarks'].size())
    
    if i == 3:
        break

# batching the dataset
# shuffling the dataset
# loadnig the data in parallel using multiprocessing workers.

dataloader = DataLoader(transformed_dataset, batch_size=4,
                      shuffle=True, num_workers=4)

# show a batch
def show_landmarks_batch(sample_batched):
    images_batch, landmarks_batch = sample_batched['image'], sample_batched['landmarks']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    
    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))


    for i in range(batch_size):
        plt.scatter(landmarks_batch[i,:, 0].numpy() + i * im_size, 
                    landmarks_batch[i, :, 1],
                    s=10, marker='.', c='r')

        plt.title('Batch from dataloader')


for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['image'].size(),
          sample_batched['landmarks'].size())
    
    if i_batch == 5:
        plt.figure()

        show_landmarks_batch(sample_batched)
        plt.axis('off')
        plt.ioff()
        plt.show()
        plt.pause(100)
        
                    
