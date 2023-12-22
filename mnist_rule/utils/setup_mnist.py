import torchvision
import os
import torchvision.transforms.functional as TF
from tqdm import tqdm
import shutil


data_dir = 'data/mnist'
image_dir = os.path.join(data_dir, 'image')
label_dir = os.path.join(data_dir, 'label')
os.makedirs(image_dir, exist_ok=True)
os.makedirs(label_dir, exist_ok=True)

# create image and label folders
trainset = torchvision.datasets.MNIST(root=data_dir, train=True, download=True)
for i, (image, label) in tqdm(enumerate(trainset)):
    image_path = os.path.join(image_dir, f'{i}.png')
    label_path = os.path.join(label_dir, f'{i}.txt')
    image = TF.to_tensor(image)
    torchvision.utils.save_image(image, image_path)
    with open(label_path, 'w') as f:
        f.write(str(label))

testset = torchvision.datasets.MNIST(root=data_dir, train=False, download=True)
for i, (image, label) in tqdm(enumerate(testset)):
    image_path = os.path.join(image_dir, f'{i+len(trainset)}.png')
    label_path = os.path.join(label_dir, f'{i+len(trainset)}.txt')
    image = TF.to_tensor(image)
    torchvision.utils.save_image(image, image_path)
    with open(label_path, 'w') as f:
        f.write(str(label))

# create 0~9 folders
for label in range(10):
    target_subdir = os.path.join(data_dir, str(label))
    os.makedirs(target_subdir, exist_ok=True)

for img_filename in tqdm(os.listdir(os.path.join(data_dir, 'image'))):
    img_path = os.path.join(data_dir, 'image', img_filename)
    label_filename = img_filename.replace('.png', '.txt')
    label_path = os.path.join(data_dir, 'label', label_filename)
    with open(label_path, 'r') as f:
        label = int(f.read().strip())
    target_subdir = os.path.join(data_dir, str(label))
    shutil.copy(img_path, os.path.join(target_subdir, img_filename))