import matplotlib.pyplot as plt
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
from tqdm import tqdm

from mnist_model import Discrininator, Generator

# GPUが使えるか否か
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ハイパーパラメータの設定
epochs = 1
lr = 2e-4
batch_size = 64
loss = nn.BCELoss()

# Generator と Discriminatorの設定
generator = Generator().to(device)
discriminator = Discrininator().to(device)

# optimizerの設定
opt_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.99))
opt_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.99))

# データセットの作成 : 学習データのみであることに注意
transfom = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))]
)
train_dataset = datasets.MNIST("mnist/", train=True, download=True, transform=transfom)
train_dataloader = DataLoader(
    train_dataset, batch_size = batch_size, shuffle = True, 
)

# 学習の開始
for epoch in range(epochs):
    print(f"Epochs{epoch} : ")
    for idx, (images, _) in enumerate(tqdm(train_dataloader)):
        # 本物の画像の生成
        real_images = images.to(device)
        real_outputs = discriminator(real_images)
        real_labels = torch.ones(real_images.size(0), 1).to(device)

        # 偽画像の生成
        noise = (torch.rand(real_images.size(0), 128) - 0.5) / 0.5
        noise = noise.to(device)
        fake_images = generator(noise)
        fake_outputs = discriminator(fake_images)
        fake_labels = torch.zeros(fake_images.size(0), 1).to(device)

        # Disciminatorの学習
        outputs = torch.cat((real_outputs, fake_outputs), 0)
        targets = torch.cat((real_labels, fake_labels), 0)

        loss_D = loss(outputs, targets)
        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        # Generatorの学習
        noise = (torch.rand(real_images.size(0), 128) - 0.5) / 0.5
        noise = noise.to(device)

        fake_images = generator(noise)
        fake_outputs = discriminator(fake_images)
        fake_labels = torch.ones(fake_images.size(0), 1).to(device)

        loss_G = loss(fake_outputs, fake_labels)
        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

        if idx % 100 == 0 or idx == len(train_dataloader):
            print(f'Epoch{epoch} Iteration{idx}')
            print(f'D_loss : {loss_D.item():.3f}')
            print(f'G_loss : {loss_G.item():.3f}')
        
    # TODO epoch 10=>1に変更している(一時的)
    if (epoch+1) % 1 == 0:
        torch.save(generator, 'Generator_epoch_{}.pth'.format(epoch))
        print('Model saved.')

# 画像の表示
for img, _ in train_dataloader:
    print("real")
    real_image = img[0][0]
    print("real_image.shape : ", real_image.shape)
    plt.imshow(real_image.reshape(28, 28))
    plt.show()

    # noiseの作成
    noise = (torch.rand(img.shape[0], 128) - 0.5) / 0.5
    print("noise : ", noise.size())
    noise = noise.to(device)
    gen_image = Generator(noise)

    # 生成画像の表示
    print('generated image')
    plt.imshow(gen_image[0][0].cpu().detach().numpy().reshape(28, 28))
    plt.show()
    plt.close()
    break
