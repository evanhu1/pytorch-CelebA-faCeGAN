import torch
import torch.optim as optim
from model import *
from transforms import *
from training_loop import *
from celeba_data import *
from torch.utils.data import DataLoader

classes = 2
topn = 1
checkpoint_dir = 'Model_Checkpoints/labels'
name = 'facegan'
batch_size = 128
gen_steps = 1
disc_steps = 1
epochs = 40
img_size = 64
lr = 0.0002
beta = 0.5
desired_attr = ['Male', 'Young']
label_size = len(desired_attr)

imgtransform = BasicImageCropTransform(size = (img_size, img_size), scale = (1, 2))
anntransform = celeb_label_transform(desired_attr)
#transform = TransformWrapper(imgtransform, anntransform)
dataset = CelebDS(imgtransform, anntransform)
dataloader = DataLoader(dataset, batch_size, pin_memory = True)
print("Total Base Examples: " + str(len(dataset)))

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on a GPU")
else:
    device = torch.device("cpu")
    print("Running on a CPU")

def weights_init(m): 
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

generator = Generator(label_size).to(device)
generator.apply(weights_init)
discriminator = Discriminator(label_size).to(device)
discriminator.apply(weights_init)
optimizerG = optim.Adam(generator.parameters(), lr=lr, betas=(beta, 0.999))
optimizerD = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta, 0.999))

gen_history, discrim_history = training_loop(dataloader, label_size, img_size, batch_size, 
                                             epochs, generator, discriminator, optimizerG, optimizerD, True,
                                             checkpoint_dir, name, gen_steps, disc_steps, device)
