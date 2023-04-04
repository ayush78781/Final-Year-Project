import os
import matplotlib.pyplot as plt

from data import DIV2K
from model.srcnn import srcnn
from train import SrcnnTrainer
from tensorflow.keras import layers

# Number of residual blocks
depth = 16

# Super-resolution factor
scale = 4

# Downgrade operator
downgrade = 'bicubic'

# Location of model weights (needed for demo)
weights_dir = f'weights/srcnn'
weights_file = os.path.join(weights_dir, 'srcnn_weights.h5')

os.makedirs(weights_dir, exist_ok=True)

div2k_train = DIV2K(scale=scale, subset='train', downgrade=downgrade)
div2k_valid = DIV2K(scale=scale, subset='valid', downgrade=downgrade)

train_ds = div2k_train.dataset(batch_size=16, random_transform=True)
valid_ds = div2k_valid.dataset(batch_size=1, random_transform=False, repeat_count=1)

upscale_factor = 4
batch_size = 32
lr_image_size = (24,24) 
hr_image_size = (lr_image_size[0] * upscale_factor, lr_image_size[1] * upscale_factor)
normalize = layers.Rescaling(1./255)
bicubic_upscale = layers.Resizing(hr_image_size[0], hr_image_size[1], interpolation="bicubic")

# Normalize and Upscale
train_ds = train_ds.map(lambda image_batch, label_batch: (bicubic_upscale(image_batch), label_batch))
valid_ds = valid_ds.map(lambda image_batch, label_batch: (bicubic_upscale(image_batch), label_batch))
train_ds = train_ds.map(lambda image_batch, label_batch: (normalize(image_batch), normalize(label_batch)))
valid_ds = valid_ds.map(lambda image_batch, label_batch: (normalize(image_batch), normalize(label_batch)))

attention = True  # change to false

trainer = SrcnnTrainer(model=srcnn(attention=attention), 
                      checkpoint_dir=f'.ckpt/srcnn')

# Train SRCNN model for 300,000 steps and evaluate model
# every 1000 steps on the first 10 images of the DIV2K
# validation set. Save a checkpoint only if evaluation
# PSNR has improved.
print("training SRCNN model...")
trainer.train(train_ds,
              valid_ds.take(10),
              steps=300000, 
              evaluate_every=1000, 
              save_best_only=True,
              model_name = "srcnn_attention")


# Restore from checkpoint with highest PSNR
trainer.restore()

print("evaluating SRCNN model...")
# Evaluate model on full validation set
psnrv = trainer.evaluate(valid_ds)
ssimv = trainer.evaluate2(valid_ds)
print(f'PSNR = {psnrv.numpy():3f}')
print(f'SSIM = {ssimv.numpy():3f}')

# Save weights to separate location (needed for demo)
trainer.model.save_weights(weights_file)


model = srcnn(attention=attention)
model.load_weights(weights_file)

from model import resolve_single
from utils import load_image, plot_sample

print("resolving SRCNN model...")
def resolve_and_plot(lr_image_path):
    lr = load_image(lr_image_path)
    sr = resolve_single(model, lr)

    plot_sample(lr, sr)
    plt.savefig("srcnn_sample")

resolve_and_plot('demo/0869x4-crop.png')
