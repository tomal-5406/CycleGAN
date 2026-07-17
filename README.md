This notebook implements a CycleGAN (Cycle-Consistent Generative Adversarial Network) to remove artifacts from X-ray images and translate them into a clean, artifact-free version. Since there is no exact clean copy available for every artifact image (the data is unpaired), a CycleGAN is used instead of a paired translation model like Pix2Pix. This notebook is a pilot run on a small sample dataset to test the approach before scaling up.

Project Methodology
Phase I: Dataset Preparation

Import the required libraries (PyTorch, Torchvision, PIL).
Unzip the pilot run dataset (pilot_run_dataset.zip) from Google Drive into the Colab runtime. The dataset has two folders: artifacts (domain A) and clean (domain B).
Build a custom Dataset class that loads images from the artifacts folder and the clean folder independently, since the two folders don't need to have the same number of images (unpaired data).
Apply Resize (224x224), Random Horizontal Flip, ToTensor and Normalize (mean 0.5, std 0.5) as transforms, then load the dataset into a Dataloader with batch size 1.
Phase II: Model Architecture

Build a Residual Block (Conv, InstanceNorm, ReLU, Conv, InstanceNorm, with a skip connection), the core building block used inside the Generator.
Build a ResNet-style Generator: an initial 7x7 convolution block, 2 downsampling layers, 6 Residual Blocks, 2 upsampling layers, and a final 7x7 convolution with Tanh activation.
Build a PatchGAN Discriminator with 4 convolution layers. Instead of judging the whole image as real or fake at once, it classifies smaller patches of the image.
Create two Generators, G_AB (artifact to clean) and G_BA (clean to artifact), and two Discriminators, D_A and D_B, to complete the CycleGAN setup.
Phase III: Training

Define three loss functions: Adversarial loss (MSE), Cycle-Consistency loss (L1) and Identity loss (L1).
Set up Adam optimizers, one for both Generators together and one each for the two Discriminators (Learning Rate: 0.0002, Betas: 0.5, 0.999).
Train for 200 epochs. In every batch, generate a fake clean image from a real artifact image, and a fake artifact image from a real clean image, using the two Generators.
Compute the Adversarial loss (how well each Generator fools its Discriminator), the Cycle-Consistency loss (translate the fake image back to its original domain and compare it with the real image), and the Identity loss (check that a Generator leaves an already-correct-domain image unchanged).
Combine the total Generator loss as Adversarial loss + 10 x Cycle-Consistency loss + 5 x Identity loss, then update both Generators together.
Update Discriminator A and Discriminator B separately, using real and generated images.
Print the Generator and Discriminator losses after every 100 batches to track training progress.
Save model checkpoints (G_AB, G_BA, D_A, D_B) and sample translated images after every epoch.
Zip the generated sample images folder for download once training is complete.
Phase IV: Inference & Visualization

Load a trained G_AB Generator checkpoint (epoch 149) for inference.
Build a function that resizes and normalizes a new image the same way as training, passes it through the Generator, and de-normalizes the output back to the original [0,1] pixel range.
Translate a sample artifact image into its clean version using G_AB and save the result.
Display the original artifact image and the translated clean image side by side to visually compare the result.
