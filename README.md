# pytorch-CelebA-faCeGAN
Deep convolutional conditional GAN implementation with CelebA dataset that generates custom faces according to textual input.

### Setup
Requires CelebA dataset (http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) to be be downloaded and placed in the main directory as a folder named .../celeba/.

### Model Architecture

Generator                  |  Discriminator
:-------------------------:|:-------------------------:
![image](https://user-images.githubusercontent.com/41599986/121992709-008cac00-cd57-11eb-83c7-8aeb6c9418e5.png) | ![image](https://user-images.githubusercontent.com/41599986/121992724-071b2380-cd57-11eb-897d-59b16f6937b2.png)



### Results

2 Binary Attributes             |  5 Binary Attributes
:-------------------------:|:-------------------------:
![image](https://user-images.githubusercontent.com/41599986/121992245-1c438280-cd56-11eb-8183-51ab1953a3b0.png)  |  ![image](https://user-images.githubusercontent.com/41599986/121992305-3d0bd800-cd56-11eb-81f5-30d77668c45b.png)
