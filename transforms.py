import torchvision
import torch
import numpy as np
from torchvision.transforms import RandomResizedCrop, ToTensor, Compose, ToPILImage

add_noise = lambda img: torch.min(torch.ones(3, 64, 64), torch.max(-torch.ones(3, 64, 64), img + torch.randn(3, 64, 64) * .1 + 0))

def BasicImageCropTransform(size, scale = (1, 2)):
    """
    Params
      - size: (H, W) tuple for the image to be when it is done
      - scale: Range of the scaling factor of the image before it is cropped...
              increasing the upper end of the range has the effect of adding 
              some slight randomness (not changing aspect ratio, but where the crop is)

    forward(x):
      - Takes in a PIL image format (H, W, C) **OF ANY SIZE** and spits out tensor of (C, H, W)
      - Output size is always (C, size[0], size[1])
    """
    ratio = (size[1]/size[0], size[1]/size[0])
    transform = Compose([
        ToTensor(),
        RandomResizedCrop(size, scale, ratio),
        lambda img: img * 2 - 1,
        add_noise
    ])
    return transform


def BasicAnnotationTransform(topn = 1, classes = 91):
    """
    Returns a transform for basic multi-hot style classes
    - topn is number of largest objects that we will put in the label
        - topn = 1 means we will label the object with the 1 largest objects in the image

    Transform Input:
      - List of dictionaries with annotations that have keys:
          'area'
          'bbox'
          'category_id'
          'id'
          'image_id'
          'iscrowd'
          'segmentation'
    
    Transform Output:
      - Returns a binary vector with 1s in all the top n largest bounding box classes
      - If there aren't enough classes, then it will just put 1s for every class that
          is present
      - If there are no classes present, marks as background class (class 0)
    """
    def transform(annotation):
        annotation.sort(key = lambda x: x['area'], reverse = True)
        labels = []
        for i in range(topn):
            try:
                labels.append(annotation[i]['category_id'])
            except IndexError:
                break
        if labels == []:
            labels.append(0)
        labels = torch.tensor([labels])
        multi_hot_encoding = torch.zeros(labels.size(0), classes).scatter_(1, labels, 1.)
        return multi_hot_encoding.squeeze()
    return transform


def TransformWrapper(imgtransform, anntransform):
    """
    topn - will include class labels for the top n largest bounding boxes
    """
    def FullTransform(image, annotations):
        image = imgtransform(image)
        annotations = anntransform(annotations)
        return image, annotations
    return FullTransform

def AdvancedImageTransforms(img_transforms, p = .2, unique = False):
    """
    Takes in a list of lists of relatively unique image transformations img_transforms, and 
    permutes the images with a transformation from each list with probability p. If unique, 
    then it does only one transformation with probability p

    ex:
    img_transforms = [[rotation 90 degrees, rotation 180 degrees, ...], [brighter, darker]]

    **relativley unique means each list in the list won't ever be combined

    If unique = True, it does only ONE of these transformations from each category at a time

    **Generally, transformations should be interchangeable... one can come before or after 
    the other
    """
    transforms_per_class = [len(transforms) for transforms in img_transforms]
    total_transforms = sum(transforms_per_class)
    num_classes = len(img_transforms)

    all_transforms = []
    for transforms in img_transforms:
        all_transforms.extend(transforms)

    def transform(image, annotations):
        if not unique:
            #which lists of transforms we will draw on
            selection = np.random.rand(num_classes) < p
            label = torch.zeros(total_transforms)

            #loop over all transform classes that we are going to use, and  
            #   select which transform from that class we want, update label
            transforms_seen = 0
            for idx, positive_class in enumerate(selection):
                if positive_class:
                    selected_transform = np.random.randint(0, transforms_per_class[idx])
                    label[transforms_seen + selected_transform] = 1
                transforms_seen += transforms_per_class[idx]

        else:
            #only use one of these augmentations with p probability
            if np.random.rand() < p:
                #gets which class we want to use, picks randomly
                selection = np.random.randint(0, num_classes)
                #gets which transformation we want to use within this transformation class
                selection_in_class = np.random.randint(0, transforms_per_class[selection])
                label = torch.zeros(total_transforms)
                #the corresponding label is just the number of previous seen (-1 for zero 
                # indexing) plus the transformation in the class that we are using
                label[sum(transforms_per_class[:selection]) - 1 + selection_in_class] = 1

        for idx, indicator in enumerate(label):
            if indicator:
                image = all_transforms[idx](image)

        annotations = torch.cat([annotations, label])

        return image, annotations

    return transform

def AdvancedWrapper(imgtransform, anntransform, img_transforms, p = .2, unique = False):
    transform = torchvision.transforms.Compose([
                                   TransformWrapper(imgtransform, anntransform),
                                   AdvancedImageTransforms(img_transforms, p = .2, unique = False)
                                   ])
    
    def transform(image, annotation):
        image = imgtransform(image)
        annotation = anntransform(annotation)
        advanced = AdvancedImageTransforms(img_transforms, p = .2, unique = False)
        image, annotation = advanced(image, annotation)
        return image, annotation
    

    return transform

#just a basic function to give you a PIL from a tensor (C, H, W) just since
#printing PILs is nice in Colab
def returnPIL(img):
    return ToPILImage()(img)
