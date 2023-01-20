# implement different functions to reshape image to 224*224 shape
# 1 add padding 2) interpolation  3) deep learning based model
import torchvision


def get_transformation():
    t = []

    t.append(torchvision.transforms.Pad((30, 70, 30, 70)))
    t.append(torchvision.transforms.Pad((50, 50, 50, 50)))
    t.append(torchvision.transforms.Pad((50, 100, 50, 40)))
    t.append(torchvision.transforms.Pad((60, 30, 30, 120)))

    t.append(torchvision.transforms.ColorJitter())
    t.append(torchvision.transforms.RandomPerspective(0.2))
    t.append(torchvision.transforms.RandomRotation(30))
    t.append(torchvision.transforms.GaussianBlur(3))
    t.append(torchvision.transforms.RandomInvert(p=0.1))
    t.append(torchvision.transforms.RandomPosterize(4, p=0.1))

    t.append(torchvision.transforms.RandomAdjustSharpness(2, p=0.1))
    t.append(torchvision.transforms.RandomEqualize(p=0.1))

    return torchvision.transforms.RandomApply(t, p=0.3)
