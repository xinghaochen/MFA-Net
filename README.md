# handgesture
Skeleton-based dynamic hand gesture recognition

## Dependencies
- [keras 2.0.6](https://keras.io)
- theano 0.9.0 or tensorflow 1.2.0

## Datasets
- [DHG Dataset](http://www-rech.telecom-lille.fr/DHGdataset/): Dynamic Hand Gesture 14/28 dataset
- [Hand Gesture SHREC 2017 Dataset](http://www-rech.telecom-lille.fr/shrec2017-hand/)

## Usage
- Train:
```bash
sh train.sh
```
- Test:
```bash
sh test.sh
```

## Reference
- [1] [Skeleton-based Dynamic hand gesture recognition](http://www.cv-foundation.org/openaccess/content_cvpr_2016_workshops/w21/papers/De_Smedt_Skeleton-Based_Dynamic_Hand_CVPR_2016_paper.pdf), Quentin De Smedt, Hazem Wannous and Jean-Philippe Vandeborre, 2016 IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW).
- [2] [SHREC'17 Track: 3D Hand Gesture Recognition Using a Depth and Skeletal Dataset](https://hal.archives-ouvertes.fr/hal-01563505/document), Quentin De Smedt, Hazem Wannous and Jean-Philippe Vandeborre, Joris Guerry, Bertrand Le Saux, David Filliat, Eurographics Workshop on 3D Object Retrieval (2017).
