# Head pose estimation for PCR students

#### Requisites
- images-framework
- tqdm
- torch
- pytorch-lightning
- torchvision
- torch-summary
- tensorboard

#### Usage
```
usage: students_headpose_test.py [-h] [--input-data INPUT_DATA] [--show-viewer] [--save-image]
```

* Use the --input-data option to set an image, directory, camera or video file as input.

* Use the --show-viewer option to show results visually.

* Use the --save-image option to save the processed images.
```
usage: Alignment --database DATABASE
```

* Use the --database option to select the database model.
```
usage: StudentsHeadpose [--gpu GPU] --backbone {resnet,efficientnet} [--batch-size BATCH_SIZE] [--epochs EPOCHS] [--patience PATIENCE]
```

* Use the --gpu option to set the GPU identifier (negative value indicates CPU mode).

* Use the --backbone option to set the deep architecture.

* Use the --batch-size option to set the number of images in each mini-batch.

* Use the --epochs option to set the number of sweeps over the dataset to train.

* Use the --patience option to set number of epochs with no improvement after which training will be stopped.
```
> python test/students_headpose_test.py --input-data test/example.tif --database aflw --gpu 0 --backbone resnet --save-image
```