#!/bin/bash
echo 'Using Docker to start the container and run tests ...'
sudo docker build --force-rm --ssh default=$HOME/.ssh/id_rsa -t students_headpose_image .
sudo docker run --name students_headpose_container --rm --gpus all -it -d students_headpose_image bash
sudo docker exec -w /home/username/students_headpose students_headpose_container python test/students_headpose_test.py --input-data test/example.tif --database 300wlp --gpu 0 --backbone resnet --save-image
echo 'Transferring data from docker container to your local machine ...'
mkdir -p output
sudo docker cp students_headpose_container:/home/username/conda/envs/students/lib/python3.10/site-packages/images_framework/output/images/. output/
sudo chown -R "${USER}":"${USER}" output
sudo docker rm -f students_headpose_container
sudo docker image rm students_headpose_image
sudo docker builder prune -a -f