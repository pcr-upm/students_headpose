#!/bin/bash
echo 'Using Docker to start the container and run tests ...'
sudo docker build --force-rm --build-arg SSH_PRIVATE_KEY="$(cat ~/.ssh/id_rsa)" -t students_headpose_image .
sudo docker volume create --name students_headpose_volume
sudo docker run --name students_headpose_container -v students_headpose_volume:/home/username --rm --gpus all -it -d students_headpose_image bash
sudo docker exec -w /home/username/students_headpose students_headpose_container python test/students_headpose_test.py --input-data test/example.tif --database aflw --gpu 0 --backbone resnet --save-image
sudo docker stop students_headpose_container
echo 'Transferring data from docker container to your local machine ...'
mkdir -p output
sudo chown -R "${USER}":"${USER}" /var/lib/docker/
rsync --delete -azvv /var/lib/docker/volumes/students_headpose_volume/_data/conda/envs/students/lib/python3.10/site-packages/images_framework/output/images/ output
sudo docker system prune --all --force --volumes
sudo docker volume rm $(sudo docker volume ls -qf dangling=true)
