#!/usr/bin/env bash
docker run -it --rm --name house-prices-back -p 5000:5000 -v $PWD/instance_in_docker:/home/sources/instance \
house-prices-back
