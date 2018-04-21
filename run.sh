#!/usr/bin/env bash

MONGODB_URI=$(heroku config:get MONGODB_URI) && docker run -it --rm --name house-prices-back -p 5000:5000 -e="MONGODB_URI=$MONGODB_URI" \
house-prices-back
