#!/bin/bash

if [[ ! -d "../data" ]]; then
    mkdir ../data
fi;

# run the label-studio docker containers
docker-compose -f ../label-studio/docker-compose.yml up

