#!/bin/bash


echo "test start"

pipenv run coverage run -m unittest discover
pipenv run coverage report
pipenv run coverage html

echo "test end"
