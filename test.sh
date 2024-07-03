#!/bin/bash


echo "test start"

coverage run -m unittest discover
coverage report
coverage html

echo "test end"
