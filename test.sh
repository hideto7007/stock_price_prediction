#!/bin/bash


echo "test start"

TEST_PATH_STR="./tests/api/database/db/test_database.db"
TEST_PATH=./tests/api/database/db/test_database.db

if [ ! -e $TEST_PATH_STR ]; then
    echo "make test database"
    touch $TEST_PATH
    chmod 666 $TEST_PATH
fi

pipenv run coverage run -m unittest discover
pipenv run coverage report
pipenv run coverage html

if [ -e $TEST_PATH_STR ]; then
    rm -r $TEST_PATH
    echo "delete test databse"
fi

echo "test end"
