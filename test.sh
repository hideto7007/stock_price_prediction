#!/bin/bash


echo "test start"

if [ ! -e "./tests/api/database/db/test_database.db" ]; then
    echo "make test database"
    touch ./tests/api/database/db/test_database.db
    chmod 666 ./tests/api/database/db/test_database.db
fi

pipenv run coverage run -m unittest discover
pipenv run coverage report
pipenv run coverage html

if [ -e "./tests/api/database/db/test_database.db" ]; then
    rm -r ./tests/api/database/db/test_database.db
    echo "delete test databse"
fi

echo "test end"
