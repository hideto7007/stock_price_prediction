#!/bin/bash

sqlite3 ./db/database.db < ./db/sql/make_table.sql
sqlite3 ./db/database.db < ./db/sql/insert.sql
