import sqlite3

dbpath = './db/table/brancd_code.db'
conn = sqlite3.connect(dbpath)

conn.close()