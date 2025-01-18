import sqlite3
import os


db_file = os.getenv("DATABASE_PATH","/app/data/sqlite.db")
def get_conn():
    conn = sqlite3.connect(db_file)
    return conn



def initialize_database():
    conn = get_conn()
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS new_words (
     id INTEGER PRIMARY KEY AUTOINCREMENT ,
     word VARCHAR(30) NOT NULL UNIQUE,
     explanations TEXT NOT NULL,
     insert_time INTEGER NOT NULL
    );
    ''')

    conn.commit()
    conn.close()

def exec_insert(insert,params=None):

    conn = get_conn()
    cursor= conn.cursor()
    try:
        cursor.execute(insert,params if params else ())
        conn.commit()
    except (Exception ,sqlite3.Error) as e:
        return "{}".format(str(e))
    except BaseException as e:
        return  "{}".format(str(e))
    finally:
        cursor.close()
        conn.close()
    return None

def exec_query(query , params=None):
    conn = get_conn()
    cursor = conn.cursor()

    cursor.execute(query,params if params else ())
    results = cursor.fetchall()
    conn.commit()

    cursor.close()
    conn.close()
    return results

def delete_newwords():
    conn = get_conn()
    cursor=conn.cursor()
    cursor.execute("DROP TABLE new_words;")






