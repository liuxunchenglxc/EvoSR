import sqlite3

def connect_db():
    con = sqlite3.connect("EA.db")
    cur = con.cursor()
    # create table main
    sql = "CREATE TABLE IF NOT EXISTS main(id INTEGER PRIMARY KEY, gene TEXT)"
    cur.execute(sql)
    # create table score
    sql = "CREATE TABLE IF NOT EXISTS score(id INTEGER PRIMARY KEY, score REAL)"
    cur.execute(sql)
    # create table runtime
    sql = "CREATE TABLE IF NOT EXISTS runtime(id INTEGER PRIMARY KEY, runtime REAL)"
    cur.execute(sql)
    # create table sr
    sql = "CREATE TABLE IF NOT EXISTS sr(id INTEGER PRIMARY KEY, psnr REAL, ssim REAL)"
    cur.execute(sql)
    # create table generation
    sql = "CREATE TABLE IF NOT EXISTS generation(id INTEGER PRIMARY KEY, father INTEGER, mother INTEGER, iteration INTEGER)"
    cur.execute(sql)
    # create table length
    sql = "CREATE TABLE IF NOT EXISTS length(id INTEGER PRIMARY KEY, length INTEGER, gene_code_length INTEGER)"
    cur.execute(sql)
    # create table number
    sql = "CREATE TABLE IF NOT EXISTS number(id INTEGER PRIMARY KEY, number INTEGER, avg_length INTEGER)"
    cur.execute(sql)
    # create table code
    sql = "CREATE TABLE IF NOT EXISTS code(id INTEGER PRIMARY KEY, code TEXT)"
    cur.execute(sql)
    # create table status
    sql = "CREATE TABLE IF NOT EXISTS status(id INTEGER PRIMARY KEY, status INTEGER, iteration INTEGER)"
    cur.execute(sql)
    # create table code_file
    sql = "CREATE TABLE IF NOT EXISTS code_file(id INTEGER PRIMARY KEY, file TEXT, ckpt_dir TEXT, save_dir TEXT, log_dir TEXT)"
    cur.execute(sql)
    # create table res_file
    sql = "CREATE TABLE IF NOT EXISTS res_file(id INTEGER PRIMARY KEY, save_name TEXT, save_dir TEXT)"
    cur.execute(sql)
    # create table bad_code
    sql = "CREATE TABLE IF NOT EXISTS bad_code(id INTEGER PRIMARY KEY, train_log TEXT)"
    cur.execute(sql)
    # create table gene_expressed
    sql = "CREATE TABLE IF NOT EXISTS gene_expressed(id INTEGER PRIMARY KEY, gene_expressed TEXT)"
    cur.execute(sql)
    # create table hash
    sql = "CREATE TABLE IF NOT EXISTS hash(id INTEGER PRIMARY KEY, hash TEXT)"
    cur.execute(sql)
    con.commit()
    return con, cur


def close_db(con, cur):
    con.commit()
    cur.close()
    con.close()


def read_status(cur: sqlite3.Cursor):
    cur.execute("select * from status order by id desc limit 1")
    row = cur.fetchone()
    if row is None:
        # write init gene and set status to generated(0)
        gene_1 = '0-1,1,1,17,5,13,0-1,1,1,4,3,1,1-1,1,1,28,3,1,1-1,2,1,28,3,1,1-1,1,1,28,3,1,1-2,1,4,3-1,3,1,28,5,1,1-2,4,7,2-1,1,1,48,3,1,0-255'
        gene_2 = '0-1,1,1,22,3,1,1-1,2,1,28,3,1,1-1,2,1,28,3,1,1-1,1,1,28,3,1,1-1,1,1,28,3,1,1-1,1,1,26,5,13,2-1,2,1,16,1,4,0-1,1,1,12,3,1,1-2,2,8,2-1,1,1,48,3,1,0-255'
        gene_3 = '0-1,1,1,22,3,1,1-1,2,1,15,3,1,1-1,1,1,4,3,1,1-1,4,1,28,3,1,0-1,3,1,16,3,1,1-1,1,1,28,3,1,2-1,1,1,4,3,1,1-1,3,1,28,5,1,1-2,1,5,2-1,1,1,48,3,1,0-255'
        gene_4 = '0-1,1,1,22,3,1,0-1,2,1,28,3,1,1-1,1,1,28,3,1,1-1,1,1,28,3,1,1-1,1,1,28,3,1,1-1,1,1,16,3,1,1-1,2,1,22,1,1,1-1,2,1,12,3,1,1-2,2,8,2-1,1,1,48,3,1,0-255'
        gene_5 = '0-1,1,1,22,3,1,1-1,2,1,15,3,1,1-1,1,1,4,3,1,1-1,4,1,28,3,1,0-1,5,1,16,3,1,1-1,1,1,28,3,1,2-1,1,1,4,3,1,1-1,3,1,28,5,1,1-2,1,5,2-1,1,1,48,3,1,0-255'
        gene_6 = '0-1,1,1,22,3,1,1-1,2,1,9,5,15,0-1,2,1,28,3,1,1-1,3,1,28,3,1,1-1,1,1,28,3,1,1-1,1,1,28,3,1,1-2,6,1,2-1,1,1,26,5,13,2-1,2,1,16,1,4,0-1,1,1,12,3,1,1-2,2,9,2-255'
        gene_7 = '0-1,1,1,22,3,1,1-2,1,2,2-1,1,1,28,3,1,1-1,1,1,26,5,13,2-2,1,2,3-1,1,1,48,3,1,0-255'
        gene_8 = '0-1,1,1,22,3,19,1-1,2,1,15,3,1,1-1,1,1,4,3,1,1-1,4,1,28,3,1,0-1,3,1,16,3,1,1-1,1,1,28,3,1,2-1,1,1,4,3,1,1-1,3,1,28,5,1,1-2,1,5,2-1,1,1,48,3,1,0-255'
        gene_9 = '0-1,1,1,22,3,1,1-1,2,1,28,3,1,1-1,2,1,28,3,1,1-1,1,1,28,3,1,1-1,5,1,31,3,12,1-1,1,1,28,3,1,1-1,1,1,26,5,13,2-2,1,2,3-1,1,1,12,3,1,1-2,2,9,2-1,1,1,48,3,1,0-255'
        gene_10 = '0-1,1,1,16,3,1,1-1,2,1,28,3,1,1-1,2,1,28,3,1,1-1,1,1,28,3,1,1-1,1,1,28,3,1,1-2,1,4,3-1,2,1,16,1,4,0-1,1,1,12,3,1,1-2,2,8,2-1,1,1,48,3,1,0-255'
        gene_11 = '0-1,1,1,22,1,1,1-1,2,1,28,3,1,1-1,1,1,28,3,1,1-1,1,1,28,3,1,1-1,1,1,28,3,1,1-2,1,4,2-1,3,1,22,1,1,1-1,1,1,12,3,1,1-2,2,8,2-1,1,1,48,3,1,0-255'
        gene_12 = '0-1,1,1,22,3,1,1-1,2,1,28,3,1,1-1,1,1,28,3,1,1-1,1,1,28,3,1,1-1,1,1,28,3,1,1-1,1,1,16,3,1,1-1,2,1,22,1,1,1-1,1,1,12,3,1,1-2,2,8,2-1,1,1,48,3,1,0-255'
        gene_13 = '0-1,1,1,22,1,1,1-1,2,1,28,3,1,1-1,1,1,28,3,1,1-1,1,1,28,3,1,1-1,1,1,28,3,1,1-2,1,4,2-1,3,1,22,1,1,1-1,1,1,12,3,1,1-2,2,8,2-255'
        gene_14 = '0-1,1,1,22,3,1,1-1,2,1,28,3,1,1-2,1,2,2-1,1,1,28,3,1,1-1,3,1,28,3,1,1-1,1,1,26,5,13,2-2,1,2,3-2,1,5,2-1,1,1,48,3,1,0-255'

        sql_script = """
        insert into main(id, gene)
        values (
            1,
            '""" + gene_1 + """'
        );
        insert into main(id, gene)
        values (
            2,
            '""" + gene_2 + """'
        );
        insert into main(id, gene)
        values (
            3,
            '""" + gene_3 + """'
        );
        insert into main(id, gene)
        values (
            4,
            '""" + gene_4 + """'
        );
        insert into main(id, gene)
        values (
            5,
            '""" + gene_5 + """'
        );
        insert into main(id, gene)
        values (
            6,
            '""" + gene_6 + """'
        );
        insert into main(id, gene)
        values (
            7,
            '""" + gene_7 + """'
        );
        insert into main(id, gene)
        values (
            8,
            '""" + gene_8 + """'
        );
        insert into main(id, gene)
        values (
            9,
            '""" + gene_9 + """'
        );
        insert into main(id, gene)
        values (
            10,
            '""" + gene_10 + """'
        );
        insert into main(id, gene)
        values (
            11,
            '""" + gene_11 + """'
        );
        insert into main(id, gene)
        values (
            12,
            '""" + gene_12 + """'
        );
        insert into main(id, gene)
        values (
            13,
            '""" + gene_13 + """'
        );
        insert into main(id, gene)
        values (
            14,
            '""" + gene_14 + """'
        );
        insert into generation(id, father, mother, iteration)
        values (
            1,
            0,
            0,
            0
        );
        insert into generation(id, father, mother, iteration)
        values (
            2,
            0,
            0,
            0
        );
        insert into generation(id, father, mother, iteration)
        values (
            3,
            0,
            0,
            0
        );
        insert into generation(id, father, mother, iteration)
        values (
            4,
            0,
            0,
            0
        );
        insert into generation(id, father, mother, iteration)
        values (
            5,
            0,
            0,
            0
        );insert into generation(id, father, mother, iteration)
        values (
            6,
            0,
            0,
            0
        );insert into generation(id, father, mother, iteration)
        values (
            7,
            0,
            0,
            0
        );insert into generation(id, father, mother, iteration)
        values (
            8,
            0,
            0,
            0
        );insert into generation(id, father, mother, iteration)
        values (
            9,
            0,
            0,
            0
        );insert into generation(id, father, mother, iteration)
        values (
            10,
            0,
            0,
            0
        );insert into generation(id, father, mother, iteration)
        values (
            11,
            0,
            0,
            0
        );insert into generation(id, father, mother, iteration)
        values (
            12,
            0,
            0,
            0
        );insert into generation(id, father, mother, iteration)
        values (
            13,
            0,
            0,
            0
        );insert into generation(id, father, mother, iteration)
        values (
            14,
            0,
            0,
            0
        );
        insert into status(id, status, iteration)
        values (
            0,
            0,
            0
        );
        """
        cur.executescript(sql_script)
        status_dict = {
            'status': 0,
            'iteration': 0
        }
        return status_dict
    else:
        status_dict = {
            'status': row[1],
            'iteration': row[2],
        }
        return status_dict


if __name__ == '__main__':
    con, cur = connect_db()
    print(read_status(cur))
    close_db(con, cur)
