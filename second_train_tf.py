import hashlib
import sqlite3
import time
import argparse
import os
import subprocess
import init_ea
import shutil

parser = argparse.ArgumentParser(description="Second train for ea.")
parser.add_argument('-n', '--name', type=str, help="Unique name of train.")
parser.add_argument('-g', '--gpu', type=int, help="Index of GPU.")
parser.add_argument('-i', '--id', type=int, help="ID of model.")
parser.add_argument('-l', '--lr', type=float, help="Learning rate.")
parser.add_argument('-b', '--bs', type=int, help="Batch size.")
parser.add_argument('-e', '--epochs', type=int, help="Epochs.")
args = parser.parse_args()


def connect_db():
    con = sqlite3.connect("SecondTrain.db")
    cur = con.cursor()
    # create table main
    sql = "CREATE TABLE IF NOT EXISTS main(hash TEXT PRIMARY KEY, id INTEGER, lr REAL, bs INTEGER, epochs INTEGER, ckpt_dir TEXT, save_dir TEXT, log_dir TEXT)"
    op_db_no_params(con, cur, sql)
    # create table score
    sql = "CREATE TABLE IF NOT EXISTS score(hash TEXT PRIMARY KEY, psnr REAL, ssim REAL, runtime REAL, score REAL)"
    op_db_no_params(con, cur, sql)
    return con, cur


def close_db(con: sqlite3.Connection, cur: sqlite3.Cursor):
    cur.close()
    con.close()


def insert_db(con_func, sql: str, params):
    con, cur = con_func()
    ok = 0
    while ok == 0:
        try:
            cur.execute(sql, params)
            con.commit()
            ok = 1
        except sqlite3.OperationalError:
            ok = 0
            time.sleep(10)
    close_db(con, cur)


def select_db_no_params(con_func, sql: str):
    con, cur = con_func()
    ok = 0
    while ok == 0:
        try:
            cur.execute(sql)
            con.commit()
            ok = 1
        except sqlite3.OperationalError:
            ok = 0
            time.sleep(10)
    row = cur.fetchone()
    close_db(con, cur)
    return row


def select_db(con_func, sql: str, params):
    con, cur = con_func()
    ok = 0
    while ok == 0:
        try:
            cur.execute(sql, params)
            con.commit()
            ok = 1
        except sqlite3.OperationalError:
            ok = 0
            time.sleep(10)
    row = cur.fetchone()
    close_db(con, cur)
    return row


def op_db_no_params(con, cur, sql: str):
    ok = 0
    while ok == 0:
        try:
            cur.execute(sql)
            con.commit()
            ok = 1
        except sqlite3.OperationalError:
            ok = 0
            time.sleep(10)


def get_real_id_and_dir(id: int):
    row = select_db(init_ea.connect_db, "select * from code_file "
                                        "where id=(select id from code where code like "
                                        "(select code from code where id=?) limit 1)", (id,))
    return row[0], row[2], row[3], row[4]


def get_train_task_info(hash: str):
    row = select_db(connect_db, "select * from main where hash=?", (hash,))
    ckpt_dir = row[5]
    save_dir = row[6]
    log_dir = row[7]
    return ckpt_dir, save_dir, log_dir


def try_score_info(hash: str):
    row = select_db(connect_db, "select * from score where hash=?", (hash,))
    if row is None:
        return 404
    else:
        return 200


def do_task(hash: str, id: int, lr: float, bs: int, epochs: int, gpu: int, lucky_string: str):
    ckpt_dir, save_dir, log_dir = get_train_task_info(hash)
    if try_score_info(hash) == 200:
        print("Second train ", hash, " is done.")
        return
    train_script = '''import train_tf
import sqlite3
import score
import models.model_{} as model_gen
import tensorflow as tf

tf.random.set_seed(666666)


def insert_db(sql: str, params):
    con = sqlite3.connect("SecondTrain.db")
    cur = con.cursor()
    ok = 0
    while ok == 0:
        try:
            cur.execute(sql, params)
            con.commit()
            ok = 1
        except sqlite3.OperationalError:
            ok = 0
            time.sleep(10)
    cur.close()
    con.commit()
    con.close()


hash = "{}"
ckpt_dir = "{}"
save_dir = "{}"
log_dir = "{}"
final_model_dirname, best_psnr, the_ssim, runtime_time = train_tf.train_second(gpu_idx={}, model_class=model_gen.GEN_SR_MODEL,
                                                                 model_ckpt_dir=ckpt_dir, model_save_dir=save_dir,
                                                                 log_dir=log_dir, epochs={}, batch_size={}, lr={})
model_score = score.score_sr(best_psnr, the_ssim, runtime_time)
insert_db("insert into score values (?, ?, ?, ?, ?)", (hash, best_psnr, the_ssim, runtime_time, model_score))
    '''.format(id, hash, ckpt_dir, save_dir, log_dir, gpu, epochs, bs, lr)
    script_path = './second_train_script_tf_gen_' + lucky_string + '.py'
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(train_script)
    if not os.path.exists("./second_train_outputs/"):
        os.mkdir("./second_train_outputs/")
    with open('./second_train_outputs/{}.out'.format(hash), 'a', encoding='utf-8') as f:
        return_code = subprocess.call(['python', '-u', script_path], stdout=f, stderr=f)
    if return_code != 0:
        print(
            "ERROR: Train process of id={} model is failed, return code={}, please check output log at ./second_train_outputs/{}.out".format(
                id, return_code, hash))
    else:
        print("Second train ", hash, " is done.")


def main():
    name = args.name
    gpu = args.gpu
    lr = args.lr
    bs = args.bs
    epochs = args.epochs
    id, o_ckpt_dir, o_save_dir, o_log_dir = get_real_id_and_dir(args.id)
    hash = hashlib.sha1((str(id) + '|' + str(lr) + '|' + str(bs) + '|' + str(epochs)).encode(encoding='utf-8')).hexdigest()
    print(id, lr, bs, epochs, hash)
    row = select_db(connect_db, "select * from main where hash=?", (hash,))
    if row is None:
        os.makedirs("./second_ckpt/", exist_ok=True)
        os.makedirs("./second_save/", exist_ok=True)
        os.makedirs("./second_logs/", exist_ok=True)
        ckpt_dir = "./second_ckpt/{}/".format(hash)
        save_dir = "./second_save/{}/".format(hash)
        log_dir = "./second_logs/{}/".format(hash)
        shutil.copytree(o_ckpt_dir, ckpt_dir, dirs_exist_ok=True)
        shutil.copytree(o_save_dir, save_dir, dirs_exist_ok=True)
        shutil.copytree(o_log_dir, log_dir, dirs_exist_ok=True)
        insert_db(connect_db, "insert into main values (?, ?, ?, ?, ?, ?, ?, ?)", (hash, id, lr, bs, epochs, ckpt_dir, save_dir, log_dir))
    do_task(hash, id, lr, bs, epochs, gpu, name)

if __name__ == '__main__':
    main()
