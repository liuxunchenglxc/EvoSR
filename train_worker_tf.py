import sqlite3
import time
import argparse
import os
import subprocess
import init_ea

parser = argparse.ArgumentParser(description="A worker of training for ea.")
parser.add_argument('-n', '--name', type=str, help="Unique name of worker.")
parser.add_argument('-g', '--gpu', type=int, help="Index of GPU.")
args = parser.parse_args()


def connect_db():
    con = sqlite3.connect("TrainWorker.db")
    cur = con.cursor()
    # create table task
    sql = "CREATE TABLE IF NOT EXISTS task(id INTEGER PRIMARY KEY)"
    op_db_no_params(con, cur, sql)
    # create table task_get
    sql = "CREATE TABLE IF NOT EXISTS task_get(id INTEGER PRIMARY KEY, worker INTEGER)"
    op_db_no_params(con, cur, sql)
    # create table worker_register
    sql = "CREATE TABLE IF NOT EXISTS worker_register(worker_id INTEGER PRIMARY KEY, lucky_string TEXT)"
    op_db_no_params(con, cur, sql)
    # create table task_end
    sql = "CREATE TABLE IF NOT EXISTS task_end(id INTEGER PRIMARY KEY)"
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


def get_train_task_info(id: int):
    row = select_db(init_ea.connect_db, "select * from code_file where id=?", (id,))
    ckpt_dir = row[2]
    save_dir = row[3]
    log_dir = row[4]
    return ckpt_dir, save_dir, log_dir


def try_score_info(id: int):
    row = select_db(init_ea.connect_db, "select * from score where id=?", (id,))
    if row is None:
        return 404
    else:
        return 200


def worker_register(con_func, lucky_string: str):
    # check lucky string
    if lucky_string.count('%') > 0 or lucky_string.count('_') > 0:
        return -1
    # give or get an id
    row = select_db(con_func, "select worker_id from worker_register where lucky_string like ?", (lucky_string,))
    if row is None:
        row = select_db_no_params(con_func, "select count(*) from worker_register")
        count = row[0]
        print("New worker id: ", count, " Name: ", lucky_string)
        insert_db(con_func, "insert into worker_register values (?, ?)", (count, lucky_string))
        return count
    else:
        return row[0]


def find_task(con_func, worker_id: int):
    while True:
        # find available task
        row = select_db_no_params(con_func, "select id from task where id not in (select id from task_get)")
        if row is None:
            time.sleep(60)
            continue
        # try to get task
        id = row[0]
        try:
            insert_db(con_func, "insert into task_get values (?, ?)", (id, worker_id))
        except sqlite3.IntegrityError:
            time.sleep(10)
            row = select_db(con_func, "select id from task_get where id=? and worker=?", (id, worker_id))
            if row is None:
                continue
            else:
                return id
        # get it here
        return id


def check_task_continue(con_func, worker_id: int):
    # find all task of worker and check it finish or not
    row = select_db(con_func, "select id from task_get where worker=? and id not in (select id from task_end)", (worker_id,))
    if row is None:
        return -1
    else:
        return row[0]


def finish_task(con_func, id: int):
    row = select_db(con_func, "select id from task_end where id=?", (id,))
    if row is None:
        try:
            insert_db(con_func, "insert into task_end values (?)", (id,))
        except sqlite3.IntegrityError:
            row = select_db(con_func, "select id from task_end where id=?", (id,))
            if row is None:
                insert_db(con_func, "insert into task_end values (?)", (id,))


def log_bad_code(id: int):
    insert_db(init_ea.connect_db, "insert into bad_code values (?, ?)", (id, './train_outputs/{}.out'.format(id)))


def do_task(con_func, id: int, gpu: int, lucky_string: str):
    ckpt_dir, save_dir, log_dir = get_train_task_info(id)
    epochs = 200
    if try_score_info(id) == 200:
        print("task ", id, " is done, and submitting.")
        finish_task(con_func, id)
        print("task ", id, " is done, and submitted.")
        return
    train_script = '''import train_tf
import sqlite3
import score
import models.model_{} as model_gen
import tensorflow as tf

tf.random.set_seed(666666)

id = {}
ckpt_dir = "{}"
save_dir = "{}"
log_dir = "{}"
final_model_dirname, best_psnr, the_ssim, runtime_time = train_tf.train(gpu_idx={}, model_class=model_gen.GEN_SR_MODEL,
                                                                 model_ckpt_dir=ckpt_dir, model_save_dir=save_dir,
                                                                 log_dir=log_dir, epochs={}, batch_size=4)
con = sqlite3.connect("EA.db")
cur = con.cursor()
cur.execute("insert into res_file values (?, ?, ?)", (id, final_model_dirname, save_dir))
# cur.execute("insert into runtime values (?, ?)", (id, runtime_time))
cur.execute("insert into sr values (?, ?, ?)", (id, best_psnr, the_ssim))
# model_score = score.score_sr(best_psnr, the_ssim, runtime_time)
# cur.execute("insert into score values (?, ?)", (id, model_score))
cur.close()
con.commit()
con.close()
    '''.format(id, id, ckpt_dir, save_dir, log_dir, gpu, epochs)
    script_path = './train_script_tf_gen_' + lucky_string + '.py'
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(train_script)
    if not os.path.exists("./train_outputs/"):
        os.mkdir("./train_outputs/")
    with open('./train_outputs/{}.out'.format(id), 'a', encoding='utf-8') as f:
        return_code = subprocess.call(['python', '-u', script_path], stdout=f, stderr=f)
    if return_code != 0:
        print("ERROR: Train process of id={} model is failed, return code={}, please check output log at ./train_outputs/{}.out".format(
            id, return_code, id))
        log_bad_code(id)
    print("task ", id, " is done, and submitting.")
    finish_task(con_func, id)
    print("task ", id, " is done, and submitted.")


def workflow(con_func, lucky_string: str, gpu: int):
    # register worker
    worker_id = worker_register(con_func, lucky_string)
    while True:
        # check non-finish task
        id = check_task_continue(con_func, worker_id)
        while id != -1:
            print("find undone task ", id)
            do_task(con_func, id, gpu, lucky_string)
            id = check_task_continue(con_func, worker_id)
        # find task and do it
        id = find_task(con_func, worker_id)
        do_task(con_func, id, gpu, lucky_string)


def main():
    workflow(connect_db, args.name, args.gpu)


if __name__ == '__main__':
    main()
