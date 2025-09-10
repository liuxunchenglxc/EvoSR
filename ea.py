import hashlib
import sqlite3
import random
import time

import ea_code_tf
import init_ea
import train_worker_tf

from graph_seq import gene_graph_seq_with_info


# Fitted and need to select and generate
# Has: fitness for selecting
# To do: select gene for new generation, crossover and mutant to generate new gene
# Selecting from top K=64 fitness
# Update iteration
def ea_select_generate(con: sqlite3.Connection, cur: sqlite3.Cursor, iteration: int):
    # update iteration
    iteration = iteration + 1
    # select at least one couple
    while select_generate_gene(con, cur, iteration) == 0:
        pass
    # update status
    cur.execute("select * from status order by id desc limit 1")
    row = cur.fetchone()
    id = row[0] + 1
    insert_db(con, cur, "insert into status values (?, ?, ?)", (id, 0, iteration))
    return iteration


# Select genes and generate gene by another function
# Binary Tournament Selection
def select_generate_gene(con: sqlite3.Connection, cur: sqlite3.Cursor, iteration):
    genes = []
    couples = []
    # read fitness order
    cur.execute("select * from number order by id desc limit 1")
    row = cur.fetchone()
    number = row[1]
    avg_length = row[2]
    # select last generation
    for row in cur.execute("select id from score where id in (select id from generation where iteration=?)"
                           " order by score desc limit ?", (iteration - 1, number - 1)).fetchall():
        genes.append(row[0])
    pos_list = [i for i in range(len(genes))]
    for _ in range(number - 1):
        father = genes[min(random.sample(pos_list, 2))]
        mother = genes[min(random.sample(pos_list, 2))]
        couples.append((father, mother))
    # generate
    for couple in couples:
        generate_gene(con, cur, couple, iteration, avg_length)
    con.commit()
    # keep the best one
    cur.execute("select * from main order by id desc limit 1")
    row = cur.fetchone()
    id = row[0] + 1
    cur.execute("select * from main where id=?", (genes[0],))
    row = cur.fetchone()
    gene = row[1]
    insert_db(con, cur, "insert into generation values (?, ?, ?, ?)",
              (id, genes[0], genes[0], iteration))
    insert_db(con, cur, "insert into main values (?, ?)", (id, gene))
    return len(couples)


# Generate gene and record it, crossover and mutant gene by other functions
def generate_gene(con: sqlite3.Connection, cur: sqlite3.Cursor, couple, iteration, avg_length):
    # get genes of couple
    cur.execute("select * from main where id=?", (couple[0],))
    row_father = cur.fetchone()
    cur.execute("select * from main where id=?", (couple[1],))
    row_mother = cur.fetchone()
    gene_f = row_father[1]
    gene_m = row_mother[1]
    # crossover gene of couple
    gene = crossover_gene(gene_f, gene_m, iteration)
    # mutant gene of couple
    gene = mutant_gene(gene, iteration, avg_length)
    # save gene as child
    # get id of child
    cur.execute("select id from main order by id desc limit 1")
    row = cur.fetchone()
    id = row[0] + 1
    # get generation of child in generation
    insert_db(con, cur, "insert into generation values (?, ?, ?, ?)",
              (id, couple[0], couple[1], iteration))
    insert_db(con, cur, "insert into main values (?, ?)", (id, gene))


# Crossover the gene
# Uniform Crossover
def crossover_gene(gene_a: str, gene_b: str, iteration: int):
    # copy the genes
    gene_a_c = gene_a
    gene_b_c = gene_b
    # prepare the genes without 0 and 255
    gene_a_units = gene_a.split('-')[1:-1]
    gene_b_units = gene_b.split('-')[1:-1]
    gene_a_len = len(gene_a_units)
    gene_b_len = len(gene_b_units)
    # crossover
    new_gene_a_units = []
    new_gene_b_units = []
    for i in range(max(gene_a_len, gene_b_len)):
        if i >= gene_a_len:
            temp_a = None
        else:
            temp_a = gene_a_units[i]
        if i >= gene_b_len:
            temp_b = None
        else:
            temp_b = gene_b_units[i]
        if random.randint(0, 1) == 0:
            if temp_a is not None:
                new_gene_a_units.append(temp_a)
            if temp_b is not None:
                new_gene_b_units.append(temp_b)
        else:
            if temp_a is not None:
                new_gene_b_units.append(temp_a)
            if temp_b is not None:
                new_gene_a_units.append(temp_b)
    # add 0 and 255
    new_gene_a_units = ['0'] + new_gene_a_units + ['255']
    new_gene_b_units = ['0'] + new_gene_b_units + ['255']
    gene_a = '-'.join(new_gene_a_units)
    gene_b = '-'.join(new_gene_b_units)
    # select one gene for next generation
    if iteration < 1:
        iteration = 1
    p_crossover = 1 / float(iteration * iteration)
    which_gene = random.randint(0, 1)
    if random.random() < p_crossover:
        if which_gene == 0:
            gene = gene_a
        else:
            gene = gene_b
    else:
        if which_gene == 0:
            gene = gene_a_c
        else:
            gene = gene_b_c
    return gene


# Mutant the gene
def mutant_gene(gene: str, iteration: int, avg_length: int):
    # mutant?
    if iteration < 1:
        iteration = 1
    if avg_length < 1:
        avg_length = 1
    if random.random() >= 1 / (iteration ** (1 / float(avg_length))):
        return gene
    # prepare the gene
    gene_units = gene.split('-')
    mutant_type = random.randint(0, 2)
    # if there is no real units, only add one unit.
    if len(gene_units) == 2:
        mutant_type = 1
    if mutant_type == 0:
        # delete one unit
        if len(gene_units) > 3:
            unit_index = random.randint(1, len(gene_units) - 2)
            gene_units = gene_units[:unit_index] + gene_units[unit_index + 1:]
        gene = '-'.join(gene_units)
        return gene
    elif mutant_type == 1:
        # add one unit
        unit_index = random.randint(1, len(gene_units) - 1)
        unit_type = random.randint(1, 3)
        if unit_type == 1:
            unit_params = ['1', str(random.randint(1, unit_index)), '1', str(random.randint(4, 32)),
                           str(random.choice([1, 3, 5])), str(random.randint(1, 16)), str(random.randint(0, 2))]
        else:
            unit_params = ['2', str(random.randint(1, unit_index)), str(random.randint(1, unit_index)),
                           str(unit_type)]
        gene_units.insert(unit_index, ','.join(unit_params))
        gene = '-'.join(gene_units)
        return gene
    # modify one unit
    # choice one unit
    unit_index = random.randint(1, len(gene_units) - 2)
    # prepare the unit
    unit_params = gene_units[unit_index].split(',')
    unit_input_num = int(unit_params[0])
    unit_type = int(unit_params[unit_input_num + 1])
    # choice part of params
    if 1 < unit_type < 5:
        part_mutant = random.randint(0, 1)
    else:
        part_mutant = random.randint(0, 2)
    # mutant
    if part_mutant == 0:  # inputs part
        input_mutant = random.randint(1, unit_input_num)
        base = int(unit_params[input_mutant])
        unit_params[input_mutant] = str(random.randint(base - int(base / 2), int((base + 1) * 1.5)))
    elif part_mutant == 1:  # unit type
        # type_mutant = random.randint(1, 6)
        type_mutant = random.randint(1, 3)
        if type_mutant == 1:  # CNN
            unit_params = ['1', unit_params[1], '1', str(random.randint(4, 32)), str(random.choice([1, 3, 5])),
                           str(random.randint(1, 16)), str(random.randint(0, 2))]
        elif type_mutant == 2 or type_mutant == 3:  # or type_mutant == 4:  # + * CAT
            if unit_input_num == 2:
                unit_params = ['2', unit_params[1], unit_params[2], str(type_mutant)]
            else:
                if unit_params[1] == '1':
                    unit_params = ['2', '1', str(random.randint(2, 8)), str(type_mutant)]
                else:
                    unit_params = ['2', '1', unit_params[1], str(type_mutant)]
    elif part_mutant == 2:  # net params
        if unit_type == 1:  # CNN
            mutant_pos = random.randint(3, 6)
            if mutant_pos == 3:  # channel
                base = int(unit_params[3])
                unit_params[3] = str(random.randint(max(base - int(base / 2), 3), int((base + 1) * 1.5)))
            elif mutant_pos == 4:  # filter_size
                unit_params[4] = str(random.choice([1, 3, 5]))
            elif mutant_pos == 5:  # group
                unit_params[5] = str(random.randint(1, int(unit_params[3])))
            elif mutant_pos == 6:  # activation
                unit_params[6] = str(random.randint(0, 2))
    # encode to gene
    gene_units[unit_index] = ','.join(unit_params)
    gene = '-'.join(gene_units)
    return gene


# Generated and need to train and score
# Has: some genes need to train
# To do：train and score genes that need to train and score
# No score gene need to train
def ea_train_score(con: sqlite3.Connection, cur: sqlite3.Cursor, iteration: int):
    # find genes that need to express to code, and express them
    for row in cur.execute("select * from main where id not in (select id from code)").fetchall():
        express_gene(con, cur, row[0], row[1])
    # find codes of genes that need to train, then train and score them
    for row in cur.execute("select * from code where id not in (select id from score) "
                           "and id not in (select id from bad_code)").fetchall():
        train_score_code(con, cur, row[0], row[1])
    # waiting for finish
    wait_train_workers()
    # update status
    cur.execute("select * from status order by id desc limit 1")
    row = cur.fetchone()
    id = row[0] + 1
    insert_db(con, cur, "insert into status values (?, ?, ?)", (id, 1, iteration))
    return iteration


def select_db_no_params(con_func, close_func, sql: str):
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
    close_func(con, cur)
    return row


def select_db(con_func, close_func, sql: str, params):
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
    close_func(con, cur)
    return row


# Wait train workers done their tasks
def wait_train_workers():
    row = select_db_no_params(train_worker_tf.connect_db, train_worker_tf.close_db,
                              "select * from task where id not in (select id from task_end)")
    while row is not None:
        time.sleep(60)
        row = select_db_no_params(train_worker_tf.connect_db, train_worker_tf.close_db,
                                  "select * from task where id not in (select id from task_end)")


# Express gene ,fix it ,and translate it to code by other functions
def express_gene(con: sqlite3.Connection, cur: sqlite3.Cursor, id: int, gene: str):
    # decode genes
    gene_units = gene.split('-')
    gene_units_params = [unit.split(',') for unit in gene_units]
    # fix gene params
    gene_units_params_fixed = fix_gene_params(gene_units_params)
    # record gene_fixed for debug
    insert_db(con, cur, "insert into gene_expressed values (?, ?)",
              (id, '-'.join([','.join(unit_params) for unit_params in gene_units_params_fixed])))
    # gene hash
    gene_units = [','.join(unit_params) for unit_params in gene_units_params_fixed][1:-1]
    seq, info = gene_graph_seq_with_info(gene_units)
    info_s = '-'.join(info)
    h = hashlib.sha1((seq + '|' + info_s).encode(encoding='utf-8')).hexdigest()
    sql = "insert or ignore into hash values (?, ?)"
    insert_db(con, cur, sql=sql, params=(id, h))
    # translate to code
    code, gene_code_length = ea_code_tf.gene_to_code(gene_units_params_fixed)
    gene_length = len(gene_units_params) - 2
    insert_db(con, cur, "insert into code values (?, ?)", (id, code))
    insert_db(con, cur, "insert into length values (?, ?, ?)", (id, gene_length, gene_code_length))


# Fix the gene
def fix_gene_params(gene_units_params):
    # record gene before fixing
    gene_before = '-'.join([','.join(unit_params) for unit_params in gene_units_params])
    # bounds error fix
    gene_units_params = fix_gene_input_bounds(gene_units_params)
    # input shape error
    gene_units_params = fix_gene_input_shape(gene_units_params)
    # record gene after fixing
    gene_after = '-'.join([','.join(unit_params) for unit_params in gene_units_params])
    # compare before and after gene
    while gene_before != gene_after:
        # record gene before fixing
        gene_before = gene_after
        # bounds error fix
        gene_units_params= fix_gene_input_bounds(gene_units_params)
        # input shape error
        gene_units_params = fix_gene_input_shape(gene_units_params)
        # record gene after fixing
        gene_after = '-'.join([','.join(unit_params) for unit_params in gene_units_params])
    return gene_units_params


# If input out of bound, set its pos to bound
def fix_gene_input_bounds(gene_units_params):
    pos = 0
    for i in range(len(gene_units_params)):
        if gene_units_params[i][0] == '0':
            continue
        if gene_units_params[i][0] == '255':
            break
        pos = pos + 1
        for j in range(int(gene_units_params[i][0])):
            if int(gene_units_params[i][j + 1]) > pos:
                gene_units_params[i][j + 1] = str(pos)
        # if gene_units_params[i][int(gene_units_params[i][0]) + 1] == '5':
        #     pos = pos + 1
    return gene_units_params


# Related unit type: + *
# Should check the number of channels
# Input c=3, output c=48=3*4*4 for x4 SR
# Fix split first param and all range to [1, x, 5, k, p_1, p_2, .., p_k], \
# p_1+p_2+..+p_k=input_channel for programing (format changed)
# Fix CNN group
def fix_gene_input_shape(gene_units_params):
    channel_record = []
    unit_insert = []
    for i in range(len(gene_units_params)):
        if gene_units_params[i][0] == '0':
            channel_record.append((i, 3))
            continue
        if gene_units_params[i][0] == '255':
            if channel_record[-1][1] != 48:
                type = int(
                    gene_units_params[channel_record[-1][0]][int(gene_units_params[channel_record[-1][0]][0]) + 1])
                if type == 1:  # CNN directly change filters
                    gene_units_params[channel_record[-1][0]][3] = '48'
                    group = int(gene_units_params[channel_record[-1][0]][5])
                    offset_a = int(gene_units_params[channel_record[-1][0]][1])
                    channel_a = channel_record[-offset_a - 1][1]
                    group = find_near_divide(group, 48, channel_a)
                    gene_units_params[channel_record[-1][0]][5] = str(group)
                else:
                    unit_insert.append((i, ['1', '1', '1', '48', '3', '1', '0']))
            break
        type = int(gene_units_params[i][int(gene_units_params[i][0]) + 1])
        if type == 1:  # CNN
            group = int(gene_units_params[i][5])
            channel = int(gene_units_params[i][3])
            offset_a = int(gene_units_params[i][1])
            channel_a = channel_record[-offset_a][1]
            group_new = find_near_divide(group, channel_a, channel)
            gene_units_params[i][5] = str(group_new)
            channel_record.append((i, channel))
        if type == 2 or type == 3:  # + *
            offset_a = int(gene_units_params[i][1])
            offset_b = int(gene_units_params[i][2])
            channel_a = channel_record[-offset_a][1]
            channel_b = channel_record[-offset_b][1]
            if channel_a == channel_b:
                channel_record.append((i, channel_a))
            else:
                if (channel_a > channel_b and gene_units_params[channel_record[-offset_a][0]][0] != '0') \
                        or gene_units_params[channel_record[-offset_b][0]][0] == '0':
                    # Fix a
                    type_a = int(gene_units_params[channel_record[-offset_a][0]][
                                     int(gene_units_params[channel_record[-offset_a][0]][0]) + 1])
                    if type_a == 1:  # CNN directly change filters
                        group = int(gene_units_params[channel_record[-offset_a][0]][5])
                        offset_a_a = int(gene_units_params[channel_record[-offset_a][0]][1])
                        channel_a_a = channel_record[-offset_a - offset_a_a][1]
                        group = find_near_divide(group, channel_b, channel_a_a)
                        gene_units_params[channel_record[-offset_a][0]][5] = str(group)
                        gene_units_params[channel_record[-offset_a][0]][3] = str(channel_b)
                        channel_record[-offset_a] = (channel_record[-offset_a][0], channel_b)
                        channel_record.append((i, channel_b))
                    else:
                        unit_insert.append((i, ['1', gene_units_params[i][1], '1', str(channel_b), '3', '1', '1']))
                        gene_units_params[i][1] = '1'
                        gene_units_params[i][2] = str(int(gene_units_params[i][2]) + 1)
                        channel_record.append((i, channel_b))
                else:
                    # Fix b
                    type_b = int(gene_units_params[channel_record[-offset_b][0]][
                                     int(gene_units_params[channel_record[-offset_b][0]][0]) + 1])
                    if type_b == 1:  # CNN directly change filters
                        group = int(gene_units_params[channel_record[-offset_b][0]][5])
                        offset_b_b = int(gene_units_params[channel_record[-offset_b][0]][1])
                        channel_b_b = channel_record[-offset_b - offset_b_b][1]
                        group = find_near_divide(group, channel_a, channel_b_b)
                        gene_units_params[channel_record[-offset_b][0]][5] = str(group)
                        gene_units_params[channel_record[-offset_b][0]][3] = str(channel_a)
                        channel_record[-offset_b] = (channel_record[-offset_b][0], channel_a)
                        channel_record.append((i, channel_a))
                    else:
                        unit_insert.append((i, ['1', gene_units_params[i][2], '1', str(channel_a), '3', '1', '1']))
                        gene_units_params[i][1] = '1'
                        gene_units_params[i][2] = str(int(gene_units_params[i][1]) + 1)
                        channel_record.append((i, channel_a))
    gene_units_params = gene_insert_units(gene_units_params, unit_insert)
    return gene_units_params


# Find near k divide num in s and t
def find_near_divide(k, s, t):
    if k < 1:
        return 1
    if s != t:
        if s > t:
            p = t
            t = s
            s = p
        r = t % s
        while r > 0:
            t = s
            s = r
            r = t % s
    if k > s:
        return s
    if s % k == 0:
        return k
    d = 1
    while d < s / 2 + 1:
        if s % (k - d) == 0:
            return k - d
        if s % (k + d) == 0:
            return k + d
        d = d + 1
    return 1


# Insert units to gene
def gene_insert_units(gene_units_params, unit_insert):
    insert_offset = 0
    for insert in unit_insert:
        insert_pos = insert[0] + insert_offset
        gene_units_params.insert(insert_pos, insert[1])
        pos = 1
        next_input_num = int(gene_units_params[insert_pos + 1][0])
        if next_input_num == 0 or next_input_num == 255:
            continue
        if gene_units_params[insert_pos + 1][next_input_num + 1] == '5':
            pos = pos + int(gene_units_params[insert_pos + 1][3]) - 1
        for i in range(insert_pos + 2, len(gene_units_params) - 1):
            input_num = int(gene_units_params[i][0])
            for j in range(input_num):
                offset = int(gene_units_params[i][j + 1])
                if offset > pos:
                    gene_units_params[i][j + 1] = str(offset + 1)
            if gene_units_params[i][input_num + 1] == '5':
                pos = pos + int(gene_units_params[i][3])
            else:
                pos = pos + 1
        insert_offset = insert_offset + 1
    return gene_units_params


def train_score_code(con: sqlite3.Connection, cur: sqlite3.Cursor, id: int, code: str):
    row = cur.execute("select * from code_file where id=?", (id,)).fetchone()
    if row is None:
        # Insert code to train and score framework and write it to file
        code_filename, code_name = ea_code_tf.code_to_file(id, code)
        # Set code file info
        ckpt_dir = './ckpt/' + code_name
        save_dir = './save/' + code_name
        log_dir = './logs/' + code_name
        insert_db(con, cur, "insert into code_file values (?, ?, ?, ?, ?)",
                  (id, code_filename, ckpt_dir, save_dir, log_dir))

    # Check whether trained
    sql = 'select psnr, ssim from sr where id in (select id from hash where hash=(select hash from hash where id=?)) ' \
          'order by psnr+ssim desc limit 1'
    params = (id,)
    row = select_db(init_ea.connect_db, init_ea.close_db, sql, params)
    if row is not None:
        psnr = row[0]
        ssim = row[1]
        row = select_db(init_ea.connect_db, init_ea.close_db, "select * from sr where id=?", (id,))
        if row is None:
            insert_db(con, cur, "insert or ignore into sr values (?, ?, ?)",
                      (id, psnr, ssim))
        return
    # Submit train task
    task_con, task_cur = train_worker_tf.connect_db()
    try:
        insert_db(task_con, task_cur, "insert into task values (?)", (id,))
    except sqlite3.IntegrityError:
        pass
    train_worker_tf.close_db(task_con, task_cur)


# Calculate avg_length and N and scores
# Has: all length
def ea_number(con: sqlite3.Connection, cur: sqlite3.Cursor, iteration: int):
    # Score all
    cur.execute("select sr.id from generation inner join sr on generation.id=sr.id where iteration=?", (iteration,))
    rows = cur.fetchall()
    id_list = []
    for row in rows:
        id_list.append(row[0])
    ea_code_tf.test_runtime(id_list)
    # avg_length
    cur.execute("select AVG(length) from length")
    row = cur.fetchone()
    avg_length = int(row[0])
    number = avg_length * 2
    cur.execute("select id from number order by id desc limit 1")
    row = cur.fetchone()
    if row is None:
        id = 0
    else:
        id = row[0] + 1
    insert_db(con, cur, "insert into number values (?, ?, ?)", (id, number, avg_length))
    cur.execute("select * from status order by id desc limit 1")
    row = cur.fetchone()
    id = row[0] + 1
    insert_db(con, cur, "insert into status values (?, ?, ?)", (id, 2, iteration))
    return iteration


# Insert row to db with lock wait
def insert_db(con: sqlite3.Connection, cur: sqlite3.Cursor, sql: str, params):
    ok = 0
    while ok == 0:
        try:
            cur.execute(sql, params)
            con.commit()
            ok = 1
        except sqlite3.OperationalError:
            ok = 0
            time.sleep(10)


def ea_loop(con: sqlite3.Connection, cur: sqlite3.Cursor, status_dict):
    iteration = status_dict['iteration']
    if status_dict['status'] == 0:
        iteration = ea_train_score(con, cur, iteration)
        iteration = ea_number(con, cur, iteration)
    elif status_dict['status'] == 1:
        iteration = ea_number(con, cur, iteration)
    while True:
        iteration = ea_select_generate(con, cur, iteration)
        iteration = ea_train_score(con, cur, iteration)
        iteration = ea_number(con, cur, iteration)
