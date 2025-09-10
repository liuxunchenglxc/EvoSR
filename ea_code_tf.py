import os
import subprocess


# Translate gene to code of tensorflow
# Class name = GEN_SR_MODEL
# For x4 SR 3 channels RGB frame
def gene_to_code(gene_units_params):
    gene_units_params = optim_gene_code(gene_units_params)
    gene_length = 0
    space_8 = "        "
    # Code class head
    class_head = '''class GEN_SR_MODEL(Model):
    def __init__(self):
        super(GEN_SR_MODEL, self).__init__()'''
    # Gene scan
    # Name the pos=p CNN as conv_{p}
    conv_layers = ""
    # Name the pos=p output as x_{p}
    forward_code = ""
    pos = 0
    for unit_params in gene_units_params:
        gene_length = gene_length + 1
        if unit_params[0] == '-1':  # Useless Unit
            pos = pos + 1
            gene_length = gene_length - 1
            continue
        if unit_params[0] == '0':  # Start Unit
            pos = pos + 1
            continue
        if unit_params[0] == '255':  # End Unit
            forward_code = forward_code + space_8 + "x = tf.nn.depth_to_space(x_{}, 4)\n".format(pos - 1)
            forward_code = forward_code + space_8 + "return x\n"
            break
        input_num = int(unit_params[0])
        if input_num == 1:
            input_name = 'x_' + str(max(0, pos - int(unit_params[1])))
            unit_type = int(unit_params[2])
            if unit_type == 1:  # CNN
                activation_num = int(unit_params[6])
                if activation_num == 1:
                    activation = "\"relu\""
                elif activation_num == 2:
                    activation = "\"sigmoid\""
                else:
                    activation = "None"
                conv_layers = conv_layers + space_8 + "self.conv_{} = Conv2D({}, {}, groups={}, padding=\"same\", activation={})\n".format(
                    pos, unit_params[3], unit_params[4], unit_params[5], activation)
                forward_code = forward_code + space_8 + "x_{} = self.conv_{}({})\n".format(pos, pos, input_name)
        elif input_num == 2:
            input_name_a = 'x_' + str(max(0, pos - int(unit_params[1])))
            input_name_b = 'x_' + str(max(0, pos - int(unit_params[2])))
            unit_type = int(unit_params[3])
            if unit_type == 2:  # +
                forward_code = forward_code + space_8 + "x_{} = {} + {}\n".format(pos, input_name_a, input_name_b)
            elif unit_type == 3:  # *
                forward_code = forward_code + space_8 + "x_{} = {} * {}\n".format(pos, input_name_a, input_name_b)
        pos = pos + 1
    # Link code
    code = "{}\n{}\n    @tf.function\n    def call(self, x_0):\n{}".format(class_head, conv_layers, forward_code)
    return code, gene_length


# Mark useless unit as -1
def optim_gene_code(gene_units_params):
    id_pos_mark = [[0, 1]]
    id = 0
    # scan for pos
    for unit_params in gene_units_params:
        if unit_params[0] == '0':
            id = id + 1
            continue
        if unit_params[0] == '255':
            id_pos_mark[-1][1] = 1
            break
        id_pos_mark.append([id, 0])
        id = id + 1
    # mark input pos and mark useless unit
    offset_base = 0
    for i in range(len(gene_units_params) - 1, -1, -1):
        if gene_units_params[i][0] == '0':
            break
        if gene_units_params[i][0] == '255':
            continue
        try:
            offset_base = offset_base + 1
            id_pos_mark.index([i, 1])  # marked unit has the quality to mark other units
            for j in range(int(gene_units_params[i][0])):
                offset = int(gene_units_params[i][1 + j])
                id_pos_mark[-offset - offset_base][1] = 1
        except ValueError:  # Unit output only use by after unit, so this unit is useless
            gene_units_params[i][0] = '-1'
    return gene_units_params


# Insert code to train and score framework, write it to file, and return filename, name.
def code_to_file(id: int, code: str):
    file_head = '''import tensorflow as tf

from tensorflow.keras.layers import Conv2D
from tensorflow.keras import Model'''
    file_content = file_head + "\n\n\n" + code
    if not os.path.exists("./models/"):
        os.mkdir("./models/")
        os.mknod("./models/__init__.py")
    with open('./models/model_{}.py'.format(id), 'w', encoding='utf-8') as f:
        f.write(file_content)
    return 'model_{}.py'.format(id), str(id)

# Test models inference time
def test_runtime(id_list):
    gpu_idx = 4
    test_script_head = '''import sqlite3
import score
import tensorflow as tf
import time
'''
    test_script_import_model = ''
    for id in id_list:
        test_script_import_model += 'import models.model_{} as model_{}\n'.format(id, id)
    test_script_middle = '''
gpu_idx = {}
if gpu_idx >= 0:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices(gpus[gpu_idx], 'GPU')
input = tf.ones([1, 180, 320, 3])
con = sqlite3.connect("EA.db")
cur = con.cursor()
'''.format(gpu_idx)
    test_script_runtime = ''
    for id in id_list:
        test_script_runtime += '''
runtime_time = 0
with tf.device('/device:GPU:0'):
    model = model_{}.GEN_SR_MODEL()
    model(input)
    for _ in range(10):
        time_a = time.perf_counter()
        model(input)
        time_b = time.perf_counter()
        runtime_time = runtime_time + time_b - time_a
runtime_time = runtime_time / 10
cur.execute("insert into runtime values (?, ?)", ({}, runtime_time))
cur.execute("select psnr,ssim from sr where id=?", ({},))
row = cur.fetchone()
model_score = score.score_sr(row[0], row[1], runtime_time)
cur.execute("insert into score values (?, ?)", ({}, model_score))
'''.format(id, id, id, id, id)
    test_script_end = '''
cur.close()
con.commit()
con.close()
'''
    test_script = test_script_head + test_script_import_model + test_script_middle + test_script_runtime + test_script_end
    with open('./test_runtime_script.py', 'w', encoding='utf-8') as f:
        f.write(test_script)
    with open('./test_runtime_script.out', 'w', encoding='utf-8') as f:
        return_code = subprocess.call(['python', '-u', 'test_runtime_script.py'], stdout=f, stderr=f)
    if return_code != 0:
        print("ERROR: Test runtime is failed, please check output log at ./test_runtime_script.out")
    return return_code
