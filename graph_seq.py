# gene_units is not included start and end
def get_adjacency_matrix_with_info(gene_units):
    matrix = [[0 for _ in range(len(gene_units) + 2)] for _ in range(len(gene_units) + 2)]
    info = ['start']
    pos = 1
    for gene_unit in gene_units:
        gene_unit_params = [int(i) for i in gene_unit.split(',')]
        if gene_unit_params[0] == 1:
            offset = gene_unit_params[1]
            matrix[max(pos - offset, 0)][pos] = 1
            activation_dict = {0: 'None', 1: 'ReLU', 2: 'Sigmoid'}
            info.append('CNN_c{}k{}g{}_{}'.format(gene_unit_params[3], gene_unit_params[4], gene_unit_params[5],
                                                  activation_dict[gene_unit_params[6]]))
        elif gene_unit_params[0] == 2:
            offset_a = gene_unit_params[1]
            offset_b = gene_unit_params[2]
            matrix[max(pos - offset_a, 0)][pos] = 1
            matrix[max(pos - offset_b, 0)][pos] = 1
            type_dict = {2: '+', 3: '*'}
            info.append('{}'.format(type_dict[gene_unit_params[3]]))
        pos += 1
    matrix[-2][-1] = 1
    info.append('end')
    return matrix, info


def clean_matrix_with_info(matrix, info):
    while True:
        temp = []
        nums = get_bin_nums(matrix)
        nums[-1] = 1
        for i in range(len(nums)):
            if nums[i] == 0:
                info[i] = 'delete'
                continue
            temp.append([])
            for j in range(len(nums)):
                if nums[j] == 0:
                    continue
                temp[-1].append(matrix[i][j])
        while 'delete' in info:
            info.remove('delete')
        if len(temp) == len(matrix):
            return temp, info
        else:
            matrix = temp


def sort_matrix_with_info(matrix, info):
    k = 0
    rc_len = len(matrix)
    while k == 0:
        nums = get_bin_nums(matrix)
        for i in range(rc_len):
            if i == rc_len - 1:
                k = 1
                break
            if nums[i] >= nums[i + 1]:
                continue
            else:
                for j in range(rc_len):
                    temp = matrix[i][j]
                    matrix[i][j] = matrix[i + 1][j]
                    matrix[i + 1][j] = temp
                for j in range(rc_len):
                    temp = matrix[j][i]
                    matrix[j][i] = matrix[j][i + 1]
                    matrix[j][i + 1] = temp
                temp = info[i]
                info[i] = info[i + 1]
                info[i + 1] = temp
                break


def get_bin_num(row):
    res = 0
    for i in row:
        res <<= 1
        res += i
    return res


def get_bin_nums(matrix):
    res = []
    for row in matrix:
        res.append(get_bin_num(row))
    return res


def matrix_seq(matrix):
    res = []
    for i in get_bin_nums(matrix):
        if i > 0:
            res.append(str(i))
    return '-'.join(res)



def gene_graph_seq_with_info(gene_units):
    adj_matrix, info = get_adjacency_matrix_with_info(gene_units)
    adj_matrix, info = clean_matrix_with_info(adj_matrix, info)
    sort_matrix_with_info(adj_matrix, info)
    adj_matrix, info = clean_matrix_with_info(adj_matrix, info)
    return matrix_seq(adj_matrix), info

