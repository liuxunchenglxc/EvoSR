import init_ea
import ea
import ea_code_tf

gene_1 = '0-1,1,1,8,3,1,0' \
         '-1,1,1,8,3,1,1-2,1,2,2' \
         '-1,1,1,8,3,1,1-2,1,2,2' \
         '-1,1,1,8,3,1,1-2,1,2,2' \
         '-1,1,1,8,3,1,1-2,1,2,2' \
         '-1,1,1,8,3,1,1-2,1,2,2' \
         '-1,1,1,8,3,1,1-2,1,2,2' \
         '-1,1,1,8,3,1,1-2,1,2,2' \
         '-1,1,1,8,3,1,1-2,1,2,2' \
         '-1,1,1,48,3,1,0-255'
gene_2 = '0-1,1,1,16,3,1,1-1,2,1,28,3,1,1' \
         '-1,1,1,28,3,1,1' \
         '-1,1,1,28,3,1,1' \
         '-1,1,1,28,3,1,1' \
         '-1,1,1,28,3,1,1' \
         '-1,1,1,28,3,1,1' \
         '-1,1,1,48,3,1,0-2,1,8,2-255'
gene_3 = '0-1,1,1,12,3,1,0' \
         '-1,1,1,12,3,1,1' \
         '-1,1,1,12,3,1,1' \
         '-1,1,1,12,3,1,1' \
         '-1,1,1,3,3,1,1' \
         '-1,1,1,1,1,1,0-2,1,15,2' \
         '-1,1,1,12,3,1,1' \
         '-1,1,1,12,3,1,1' \
         '-1,1,1,12,3,1,1' \
         '-1,1,1,3,3,1,1-2,1,4,3' \
         '-2,1,3,2-1,1,1,1,1,1,0-2,1,15,2' \
         '-1,1,1,12,3,1,1' \
         '-1,1,1,12,3,1,1' \
         '-1,1,1,12,3,1,1' \
         '-1,1,1,3,3,1,1' \
         '-1,1,1,1,1,1,0-2,1,15,2' \
         '-1,1,1,12,1,1,1-2,1,47,2-1,1,1,48,3,1,0-255'
gene_4 = '0-1,1,1,8,3,1,0' \
         '-1,1,1,20,3,1,1-1,1,1,12,3,1,1' \
         '-1,1,1,4,3,1,1' \
         '-2,1,4,2-1,1,1,8,3,1,1' \
         '-1,1,1,8,3,1,2-2,1,2,3' \
         '-1,14,1,20,3,1,1-1,1,1,12,3,1,1' \
         '-1,1,1,4,3,1,1' \
         '-1,1,1,8,3,1,1' \
         '-1,1,1,8,3,1,2-2,1,2,3' \
         '-1,1,1,48,3,1,0-255'

if __name__ == '__main__':
    con, cur = init_ea.connect_db()
    status = init_ea.read_status(cur)
    print("Status of ea: {}".format(status))
    # mutant gene test
    new_gene = ea.mutant_gene(gene_1, 1, 25)
    print("\nMutant Test")
    print("Old gene: {}".format(gene_1))
    print("New gene: {}".format(new_gene))
    print("Equal? {}".format(gene_1 == new_gene))

    new_gene = ea.crossover_gene(gene_1, gene_2, 1)
    print("\nCrossover Test")
    print("Old gene: {}".format(gene_1))
    print("Old gene: {}".format(gene_2))
    print("New gene: {}".format(new_gene))
    print("Equal? {}".format(gene_1 == new_gene))

    print("\nFix Input Bounds Test")
    gene_units_params = [['0'], ['1', '5', '1', '8', '3', '1', '1'], ['1', '5', '1', '8', '3', '1', '1'], ['255']]
    print("Old gene: {}".format(gene_units_params))
    new_gene_units_params = ea.fix_gene_input_bounds(gene_units_params)
    print("New gene: {}".format(new_gene_units_params))

    print("\nFix Input Shape Test")
    gene_units_params = [['0'], ['1', '5', '1', '8', '3', '1', '1'], ['1', '5', '1', '4', '3', '1', '1'],
                         ['2', '1', '2', '2'], ['2', '1', '2', '3'], ['255']]
    print("Old gene: {}".format(gene_units_params))
    new_gene_units_params = ea.fix_gene_params(gene_units_params)
    print("New gene: {}".format(new_gene_units_params))

    gene_units_params = [['0'], ['1', '5', '1', '8', '3', '1', '1'], ['1', '5', '1', '4', '3', '1', '1'],
                         ['2', '1', '2', '2'], ['2', '1', '2', '3'], ['1', '1', '1', '8', '3', '1', '1'], ['255']]
    print("Old gene: {}".format(gene_units_params))
    new_gene_units_params = ea.fix_gene_params(gene_units_params)
    print("New gene: {}".format(new_gene_units_params))

    gene_units_params = [['0'], ['1', '1', '1', '28', '3', '1', '2'],
                         ['1', '1', '1', '17', '3', '1', '1'],
                         ['1', '1', '1', '17', '3', '1', '0'],
                         ['2', '1', '2', '2'],
                         ['1', '1', '1', '48', '3', '1', '0'],
                         ['1', '5', '1', '17', '3', '1', '1'],
                         ['2', '2', '1', '2'],
                         ['1', '1', '1', '48', '3', '1', '1'],
                         ['255']]
    print("Old gene: {}".format(gene_units_params))
    new_gene_units_params = ea.fix_gene_params(gene_units_params)
    print("New gene: {}".format(new_gene_units_params))

    print("\nGene To Code Test")
    print("Gene: {}".format(gene_1))
    gene_units = gene_1.split('-')
    gene_units_params = [unit.split(',') for unit in gene_units]
    code, gene_length = ea_code_tf.gene_to_code(ea.fix_gene_params(gene_units_params))
    print("Code:")
    print(code)
    print("Gene Length: {}".format(gene_length))
    print("\nGene: {}".format(gene_2))
    gene_units = gene_2.split('-')
    gene_units_params = [unit.split(',') for unit in gene_units]
    code, gene_length = ea_code_tf.gene_to_code(ea.fix_gene_params(gene_units_params))
    print("Code:")
    print(code)
    print("Gene Length: {}".format(gene_length))
    print("\nGene: {}".format(gene_3))
    gene_units = gene_3.split('-')
    gene_units_params = [unit.split(',') for unit in gene_units]
    code, gene_length = ea_code_tf.gene_to_code(ea.fix_gene_params(gene_units_params))
    print("Code:")
    print(code)
    print("Gene Length: {}".format(gene_length))
    print("\nGene: {}".format(gene_4))
    gene_units = gene_4.split('-')
    gene_units_params = [unit.split(',') for unit in gene_units]
    code, gene_length = ea_code_tf.gene_to_code(ea.fix_gene_params(gene_units_params))
    print("Code:")
    print(code)
    print("Gene Length: {}".format(gene_length))

    print("\nStart EA Process")
    ea.ea_loop(con, cur, status)

    init_ea.close_db(con, cur)
