from Bio import SeqIO
import numpy as np

def to_OHE(sequence):
    matrix = np.zeros((200, 4))
    # ACGT = 0123
    for c in range(len(sequence)):
        if sequence[c] == 'A' or sequence[c] == 'a':
            matrix[c][0] = 1
        elif sequence[c] == 'C' or sequence[c] == 'c':
            matrix[c][1] = 1
        elif sequence[c] == 'G' or sequence[c] == 'g':
            matrix[c][2] = 1
        elif sequence[c] == 'T' or sequence[c] == 't':
            matrix[c][3] = 1
    return matrix

def store_data(input_file, output_file):
    matrix = np.zeros((4, 200))
    fasta_sequences = SeqIO.parse(open(input_file), 'fasta')
    with open(output_file, "a+") as out_file:
        matrix_arr = []
        for fasta in fasta_sequences:
            name, sequence = fasta.id, str(fasta.seq)
            # print(name, sequence)
            matrix = to_OHE(sequence)
            matrix_arr.append(matrix)

        np.savez(output_file, *matrix_arr)

input_file = 'data\\bioinfo\\GM12878.fa'
output_file = 'data\\bioinfo\\GM12878_in.npz'

store_data(input_file, output_file)

matrices = np.load(output_file)

for m in matrices.items():
    print(m)
    break