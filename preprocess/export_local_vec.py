import numpy as np
import pickle
import io


def main():
    local_reps = pickle.load(
        open('trained_models/model_latxb_7/struct_reps_model_latxb_7.pickle', 'rb'))
    local_ofm = pickle.load(
        open('preprocess/ofm_local_reps_latxb.pickle', 'rb'))

    struct_reps = pickle.load(
        open('trained_models/model_latxb_7/struct_reps_model_latxb_7.pickle', 'rb'))
    struct_ofm = pickle.load(
        open('preprocess/ofm_struct_reps_latxb.pickle', 'rb'))

    atom = 'H'
    out_v = io.open('../material-dl/result/local_vecs_10_' +
                    atom + '.tsv', 'w', encoding='utf-8')
    out_st = io.open('../material-dl/result/struct_vecs_10_' +
                     '.tsv', 'w', encoding='utf-8')

    out_vm = io.open('../material-dl/result/local_meta_10_' +
                     atom + '.tsv', 'w', encoding='utf-8')
    out_sm = io.open('../material-dl/result/struct_meta_10_' +
                     '.tsv', 'w', encoding='utf-8')

    for i in range(len(local_reps_ct)):
        sym = ['H' if x == 0 else 'O' for x in at]

        for j in range(len(sym)):
            if sym[j] == atom:
                out_v.write('\t'.join([str(x)
                                       for x in local_reps_ct[i][j]]) + "\n")

                out_vm.write(str(sym[j]) + str(j) + '_' + str(i) + "\n")

        out_st.write('\t'.join([str(x) for x in struct_reps[i]]) + "\n")
        out_sm.write(str(i) + '_' + str(-energy[i]) + "\n")

    out_v.close()
    out_st.close()
    out_vm.close()
    out_sm.close()


if __name__ == "__main__":
    main()
