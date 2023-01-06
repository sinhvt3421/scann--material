from trained_models.model_ptcnt_au_graphene_26k_norm_ref_total_energy.model_dev import create_model_infer
# from model.model_latxb import create_model_infer
import numpy as np
from sklearn.model_selection import train_test_split
from utils.datagenerator import DataIterator
import tensorflow as tf
import os
from ase.db import connect
import yaml
import pickle
import argparse

# tf.keras.backend.set_floatx(
#     'float64'
# )


def schedule(epoch):
    lr = 1e-4
    if epoch > 0:
        lr = 0.001  # learning rate
    if epoch > 70:
        lr = 1e-4
    if epoch > 500:
        lr = 5e-5
    return lr


def main(args):
    config = yaml.safe_load(
        open('trained_models/model_ptcnt_au_graphene_26k_norm_ref_total_energy/config.yaml'))

    model = create_model_infer(config)
    model.load_weights(
        'trained_models/model_ptcnt_au_graphene_26k_norm_ref_total_energy/models/model-577.h5')

    print('Load data neighbors for dataset ')
    all_data = np.load(config['hyper']['data_nei_path'], allow_pickle=True)
    # all_data = []
    # data_neigh = np.load('preprocess/ptcnt/no_pt_graphene_data_voroinn_neigh_1k8.npy', allow_pickle=True)
    # data_neigh = np.load(config['hyper']['data_nei_path'], allow_pickle=True)
    # all_data.extend(data_neigh)
    # data_neigh = np.load('preprocess/ptcnt/mix_pt_graphene_optimization_data_voronoi_neigh_184.npy', allow_pickle=True)
    # all_data.extend(data_neigh)


    all_data = np.array(all_data, dtype='object')

    print('Load data target: ', 'lumo')
    # data_full = np.load('preprocess/ptcnt/mix_pt_graphene_optimization_184.npy', allow_pickle=True)
    data_full = np.load(
        config['hyper']['data_energy_path'], allow_pickle=True)

    data_energy = []
    for d in data_full:
        if args.use_ring:
            data_energy.append([d['Atomic'], d['Properties']
                                ['u0'], d['Ring'], d['Aromatic']])
        else:
            data_energy.append([d['Atomic'], d['Properties']
                                ['total_energy']])

    data_energy = np.array(data_energy, dtype='object')

    # if config['model']['use_ofm']:
    #     data_ofm_raw = pickle.load(open(
    #         config['hyper']['data_ofm_path'], 'rb'))
    #     struct_ofm = np.array(pickle.load(open(
    #         config['hyper']['struct_ofm_path'], 'rb')))
    #     # print(data_energy[0])
    #     idx = ~np.all(struct_ofm == 0, axis=0)
    #     data_ofm = []
    #     for ofm in data_ofm_raw:
    #         data_ofm.append(ofm[:, idx])
    #     data_ofm = np.array(data_ofm)
    # else:
    #     data_ofm = []
    data_ofm = []
    print(len(all_data))

    test = DataIterator(type='train', batch_size=config['hyper']['batch_size'],
                        indices=range(len(all_data)), data_neigh=all_data,
                        data_energy=data_energy, data_ofm=data_ofm,
                        use_ofm=False, converter=True, use_ring=False)

    local_reps = []
    attn_atoms = []
    local_ofm = []
    struct_reps = []
    struct_energy = []
    final_embed = []
    # for i in range(0, len(all_data), 32):
    #     instance, op = test._get_batches_of_transformed_samples(
    #         range(i, min(i+32,len(all_data))))
    #     energy, context, attn_local, attn_global, struc_rep, dense_embed = model.predict(
    #         instance)
    #     # energy, context, attn, struct,_     = model.predict(
    #     #     instance)
    #     local_reps.append(context)
    #     struct_energy.append(energy)
    #     attn_atoms.append(attn_global)
    #     struct_reps.append(struc_rep)
    #     final_embed.append(attn_local)
    #     if i % 100 == 0:
    #         print(i)

    test_idx = [5436, 5948]
    instance, op = test._get_batches_of_transformed_samples(test_idx)
    print('Predict example')
    energy, context, attn_local, attn_global, struc_rep, dense_embed = model.predict(
            instance)
        # energy, context, attn, struct,_     = model.predict(
        #     instance)
    local_reps.append(dense_embed)
    struct_energy.append(energy)
    attn_atoms.append(attn_global)
    struct_reps.append(struc_rep)
    final_embed.append(attn_local)

    # np.save('attn_atoms.npy',attn_atoms)
    pickle.dump(attn_atoms, open(
        'trained_models/model_ptcnt_au_graphene_26k_norm_ref_total_energy/attn_score_au_norm_total_energy_5436-5948.pickle', 'wb'))
        
    # pickle.dump(struct_reps, open(
    #     'trained_models/model_ptcnt_au_graphene_26k_norm_ref_total_energy/struct_reps_model_qm9_full_100.pickle', 'wb'))

    pickle.dump(local_reps, open(
        'trained_models/model_ptcnt_au_graphene_26k_norm_ref_total_energy/local_reps_model_ptcnt_5436-5948.pickle', 'wb'))

    # pickle.dump(struct_energy, open(
    #     'trained_models/model_ptcnt_au_graphene_26k_norm_ref_total_energy/energy_pre_model_pt_au_norm_total_energy_577_opt.pickle', 'wb'))

    pickle.dump(final_embed, open(
        'trained_models/model_ptcnt_au_graphene_26k_norm_ref_total_energy/attn_agg_full_ptcnt_5436-5948.pickle', 'wb'))

    np.save('trained_models/model_ptcnt_au_graphene_26k_norm_ref_total_energy/structure_neighbor_5436-5948.npy',all_data[test_idx])



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    # parser.add_argument('target', type=str,
    #                     help='Target energy for training')
    # parser.add_argument('dataset', type=str, help='Path to dataset configs')
    parser.add_argument('--use_ofm', type=bool, default=False,
                        help='Whether to use ofm as extra embedding')
    parser.add_argument('--use_ring', type=bool, default=False,
                        help='Whether to use ring as extra emedding')
    args = parser.parse_args()
    main(args)
