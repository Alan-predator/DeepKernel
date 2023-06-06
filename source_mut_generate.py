import os
import tensorflow as tf
import numpy as np
import keras
import random, math
import utils

import network
import source_mut_operators

acc = XX

if __name__ == '__main__':
    utils = utils.GeneralUtils()
    network = network.FCNetwork()

    # (train_datas, train_labels), (test_datas, test_labels) = network.load_data()
    (train_datas, train_labels), (test_datas, test_labels) = network.load_cifar10()

    # 加载模型
    # model = network.Lenet1()
    # model = network.Lenet5()
    model = network.Cifarmodel()

    source_mut_opts = source_mut_operators.SourceMutationOperators()

    mutation_ratio = 0.01
    i = 0
    while (i < 100):
        (DR_train_datas, DR_train_labels), DR_model = source_mut_opts.DR_mut((train_datas, train_labels), model, mutation_ratio)
        utils.print_messages_SMO('DR', train_datas=train_datas, train_labels=train_labels, mutated_datas=DR_train_datas,
                                 mutated_labels=DR_train_labels, mutation_ratio=mutation_ratio)
        DR_model = network.compile_model(DR_model)
        DR_model = network.train_model(DR_model, DR_train_datas, DR_train_labels)
        if network.evaluate_model(DR_model, test_datas, test_labels) > acc:
            name = 'cifar_DR_mut' + str(i)
            os.chdir('XXX')
            network.save_model(DR_model, name)
            i = i + 1


    mutation_ratio = 0.01
    i = 0
    while (i < 100):
        (LE_train_datas, LE_train_labels), LE_model = source_mut_opts.LE_mut((train_datas, train_labels), model, 0, 9,
                                                                             mutation_ratio)
        mask_equal = LE_train_labels == train_labels
        mask_equal = np.sum(mask_equal, axis=1) == 10
        count_diff = len(train_labels) - np.sum(mask_equal)
        LE_model = network.compile_model(LE_model)
        LE_model = network.train_model(LE_model, LE_train_datas, LE_train_labels)
        if network.evaluate_model(LE_model, test_datas, test_labels) > acc:
            name = 'cifar_LE_mut' + str(i)
            os.chdir('XXX')
            network.save_model(LE_model, name)
            i = i + 1



    mutation_ratio = 0.01
    i = 0
    while (i < 100):
        (DM_train_datas, DM_train_labels), DM_model = source_mut_opts.DM_mut((train_datas, train_labels), model,
                                                                             mutation_ratio)
        DM_model = network.compile_model(DM_model)
        DM_model = network.train_model(DM_model, DM_train_datas, DM_train_labels)
        if network.evaluate_model(DM_model, test_datas, test_labels) > acc:
            name = 'cifar_DM_mut' + str(i)
            os.chdir('XXX')
            network.save_model(DM_model, name)
            i = i + 1


    mutation_ratio = 0.01
    i = 0
    while (i < 100):
        (DF_train_datas, DF_train_labels), DF_model = source_mut_opts.DF_mut((train_datas, train_labels), model,
                                                                             mutation_ratio)
        DF_model = network.compile_model(DF_model)
        DF_model = network.train_model(DF_model, DF_train_datas, DF_train_labels)
        if network.evaluate_model(DF_model, test_datas, test_labels) > acc:
            name = 'cifar_DF_mut' + str(i)
            os.chdir('XXX')
            network.save_model(DF_model, name)
            i = i + 1



    mutation_ratio = 0.01
    STD = 1
    i = 0
    while (i < 100):
        (NP_train_datas, NP_train_labels), NP_model = source_mut_opts.NP_mut((train_datas, train_labels), model,
                                                                             mutation_ratio, STD=STD)
        NP_model = network.compile_model(NP_model)
        NP_model = network.train_model(NP_model, NP_train_datas, NP_train_labels)
        if network.evaluate_model(NP_model, test_datas, test_labels) > acc:
            name = 'cifar_NP_mut' + str(i)
            os.chdir('XXX')
            network.save_model(NP_model, name)
            i = i + 1


    i = 0
    while (i < 100):
        (LAs_train_datas, LAs_train_labels), LAs_model = source_mut_opts.LAs_mut((train_datas, train_labels), model)
        LAs_model = network.compile_model(LAs_model)
        LAs_model = network.train_model(LAs_model, train_datas, train_labels)
        if network.evaluate_model(LAs_model, test_datas, test_labels) > acc:
            name = 'cifar_LAs_mut2_' + str(i)
            os.chdir('XXX')
            network.save_model(LAs_model, name)
            i = i + 1

    i = 0
    while (i < 100):
        (AFRs_train_datas, AFRs_train_labels), AFRs_model = source_mut_opts.AFRs_mut((train_datas, train_labels), model)
        AFRs_model = network.compile_model(AFRs_model)
        AFRs_model = network.train_model(AFRs_model, AFRs_train_datas, AFRs_train_labels)
        if network.evaluate_model(AFRs_model, test_datas, test_labels) > acc:
            name = 'cifar_AFRs_mut' + str(i)
            os.chdir('XXX')
            network.save_model(AFRs_model, name)
            i = i + 1
