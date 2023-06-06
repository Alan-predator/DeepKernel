import os

import tensorflow as tf
import numpy as np
import keras

import random, math
import utils
import network
import model_mut_operators

acc = XX
if __name__ == '__main__':
    utils = utils.GeneralUtils()
    network = network.FCNetwork()

    # (train_datas, train_labels), (test_datas, test_labels) = network.load_data()
    (train_datas, train_labels), (test_datas, test_labels) = network.load_cifar10()
    model = network.load_model("XXX.h5")


    model_mut_opts = model_mut_operators.ModelMutationOperators()

    mutation_ratio = 0.01
    STD = 0.2
    i=0
    while(i<100):
        GF_model = model_mut_opts.GF_mut(model, mutation_ratio, STD=STD)
        GF_model = network.compile_model(GF_model)
        # network.evaluate_mutscore(GF_model,datapath)
        if utils.print_messages_MMM_generators('GF', network=network, test_datas=test_datas, test_labels=test_labels,
                                             model=model, mutated_model=GF_model, STD=STD, mutation_ratio=mutation_ratio) > acc:
            name = 'cifar_GF_mut' + str(i)
            os.chdir('XXX')
            network.save_model(GF_model, name)
            i = i + 1


    mutation_ratio = 0.01
    STD = 0
    i = 0
    while(i<100):
        WS_model = model_mut_opts.WS_mut(model, mutation_ratio)
        WS_model = network.compile_model(WS_model)
        if utils.print_messages_MMM_generators('WS', network=network, test_datas=test_datas, test_labels=test_labels,
                                             model=model, mutated_model=WS_model, STD=STD, mutation_ratio=mutation_ratio) > acc:
            name = 'cifar_WS_mut' + str(i)
            os.chdir('XXX')
            network.save_model(WS_model, name)
            i = i + 1


    mutation_ratio = 0.01
    STD = 0.20
    i=0
    while(i<100):
        NEB_model = model_mut_opts.NEB_mut(model, mutation_ratio)
        NEB_model = network.compile_model(NEB_model)
        if utils.print_messages_MMM_generators('NEB', network=network, test_datas=test_datas, test_labels=test_labels,
                                             model=model, mutated_model=NEB_model, STD=STD, mutation_ratio=mutation_ratio) > acc:
            name = 'cifar_NEB_mut' + str(i)
            os.chdir('XXXX')
            network.save_model(NEB_model, name)
            i = i + 1


    mutation_ratio = 0.01
    STD = 0.2
    i=0
    while(i<100):
        NAI_model = model_mut_opts.NAI_mut(model, mutation_ratio)
        NAI_model = network.compile_model(NAI_model)
        print(utils.print_messages_MMM_generators('NAI', network=network, test_datas=test_datas, test_labels=test_labels,
                                             model=model, mutated_model=NAI_model, STD=STD, mutation_ratio=mutation_ratio))
        if utils.print_messages_MMM_generators('NAI', network=network, test_datas=test_datas, test_labels=test_labels,
                                             model=model, mutated_model=NAI_model, STD=STD, mutation_ratio=mutation_ratio) > acc:
            name = 'cifar_NAI_mut' + str(i)
            os.chdir('XXX')
            network.save_model(NAI_model, name)
            i = i + 1

    mutation_ratio = 0.01
    STD = 0.20
    i=0
    while(i<100):
        NS_model = model_mut_opts.NS_mut(model, mutation_ratio)
        NS_model = network.compile_model(NS_model)
        if utils.print_messages_MMM_generators('NS', network=network, test_datas=test_datas, test_labels=test_labels,
                                             model=model, mutated_model=NS_model, STD=STD, mutation_ratio=mutation_ratio) > acc:
            name = 'cifar_NS_mut' + str(i)
            os.chdir('XXX')
            network.save_model(NS_model, name)
            i = i + 1

