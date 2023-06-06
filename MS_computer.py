import os
import utils
import network
from openpyxl import Workbook, load_workbook


if __name__ == '__main__':
    utils = utils.GeneralUtils()
    network = network.FCNetwork()

    model = network.Cifarmodel()
    model = network.compile_model(model)

    datapath = 'XXX'

    modelpath = 'XXX'
    filenames = os.listdir(modelpath)

    f = open('XXX', 'r+', encoding="utf-8")

    i = 0
    # error_rate_data = []
    for filename in filenames:
        print('-------------',i)
        i = i + 1
        filename = 'XXX' + filename
        model.load_weights(filename)
        error = network.evaluate_mutscore(model, datapath)
        f.write(str(error))
        f.write("\n")




