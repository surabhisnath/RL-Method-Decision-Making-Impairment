import numpy as np
from RLDMToolkit import Model, ModelFree, Synthesizer
import threading
from scipy.optimize import minimize
from scipy.optimize import Bounds
from cvxopt import matrix, solvers
np.random.seed(3)
from numpy import linalg as LA
from pyDOE import *
from scipy.stats import uniform
import pandas as pd
import xlsxwriter
import random
import matplotlib.pyplot as plt

def initialize(lr=0.5, mem=0.5, tdf=0.5, itf_synth=0.2):

    Q_upperlim = 1
    init_Q_values_mf = np.random.uniform(0, Q_upperlim, size=(num_states, num_actions))

    # TEMP
    # init_Q_values_mf = np.zeros((num_states, num_actions))

    model = Model(state_list, action_list, rewards, init_state)

    # Initialize Model Free model
    modelfree = ModelFree(model, init_Q_values_mf, lr, mem, tdf)

    synthesizer = Synthesizer(model, itf_synth, modelfree)

    return modelfree, synthesizer


def model_mle(h, datamle, rews, Qdict):
    # print(h)
    temp = h.copy()
    h = np.clip(h, 0.0, 1.0)
    likelihood = 0
    tempvar = 0

    modelfree, synthesizer = initialize(*h)

    assert np.array_equal(modelfree.rewards, rews)

    for sim in range(num_simulations - 1):

        modelfree.algorithm_selector(option, datamle, sim, 1)  # 1 for Q_learning, 2 for SARSA
        tempvar += modelfree.hyperparams["lr"] * modelfree.rewards[modelfree.state, datamle[0, 1, sim]]

        Qval_softmax = synthesizer.mixer(option, datamle, sim)

        likelihood += Qval_softmax

        next_state = datamle[0, 0, sim + 1]

        modelfree.state = next_state
        synthesizer.state = next_state
        current_state = next_state

    # print("Negative Log Likelihood", likelihood)

    # CONSTRAINED
    hyperparamarray = np.array([h[0], h[1], h[2], h[3]], dtype=float)

    # lambda_reg = 10
    # regressed = likelihood - lambda_reg * LA.norm(hyperparamarray)

    regressed = likelihood
    # print(regressed, h)
    temp1 = temp.copy().tostring()
    temp2 = synthesizer.Q_values.copy().tostring()
    Qdict[temp1] = temp2
    return regressed


def perform_mle(pid):
    datamle = data[:, pid, :, :]
    # print(data[:, pid, 0])

    # Latin Hypercube Sampling REMOVED
    # # Latin Hypercube Sampling
    # num_parameters = 4
    # num_samples = 2
    # design = lhs(num_parameters, samples=num_samples)
    # for i in range(num_parameters):
    #     design[:, i] = uniform().ppf(design[:, i])
    # # print(design)

    bounds = Bounds([0, 0, 0, 0], [1, 1, 1, 1])  # After domain change from [0,inf) to [0,1]
    stepsize = steps

    funval = 0
    predhyper = []
    optQ = []

    Qdict = {}

    hyperparams = [0.7, 0.9, 0.9, 1 - (1/np.e)]  # Hardcode values for ideal normal human as initial hyperparameters

    # CONSTRAINED
    optimum = minimize(model_mle, np.array(hyperparams), args=(datamle, rewards, Qdict), method='trust-constr',
                       bounds=bounds,
                       options={'xtol': tol, 'gtol': tol, 'barrier_tol': tol, 'verbose': 0, 'maxiter': maxi,
                                'finite_diff_rel_step': stepsize})
    # optimum = minimize(model_mle, np.array(hyperparams), args=(datamle, rewards), method='SLSQP',
    #                    bounds=bounds,
    #                    options={'ftol': 1e-01, 'disp': True, 'maxiter': 1000,
    #                             'eps': stepsize})

    # print(optimum)

    # CONSTRAINED
    if optimum.success:
        predhyper = optimum.x
        temp = predhyper.copy().tostring()
        funval = optimum.fun
        optQ = (np.fromstring(Qdict[temp])).reshape(num_states, num_actions)

    return np.clip(predhyper, 0.0, 1.0), funval, optQ


def find_stats(dep):
    if dep == 1:
        length = len(deparr)
        # fs = (15, 15)
        arr = deparr
        toprint = "DEPRESSED, " + "WITH EMOTION"

    elif dep == 0:
        length = len(nondeparr)
        # fs = (20, 20)
        arr = nondeparr
        toprint = "NON DEPRESSED, " + "WITH EMOTION"

    meanmean = 0
    meanvar = 0
    meanstd = 0

    # fig, a = plt.subplots(1, length, squeeze=False, figsize=fs)
    cnt = 0

    for i in arr:
        string = "../output/Q_participant_" + str(i) + "_withemotion.xlsx"
        Q = pd.read_excel(string)
        Q = np.array(Q)
        #         Q = Q[:,1:]
        mean = np.mean(Q)
        var = np.var(Q)
        std = np.std(Q)
        meanmean += mean
        meanvar += var
        meanstd += std
        # a[0][cnt].imshow(Q, cmap='hot', interpolation='nearest')
        # a[0][cnt].set_title("Participant " + str(i) + "\n" + "Mean = " + str(mean.round(2)) + "\n" + "Var = " + str(
        #     var.round(2)) + "\n" + "Std = " + str(std.round(2)))
        cnt += 1

    print()
    # print(toprint)
    # print("Mean mean =", meanmean / length)
    # print("Mean variance =", meanvar / length)
    # print("Mean standard deviation =", meanstd / length)
    # plt.show()
    return meanmean / length, meanvar / length, meanstd / length

if __name__ == '__main__':

    # dig = 4  # Set to 4 constantly
    # possiblestepss = [0.1, 0.01, 0.001, 0.0001]
    # steps = random.randint(0, 3)
    # steps = possiblestepss[steps]
    # steps = 0.1
    # possibletols = [0.1, 0.01, 0.001, 0.0001]
    # tol = random.randint(0, 3)
    # tol = possibletols[tol]
    # tol = 0.1
    maxi = 10000

    deparr = [1, 5, 6, 9, 11, 12, 13, 20]
    nondeparr = [2, 3, 4, 8, 14, 15, 16, 17, 18, 21]
    depressed = {1: 1, 5: 1, 6: 1, 9: 1, 11: 1, 12: 1, 13: 1, 20: 1, 2: 0, 3: 0, 4: 0, 8: 0, 14: 0, 15: 0, 16: 0, 17: 0,
                 18: 0, 21: 0}
    depfnvalavg = 0
    nondepfnvalavg = 0
    deplravg = 0
    nondeplravg = 0
    depmemavg = 0
    nondepmemavg = 0
    deptdfavg = 0
    nondeptdfavg = 0
    depitfavg = 0
    nondepitfavg = 0

    depvsnondepfn = open('../output/output_dep_vs_nondep_stats_withemotion.txt', 'a')
    # depvsnondepfn.write("Random Q \n\n")

    name = '../output/hyperparameters_withemotion.csv'
    workbook = xlsxwriter.Workbook(name + '.xlsx')
    worksheet = workbook.add_worksheet()
    data_format = workbook.add_format({'bg_color': '#FFFF00'})
    worksheet.write(0, 0, "Participant_ID")
    worksheet.write(0, 1, "Participant_Num")
    worksheet.write(0, 2, "Funcval")
    worksheet.write(0, 3, "lr")
    worksheet.write(0, 4, "mem")
    worksheet.write(0, 5, "tdf")
    worksheet.write(0, 6, "itf")

    outfile = open(name + ".csv", 'w')
    outfile.write("Participant_ID,Participant_Num,Funcval,lr,mem,tdf,itf\n")

    num_data = 1
    num_participants = 18
    num_simulations = 102

    data = np.zeros((num_data, num_participants, 2, num_simulations), dtype=int)

    # print("MODEL FITTING")

    pidtopnum = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 8, 7: 9, 8: 11, 9: 12, 10: 13, 11: 14, 12: 15, 13: 16, 14: 17, 15: 18, 16: 20, 17: 21}

    # num_states = dig
    # num_actions = dig
    num_states = 6
    num_actions = 3
    state_list = np.arange(num_states)
    action_list = np.arange(num_actions)

    for i in range(num_participants):
        key = pidtopnum[i]
        Qdict = {}

        # filename = "../data/data_participant_" + str(key) + "_nanremoved_digitized" + str(dig) + ".csv"
        filename = "../data/data_participant_" + str(key) + "_withemotion.csv"

        pdata = pd.read_csv(filename)
        # data_state = list(pdata['learning_difficulty_ma'])
        # data_action = list(pdata['learning_effort_ma'])
        data_state = list(pdata['state_2dif3emot_interp'])
        data_action = list(pdata['learning_effort_cat_interp'])

        # state_list = pdata.learning_difficulty_ma.unique()
        state_list = np.arange(num_states)
        # statetosid = dict(zip(state_list, range(0, len(state_list))))

        # action_list = pdata.learning_effort_ma.unique()
        action_list = np.arange(num_actions)
        # actiontoaid = dict(zip(action_list, range(0, len(action_list))))

        states = list(map(lambda x: int(x), data_state))
        actions = list(map(lambda x: int(x), data_action))

        data[0, i, 0] = states
        data[0, i, 1] = actions

    for i in range(num_participants):
        key = pidtopnum[i]

        print("Participant", i, key)

        # filename = "../data/data_participant_" + str(key) + "_nanremoved_digitized" + str(dig) + ".csv"
        filename = "../data/data_participant_" + str(key) + "_withemotion.csv"

        pdata = pd.read_csv(filename)

        # data_state = list(pdata['learning_difficulty_ma'])
        # data_action = list(pdata['learning_effort_ma'])
        # data_reward = list(pdata['learning_newreward_ma'])

        data_state = list(pdata['state_2dif3emot_interp'])
        data_action = list(pdata['learning_effort_cat_interp'])
        data_reward = list(pdata['learning_reward_ma_interp'])

        data_state = list(map(lambda x: int(x), data_state))
        data_action = list(map(lambda x: int(x), data_action))
        data_reward = list(map(lambda x: int(x), data_reward))

        assert len(data_state) == len(data_action) == len(data_reward)  # this is equivalent to num_simulations
        num_simulations = len(data_state)
        # print("NUMSIM =", num_simulations)

        rewards = np.zeros((num_states, num_actions))

        for l in range(num_simulations):
            state = data_state[l]
            action = data_action[l]
            rewards[state, action] = data_reward[l]/100

        init_state = data_state[0]

        option = "mle"

        predhyper, funval, optQ = perform_mle(i)

        h1 = predhyper[0]
        h2 = predhyper[1]
        h3 = predhyper[2]
        # h4 = predhyper[3]
        h4 = np.log(np.divide(1, 1 - predhyper[3]))

        # h = [h1, h2, h3, h4, h5, h6]

        # print("funcval =", funval)
        # print("lr =", h1)
        # print("mem =", h2)
        # print("tdf =", h3)
        # print("itf =", h4)

        if key in deparr:
            depfnvalavg += funval
        elif key in nondeparr:
            nondepfnvalavg += funval

        if key in deparr:
            depfnvalavg += funval
        elif key in nondeparr:
            nondepfnvalavg += funval

        if key in deparr:
            deplravg += h1
        elif key in nondeparr:
            nondeplravg += h1

        if key in deparr:
            depmemavg += h2
        elif key in nondeparr:
            nondepmemavg += h2

        if key in deparr:
            deptdfavg += h3
        elif key in nondeparr:
            nondeptdfavg += h3

        if key in deparr:
            depitfavg += h4
        elif key in nondeparr:
            nondepitfavg += h4

        outfile.write(str(i) + "," + str(key) + "," + str(funval) + "," + str(h1) + "," + str(h2) + "," + str(h3) + "," + str(h4) + "\n")

        if key in deparr:
            worksheet.set_row(i+1, cell_format=data_format)
        worksheet.write(i+1, 0, i)
        worksheet.write(i+1, 1, key)
        worksheet.write(i+1, 2, funval)
        worksheet.write(i+1, 3, h1)
        worksheet.write(i+1, 4, h2)
        worksheet.write(i+1, 5, h3)
        worksheet.write(i+1, 6, h4)

        fn = "Q_participant_" + str(key) + "_withemotion"
        df = pd.DataFrame(optQ)
        string = "../output/" + fn + ".xlsx"
        df.to_excel(excel_writer=string)

    outfile.close()
    workbook.close()

    depvsnondepfn.write("Stepsize = " + str(steps) + " " + "Tolerance = " + str(tol) + " " + "Maxiter = " + str(maxi) + "\n\n")

    depvsnondepfn.write("\tHyperparameter Analysis\n\n")
    depvsnondepfn.write("\t\tNegative Log Likelihood Fn Value Statistics\n\n")
    print("Mean Fn Val Depressed =", depfnvalavg/len(deparr))
    depvsnondepfn.write("\t\t\tMean Fn Val Depressed = " + str(depfnvalavg/len(deparr)) + "\n")
    print("Mean Fn Val Non-depressed =", nondepfnvalavg/len(nondeparr))
    depvsnondepfn.write("\t\t\tMean Fn Val Non-depressed = " + str(nondepfnvalavg/len(nondeparr)) + "\n")
    print()
    depvsnondepfn.write("\n")
    depvsnondepfn.write("\t\tLearning Rate Statistics\n\n")
    print("Mean lr Depressed =", deplravg/len(deparr))
    depvsnondepfn.write("\t\t\tMean lr Depressed = " + str(deplravg/len(deparr)) + "\n")
    print("Mean lr Non-depressed =", nondeplravg/len(nondeparr))
    depvsnondepfn.write("\t\t\tMean lr Non-depressed = " + str(nondeplravg/len(nondeparr)) + "\n")
    print()
    depvsnondepfn.write("\n")
    depvsnondepfn.write("\t\tMemory Factor Statistics\n\n")
    print("Mean mem Depressed =", depmemavg/len(deparr))
    depvsnondepfn.write("\t\t\tMean mem Depressed = " + str(depmemavg/len(deparr)) + "\n")
    print("Mean mem Non-depressed =", nondepmemavg/len(nondeparr))
    depvsnondepfn.write("\t\t\tMean mem Non-depressed = " + str(nondepmemavg/len(nondeparr)) + "\n")
    print()
    depvsnondepfn.write("\n")
    depvsnondepfn.write("\t\tTemporal Discount Factor Statistics\n\n")
    print("Mean tdf Depressed =", deptdfavg / len(deparr))
    depvsnondepfn.write("\t\t\tMean tdf Depressed = " + str(deptdfavg/len(deparr)) + "\n")
    print("Mean tdf Non-depressed =", nondeptdfavg / len(nondeparr))
    depvsnondepfn.write("\t\t\tMean tdf Non-depressed = " + str(nondeptdfavg/len(nondeparr)) + "\n")
    print()
    depvsnondepfn.write("\n")
    depvsnondepfn.write("\t\tInverse Temperature Factor Statistics\n\n")
    print("Mean itf Depressed =", depitfavg / len(deparr))
    depvsnondepfn.write("\t\t\tMean itf Depressed = " + str(depitfavg/len(deparr)) + "\n")
    print("Mean itf Non-depressed =", nondepitfavg / len(nondeparr))
    depvsnondepfn.write("\t\t\tMean itf Non-depressed = " + str(nondepitfavg/len(nondeparr)) + "\n\n")

    depvsnondepfn.write("\tQ values Analysis\n\n")

    meanQmeandep, meanQvardep, meanQstddep = find_stats(1)
    meanQmeannondep, meanQvarnondep, meanQstdnondep = find_stats(0)

    depvsnondepfn.write("\t\tQ_mean Statistics\n\n")
    depvsnondepfn.write("\t\t\tMean Q_mean Depressed = " + str(meanQmeandep) + "\n")
    print("Mean Q_mean Depressed = " + str(meanQmeandep))
    depvsnondepfn.write("\t\t\tMean Q_mean Non-dperessed = " + str(meanQmeannondep) + "\n\n")
    print("Mean Q_mean Non-depressed = " + str(meanQmeannondep))
    print()

    depvsnondepfn.write("\t\tQ_variance Statistics\n\n")
    depvsnondepfn.write("\t\t\tMean Q_variance Depressed = " + str(meanQvardep) + "\n")
    print("Mean Q_variance Depressed = " + str(meanQvardep))
    depvsnondepfn.write("\t\t\tMean Q_variance Non-depressed = " + str(meanQvarnondep) + "\n\n")
    print("Mean Q_variance Non-depressed = " + str(meanQvarnondep))
    print()

    depvsnondepfn.write("\t\tQ_stdev Statistics\n\n")
    depvsnondepfn.write("\t\t\tMean Q_stdev Depressed = " + str(meanQstddep) + "\n")
    print("Mean Q_stdev Depressed = " + str(meanQstddep))
    depvsnondepfn.write("\t\t\tMean Q_stdev Non-depressed = " + str(meanQstdnondep) + "\n")
    print("Mean Q_stdev Non-depressed = " + str(meanQstdnondep))
    print("---------------------------------------------------------------------\n")
    depvsnondepfn.write("---------------------------------------------------------------------\n")
    depvsnondepfn.close()
