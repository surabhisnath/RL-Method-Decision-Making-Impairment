import numpy as np
import pandas as pd
np.seterr(all='raise')


class Model:
    """Base Model class"""

    def __init__(self, state_list, action_list, rewards, init_state):
        """Initialize all variables used by both Model Free and Model Based"""
        self.state_list = np.array(state_list)
        self.action_list = np.array(action_list)
        self.rewards = np.array(rewards)
        self.state = init_state

    def softmax(self, array, hyperparam, numer=None):
        """
        Returns probs after evaluating Softmax on array
        hyperparameter: Inverse temperature factor
        """
        # Domian change
        # hyperparam = np.log(np.divide(1, 1 - hyperparam))

        num = np.exp(hyperparam * array)
        den = np.sum(num)

        if numer is not None:
            return np.divide(np.exp(hyperparam * numer), den)

        probs = np.divide(num, np.sum(num))
        return probs


class ModelFree(Model):
    """Class for Model Free RL"""

    def __init__(self, model, Q_values, lr, mem, tdf):
        """Initialize variables"""
        super().__init__(model.state_list, model.action_list, model.rewards, model.state)
        # Other initializations like hyperparameters - dict
        self.Q_values = np.array(Q_values)
        self.hyperparams = {"lr": lr,
                            "mem": mem,
                            "tdf": tdf}
        # self.prevstate = -1
        # self.prevaction = -1
        # print("Model Free RL model initialized")

    def algorithm_selector(self, typ, data, sim, choice):
        """Select Algorithm"""
        if choice == 0:
            print("Select Model-Free algorithm to implement: ")
            print("1. Q Learning\n2. SARSA")
            option = int(input())
            # print("algo selector", self.Q_values)
            if option == 1:
                self.Q_learning(typ, data, sim)
            elif option == 2:
                self.SARSA(typ, data, sim)
        elif choice == 1:
            self.Q_learning(typ, data, sim)
        elif choice == 2:
            self.SARSA(typ, data, sim)


    def Q_learning(self, typ, data, sim):
        """
        Perform Q learning and update Q values
        """
        # print("Q Learning")

        action = -1
        next_state = -1

        reward = 0

        if typ == "mle" or typ == "ce":
            action = data[0, 1, sim]
#
            next_state = data[0, 0, sim + 1]
            reward = self.rewards[self.state, action]
            # next_action = data[0, 1, sim + 1]

        # Qlearning Update

        # CONSTRAINED
        td_target = reward + self.hyperparams["tdf"] * max(self.Q_values[next_state])
        td_delta = td_target - self.Q_values[self.state][action]

        self.Q_values[self.state][action] = self.hyperparams["mem"] * self.Q_values[self.state][action] + \
                                            self.hyperparams["lr"] * td_delta


class Synthesizer(Model):
    """Class for Mixing Model Free and Model Based RL"""

    def __init__(self, model, itf, model_free):
        """Initialize variables"""
        super().__init__(model.state_list, model.action_list, model.rewards, model.state)
        # Other initializations like hyperparameters - dict
        itf = min(itf, 0.9999)
        self.hyperparams = {"itf": np.log(np.divide(1, 1 - itf))}
        # self.hyperparams = {"itf": itf}
        self.model_free = model_free
        self.Q_values = np.zeros((len(self.state_list), len(self.action_list)))

        # print("Synthesizer initialized")

    def mixer(self, typ, data, sim):
        """
        Weigh Q values of Model Free and Model Based to find next action
        Return action to be taken
        """

        # CONSTRAINED
        # self.Q_values = self.hyperparams["w"] * self.model_free.Q_values + (
        #             1 - self.hyperparams["w"]) * self.model_based.Q_values

        self.Q_values = self.model_free.Q_values

        if typ == "ce":
            action = data[0, 1, sim]

            # CROSS ENTROPY

            # CONSTRAINED
            return -1 * np.log(self.softmax(self.Q_values[self.state], self.hyperparams["itf"],
                                       self.Q_values[self.state, action]))

        elif typ == "mle":

            # try:
            action = data[0, 1, sim]

            term1 = np.multiply(self.hyperparams["itf"], self.Q_values[self.state, action])
            term2 = 0

            for kk in range(len(self.action_list)):
                # if self.Q_values[self.state][kk] == 0:
                #     continue
                # print(self.hyperparams["itf"], self.Q_values[self.state][kk], self.state, kk)
                term2 += np.exp(np.multiply(self.hyperparams["itf"], self.Q_values[self.state][kk]))
            # print("term2", term2)
            term2 = np.log(term2)
            # print("term2", term2)

            term = term1 - term2
            # print("term", term)
            return -1 * term

            # except:
            #     print(self.hyperparams["itf"], self.Q_values[self.state][kk], self.state, kk)
