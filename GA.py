# ignore warnings
from datetime import datetime
import numpy as np
# from sklearn.metrics import recall_score
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor as xgbmodel
import pandas as pd
from sklearn.model_selection import train_test_split
import math
import copy
from operator import itemgetter
import random
import warnings
import os
import time
import csv
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


class GA(object):
    def __init__(self, percentage_split, percentage_back_test,
                 split_training_data, fixed_splitted_data, shuffle_gen, tmp):
        self.features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        self.target_feature = ['Outcome']
        self.percentage_split = percentage_split
        self.percentage_back_test = percentage_back_test
        self.shuffle_gen = shuffle_gen
        self.dataset_csv = 'data/pima_indians_diabetes.csv'
        self.split_training_data = split_training_data
        self.fixed_splitted_data = fixed_splitted_data
        self.number_of_minor_dataset = self.split_data()
        self.tmp = tmp
        self.gen = 1
        self.count_gen = 0
        self.model = xgbmodel()

    def split_data(self):
        if self.split_training_data:
            dataset = pd.read_csv(self.dataset_csv)
            pivot_split_train_test = int(0.8*(len(dataset)))
            dataset_train = dataset[0: pivot_split_train_test]
            dataset_train.to_csv('data/csv/ga/dataset_train.csv')
            dataset_test = dataset[pivot_split_train_test:]
            dataset_test.to_csv('data/csv/ga/dataset_test.csv')
            if 100 % self.percentage_split == 0:
                number_of_minor_dataset = int(100 / self.percentage_split)
            else:
                number_of_minor_dataset = int(100 / self.percentage_split) + 1
            if self.fixed_splitted_data:
                pivot = int(self.percentage_split * len(dataset_train) / 100)
                for i in range(number_of_minor_dataset - 1):
                    tmp_dataset = dataset_train.iloc[i * pivot:(i + 1) * pivot].copy()
                    tmp_dataset.to_csv('data/csv/ga/dataset_{}.csv'.format(i +
                                                                           1))

                tmp_dataset = dataset_train.iloc[pivot *
                                           (number_of_minor_dataset - 1):]
                tmp_dataset.to_csv('data/csv/ga/dataset_{}.csv'.format(
                    number_of_minor_dataset))

        return number_of_minor_dataset

    def fitness_shuffle_gen(self, gen_array):
        input_features = []
        for index, value in enumerate(gen_array, start=0):
            if value == 1:
                input_features.append(self.features[index])

        if self.split_training_data:
            dataset = pd.read_csv(self.dataset_csv)
            if self.fixed_splitted_data:
                random_number_dataset = random.randint(
                    1, self.number_of_minor_dataset)
                tmp_dataset = pd.read_csv('data/csv/ga/dataset_{}.csv'.format(str(random_number_dataset)), usecols=input_features + self.target_feature).to_numpy()
                self.X_train, self.y_train, self.X_valid, self.y_valid, self.X_test, self.y_test = split_dataset(tmp_dataset)
                # TO DO HERE
            else:
                pivot = int(self.percentage_split * len(dataset) / 100)
                random_start_point = random.randint(0, len(dataset) - pivot)
                tmp_dataset = dataset.iloc[
                    random_start_point:random_start_point + pivot]
                tmp_dataset.to_csv('data/csv/ga/flex_shuffle_split_data_{}.csv'.format(str(self.tmp)))
                tmp_dataset = pd.read_csv('data/csv/ga/flex_shuffle_split_data_{}.csv'.format(str(self.tmp)), usecols=input_features + self.target_feature).to_numpy()
                self.X_train, self.y_train, self.X_valid, self.y_valid, self.X_test, self.y_test = split_dataset(tmp_dataset)
                # TO DO HERE

        start_time = time.time()
        self.model.fit(self.X_train, self.y_train, eval_set=[(self.X_train, self.y_train), (self.X_valid, self.y_valid)], verbose=0)
        test_results = self.model.predict(self.X_test)
        recall = mean_absolute_error(self.y_test, test_results)
        return recall, np.sum(np.array(time.time() - start_time))

    def fitness(self, gen_array, random_number_dataset):
        input_features = []
        for index, value in enumerate(gen_array, start=0):
            if value == 1:
                input_features.append(self.features[index])

        if self.split_training_data:
            dataset = pd.read_csv(self.dataset_csv)
            if self.fixed_splitted_data:
                tmp_dataset = pd.read_csv('data/csv/ga/dataset_{}.csv'.format(str(random_number_dataset)), usecols=input_features + self.target_feature).to_numpy()
                self.X_train, self.y_train, self.X_valid, self.y_valid, self.X_test, self.y_test = split_dataset(tmp_dataset)
            else:
                pivot = int(self.percentage_split * len(dataset) / 100)
                if self.gen != self.count_gen:
                    random_start_point = random.randint(0, len(dataset) - pivot)
                    tmp_dataset = dataset.iloc[
                        random_start_point:random_start_point + pivot]
                    tmp_dataset.to_csv('data/csv/ga/flex_no_shuffle_split_data_{}.csv'.format(str(self.tmp)))
                    tmp_dataset = pd.read_csv('data/csv/ga/flex_no_shuffle_split_data_{}.csv'.format(str(self.tmp)), usecols=input_features + self.target_feature).to_numpy()
                    self.X_train, self.y_train, self.X_valid, self.y_valid, self.X_test, self.y_test = split_dataset(tmp_dataset)
                    self.count_gen = self.gen
                # TO DO HERE

        start_time = time.time()
        self.model.fit(self.X_train, self.y_train, eval_set=[(self.X_train, self.y_train), (self.X_valid, self.y_valid)], verbose=0)
        test_results = self.model.predict(self.X_test)
        recall = mean_absolute_error(self.y_test, test_results)
        return recall, np.sum(np.array(time.time() - start_time))

    def individual(self, total_feature):
        a = [0 for _ in range(total_feature)]
        for i in range(total_feature):
            r = random.random()
            if r < 0.5:
                a[i] = 1
        indi = {"gen": a, "fitness": 0, "time": 0}
        indi["fitness"], indi["time"] = self.fitness_shuffle_gen(indi["gen"])
        return indi

    def crossover(self, father, mother, total_feature, random_number_dataset):
        cutA = random.randint(1, total_feature - 1)
        cutB = random.randint(1, total_feature - 1)
        while cutB == cutA:
            cutB = random.randint(1, total_feature - 1)
        start = min(cutA, cutB)
        end = max(cutA, cutB)
        child1 = {
            "gen": [0 for _ in range(total_feature)],
            "fitness": 0,
            "time": 0
        }
        child2 = {
            "gen": [0 for _ in range(total_feature)],
            "fitness": 0,
            "time": 0
        }

        child1["gen"][:start] = father["gen"][:start]
        child1["gen"][start:end] = mother["gen"][start:end]
        child1["gen"][end:] = father["gen"][end:]
        if self.shuffle_gen == False:
            child1["fitness"], child1["time"] = self.fitness(
                child1["gen"], random_number_dataset)
        else:
            child1["fitness"], child1["time"] = self.fitness_shuffle_gen(
                child1["gen"])

        child2["gen"][:start] = mother["gen"][:start]
        child2["gen"][start:end] = father["gen"][start:end]
        child2["gen"][end:] = mother["gen"][end:]
        if self.shuffle_gen == False:
            child2["fitness"], child2["time"] = self.fitness(
                child2["gen"], random_number_dataset)
        else:
            child2["fitness"], child2["time"] = self.fitness_shuffle_gen(
                child2["gen"])
        return child1, child2

    def mutation(self, father, total_feature, random_number_dataset):
        a = copy.deepcopy(father["gen"])
        i = random.randint(0, total_feature - 1)
        if a[i] == 0:
            a[i] = 1
        else:
            a[i] = 0
        child = {"gen": a, "fitness": 0, "time": 0}
        if self.shuffle_gen == False:
            child["fitness"], child["time"] = self.fitness(
                child["gen"], random_number_dataset)
        else:
            child["fitness"], child["time"] = self.fitness_shuffle_gen(
                child["gen"])
        return child

    def selection(self, popu, population_size, best_only=True):
        if best_only:
            new_list = sorted(popu, key=itemgetter("fitness"), reverse=True)
            return new_list[:population_size]
        else:
            n = math.floor(population_size / 2)
            temp = sorted(popu, key=itemgetter("fitness"), reverse=True)
            new_list = temp[:n]
            while len(new_list) < population_size:
                i = random.randint(n, len(temp) - 1)
                new_list.append(temp[i])
                temp.remove(temp[i])
            return new_list

    def evolution(self,
                  total_feature,
                  pc=0.8,
                  pm=0.2,
                  population_size=50,
                  max_gen=1000,
                  select_best_only=True):
        ga_log_path = "./log/GA/"
        population = []
        total_time_training = 0
        first_training_time = 0
        start_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        print("Initializing population")
        for _ in range(population_size):
            indi = self.individual(total_feature=total_feature)
            population.append(indi)
            first_training_time += indi["time"]
        new_pop = sorted(population, key=itemgetter("fitness"), reverse=False)
        write_log(path=ga_log_path,
                           filename="fitness_gen.csv",
                           error=[
                               start_time, population[0]["gen"],
                               population[0]["fitness"], first_training_time
                           ])
        print("Done initialization")
        total_time_training += first_training_time
        while self.gen <= max_gen:
            print("Start generation: ", self.gen)
            random_number_dataset = random.randint(
                1, self.number_of_minor_dataset)
            training_time_gen = 0
            temp_population = []
            for i, _ in enumerate(population):
                r = random.random()
                if r < pc:
                    j = random.randint(0, population_size - 1)
                    while j == i:
                        j = random.randint(0, population_size - 1)
                    f_child, m_child = self.crossover(population[i].copy(),
                                                      population[j].copy(),
                                                      total_feature,
                                                      random_number_dataset)
                    temp_population.append(f_child)
                    temp_population.append(m_child)
                    training_time_gen += f_child["time"] + m_child["time"]
                if r < pm:
                    off = self.mutation(population[i].copy(), total_feature,
                                        random_number_dataset)
                    temp_population.append(off)
                    training_time_gen += off["time"]

            # Giu lai x% các cá thể cũ để train lại với bộ dataset mới
            for _ in range(
                    int(self.percentage_back_test / 100 * population_size)):
                random_position = random.randint(0, population_size - 1)
                if self.shuffle_gen == False:
                    population[random_position]['fitness'], population[
                        random_position]['time'] = self.fitness(
                            population[random_position]["gen"],
                            random_number_dataset)
                else:
                    population[random_position]['fitness'], population[
                        random_position]['time'] = self.fitness_shuffle_gen(
                            population[random_position]["gen"])
                training_time_gen += population[random_position]['time']

            population = self.selection(population.copy() + temp_population,
                                        population_size, select_best_only)

            pop_fitness = population[0]["fitness"]
            log_gen = [
                self.gen, population[0]["gen"], pop_fitness, training_time_gen
            ]
            write_log(path=ga_log_path,
                               filename="fitness_gen.csv",
                               error=log_gen)
            print("gen =", self.gen, "fitness =", pop_fitness, "time =",
                  training_time_gen)
            print("Done generation ", self.gen)
            self.gen = self.gen + 1
            total_time_training += training_time_gen

        input_features = []
        for index, value in enumerate(population[0]["gen"], start=0):
            if value == 1:
                input_features.append(self.features[index])
        print(input_features)
        dataset_train = pd.read_csv('data/csv/ga/dataset_train.csv', usecols=input_features + self.target_feature).to_numpy()
        dataset_test = pd.read_csv('data/csv/ga/dataset_test.csv', usecols=input_features + self.target_feature).to_numpy()
        x_test = dataset_test[:, 0:-1]
        y_test = dataset_test[:, -1:]
        X = dataset_train[:, 0:-1]
        Y = dataset_train[:, -1:]
        x_train, x_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.2, random_state=42)
        test_start_time = time.time()
        self.model.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_valid, y_valid)], verbose=0)
        test_results = self.model.predict(x_test)
        recall_test = mean_absolute_error(y_test, test_results)
        write_log(path=ga_log_path,
                           filename="fitness_gen.csv",
                           error=[total_time_training]+[recall_test])
        return pop_fitness, population[0]["gen"], recall_test


def write_log(path, filename, error, input_feature = []):
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    if isinstance(error, list):
        error.insert(0, dt_string)
    if os.path.exists(path) == False:
        os.makedirs(path)
    with open(path + filename, 'a') as file:
        writer = csv.writer(file)
        writer.writerow(error)
        writer.writerow(input_feature)

def split_dataset(dataset, train_per=0.6, valid_per=0.2):
    X = dataset[:, 0:-1]
    Y = dataset[:, -1]
    # split data into train and test sets
    train_size = int(len(dataset)*train_per)
    valid_size = int(len(dataset)*valid_per)
    X_train = X[0:train_size]
    y_train = Y[0:train_size]
    X_valid = X[train_size:train_size+valid_size]
    y_valid = Y[train_size:train_size+valid_size]
    X_test = X[train_size+valid_size:]
    y_test = Y[train_size+valid_size:]

    return X_train, y_train, X_valid, y_valid, X_test, y_test
