from random import randint
import matplotlib.pyplot as plt

def random_training_set_generator(size, min, max, target_function):
    training_set = []
    for i in range (0, size):
        x = randint(min, max)
        y = randint(min, max)
        while(y == target_function(x)):
            y = randint(min, max)
        if(y > target_function(x)):
            training_set.append([x, y, 1])
        else:
            training_set.append([x, y, -1])
    return training_set

def PLA_dot_product(data, weights, threshold):
    result = data[0]*weights[0] + data[1]*weights[1]
    if(result > threshold):
        return 1
    return -1

def PLA_correction(data, weights, threshold, correct_label):
    return sum_vectors(weights, scalar_times_vector(correct_label, data)), (threshold - correct_label)

def sum_vectors(v1, v2):
    v3 = []
    for i in range(len(v1)):
        v3.append(v1[i] + v2[i])
    return v3

def scalar_times_vector(scalar, vector):
    return [vector[0]*scalar, vector[1]*scalar]

def is_misclassified(data, classification):
    return data[2] != classification

def correct_classification(data):
    return data[2]

def PLA(training_set, weights, threshold, num_iterations):
    misclassification_flag = 1
    iteration = 0
    while(iteration < num_iterations and misclassification_flag == 1):
        misclassification_flag = 0
        for i in range(0, len(training_set)):
            classification = PLA_dot_product(training_set[i], weights, threshold)
            print("dot " + str(training_set[i]) + " threshold: " + str(threshold) + " " + "weights: " + str(weights) + "class: " + str(classification))
            if (is_misclassified(training_set[i], classification)):
                misclassification_flag = 1
                weights, threshold = PLA_correction(training_set[i], weights, threshold, correct_classification(training_set[i]))
                print("miss: " + str(training_set[i]) + " threshold: " + str(threshold) + " " + "weights: " + str(weights))
        iteration += 1
    return weights, threshold

def my_plot(data, reference):
    plt.plot([reference[0][0], reference[1][0]], [reference[0][1], reference[1][1]], color='k', linestyle='-', linewidth=1)
    for dot in data:
        plt.plot(dot[0], dot[1], 'ko')
    plt.show()

def my_plot_classification(data, labels, reference):
    plt.plot([reference[0][0], reference[1][0]], [reference[0][1], reference[1][1]], color='k', linestyle='-', linewidth=1)
    for i in range(0, len(data)):
        if (labels[i] == 1):
            plt.plot(data[i][0], data[i][1], 'ro')
        elif (labels[i] == -1):
            plt.plot(data[i][0], data[i][1], 'bo')
    plt.show()

def classify(data_set, weights, threshold):
    labels = []
    for i in range(0, len(data_set)):
        labels.append(PLA_dot_product(data_set[i], weights, threshold))
    return labels

def main():

    #You can alter these variables
    plot_min = 0
    plot_max = 500
    data_set_size = 10
    target_function = lambda x: 250
    weights = [0, 0]
    threshold = 0
    iterations = 100


    training_set = random_training_set_generator(data_set_size, plot_min, plot_max, target_function)
    reference = [[plot_min, target_function(plot_min)], [plot_max, target_function(plot_max)]]
    my_plot(training_set, reference)
    weights, threshold = PLA(training_set, weights, threshold, iterations)
    labels = classify(training_set, weights, threshold)
    my_plot_classification(training_set, labels, reference)

main()
