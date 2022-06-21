import math

def softmax_func(x):
    vector_size = len(x)
    exp_vector = [math.exp(v) for v in x]
    exp_sum = sum(exp_vector)
    result_vector = [v /exp_sum for v in exp_vector]
    return result_vector

true_labels = [
    [0,0,1],
    [1,0,0],
    [0,1,0],
    [1,0,0],
    [1,0,0]
]

prediction_weights = [
    [5,1,1],
    [5,1,1],
    [1,1,5],
    [2,3,2],
    [6,2,2]
]

predictions = [softmax_func(x) for x in prediction_weights]

def cross_entropy_loss(predicted_labes, true_labels):
    n_rows = len(predicted_labes)
    n_features = len(predicted_labes[0])#
    losses = [0] * n_rows
    for i in range(0, n_rows, 1):
        loss_sum = 0
        for j in range(0, n_features, 1):
            result_loss = -true_labels[i][j] * math.log(predicted_labes[i][j])#
            loss_sum = loss_sum + result_loss
        losses[i] = loss_sum
    return losses
print(cross_entropy_loss(predictions, true_labels))