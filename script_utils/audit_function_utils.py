import Compiler.types as types
import Compiler.library as library

from Compiler import ml
from Compiler.types import sint, cfix, sfix, MultiArray, MemValue

tf = ml

def compute_median(X_arr):
    """Compute the median of an Array.
    
    This function computes the median of an array by sorting a copy of the provided array and either returns the arithmetic mean of the
    middle elements of the sorted array (for an array of even length), or returns the middle element of the sorted array (for an array of 
    odd length). The function expects that X_arr is an instance of Array() 
    """
    X_copy = types.Array(X_arr.length, X_arr.value_type)
    X_copy.assign(X_arr)
    X_copy.sort()
    the_median = 0
    if X_arr.length % 2 == 0:
        the_median = (X_copy[(X_copy.length-1)//2] + X_copy[X_copy.length//2])/2
    else:
        the_median = X_copy[X_copy.length//2]
    return the_median

def compute_abs_diff_value(X_arr, the_val):
    """This functions computes the absolute difference of each array entry of X_arr and the_val.

    The function expects X_arr to be of type Array and to be not None.
    It also expects the_val to be not None and an instance of a compatible value type of the value type of X_arr 
    """
    X_copy = types.Array(X_arr.length, X_arr.value_type)
    @library.for_range(start=0,stop=X_arr.length,step=1)
    def _(i):
        X_copy[i] = abs(X_arr[i] - the_val)
    return X_copy

# We expect an Array
def MAD_Score(X_array):
    """This function computes the MAD_Score of X_array.
    
    The function expects X_array to be not None and to be an instance of Array.
    The MAD_Score is defined as: MAD_Score(X_arr) = (X_arr - median(X_Arr))/(median_absolute_difference(X_arr))
    """
    score_array = types.Array(X_array.length, X_array.value_type)
    X_median = compute_median(X_array)
    X_abs_diff = compute_abs_diff_value(X_array, X_median)
    X_abs_median = compute_median(X_abs_diff)
    
    @library.for_range(start=0,stop=X_array.length,step=1)
    def _(i):
        score_array[i] = (X_array[i] - X_median)/(X_abs_median)
    return score_array

    
def print_predict_accuracy_model(model, test_samples, test_labels):
    guesses = model.predict(test_samples, batch_size=128)
    library.print_ln('guess %s', guesses.reveal_nested()[:3])
    library.print_ln('truth %s', test_labels.reveal_nested()[:3])

    @library.map_sum_opt(28, 10000, [sint])
    def accuracy(i):
      correct = sint((ml.argmax(guesses[i].reveal()) == ml.argmax(test_labels[i].reveal())))
      return correct

    acc = accuracy().reveal()
    library.print_ln("correct %s %s", acc, acc * cfix(0.0001))


def print_predict_accuracy_opt(opt, batch_size, test_samples, test_labels):
    guesses = opt.eval(test_samples, batch_size)
    library.print_ln('guess %s', guesses.reveal_nested()[:3])
    library.print_ln('truth %s', test_labels.reveal_nested()[:3])

    @library.map_sum_opt(28, test_labels.sizes[0], [sint])
    def accuracy(i):
        correct = sint((ml.argmax(guesses[i].reveal()) == ml.argmax(test_labels[i].reveal())))
        return correct

    acc = accuracy().reveal()
    library.print_ln("correct %s %s", acc, acc * cfix(0.0001))

    return guesses


def print_predict_accuracy_layers(layers, original_model_to_copy_weights_from, batch_size, test_samples, test_labels):
    evaluation_model = tf.keras.models.Sequential(layers)
    optim = tf.keras.optimizers.SGD()
    evaluation_model.compile(optimizer=optim)
    evaluation_model.build(test_samples.sizes, batch_size=batch_size)

    for orig_var, eval_var in zip(original_model_to_copy_weights_from.trainable_variables, evaluation_model.trainable_variables):
        eval_var.address = orig_var.address

    evaluation_graph = evaluation_model.opt
    evaluation_graph.layers[0].X.address = test_samples.address
    evaluation_graph.layers[-1].Y.address = test_labels.address
    return print_predict_accuracy_opt(evaluation_graph, batch_size, test_samples, test_labels)


def get_unlearn_data_for_party(data_owner, train_samples, train_labels, null_label, unlearn_size):
    unlearn_start_region = data_owner * unlearn_size

    modified_training_labels = MultiArray([train_labels.sizes[0], 10], sfix)
    modified_training_labels.assign(train_labels)
    modified_training_labels.get_part(unlearn_start_region, unlearn_size).assign(null_label)

    return train_samples, modified_training_labels


def compute_loss(X, Y):
    n_classes = 10
    n_data_owners = X.sizes[0]
    n_samples = X.sizes[1]
    loss_array = MultiArray([n_samples, n_data_owners], sfix)

    # for sample_id in range(n_samples):
    @library.for_range(start=0, stop=n_samples, step=1)
    def _(sample_id):
        for model_id in range(n_data_owners):
            # sum_holder = MemValue(sfix(0))
            sum_holder = sfix(0)
            # loss_array[sample_id][model_id] = predicted_class * ml.log_e(X[model_id][sample_id][predicted_class])
            for out_class_id in range(n_classes):
                tmp = Y[sample_id][out_class_id] * ml.log_e(X[model_id][sample_id][out_class_id])
                sum_holder += tmp
            # # sum_holder.write(-(sum_holder.read()))
            loss_array[sample_id][model_id] = -sum_holder

    # @library.for_range(start=0, stop=n_samples, step=1)
    # def _(sample_id):
    #     @library.for_range(start=0, stop=n_data_owners, step=1)
    #     def _(model_id):
    #         sum_holder = MemValue(sfix(0))
    #
    #         @library.for_range(start=0, stop=n_classes, step=1)
    #         def _(out_class_id):
    #             tmp = Y[sample_id][out_class_id] * ml.log_e(X[model_id][sample_id][out_class_id])
    #             sum_holder.write(sum_holder.read() + tmp)
    #         # sum_holder.write(-(sum_holder.read()))
    #         loss_array[sample_id][model_id] = -sum_holder.read()

    return loss_array
