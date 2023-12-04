from Compiler.types import MultiArray, Array, sfix, cfix, sint, cint, regint, MemValue, Matrix
from Compiler.dijkstra import HeapQ
from Compiler.oram import OptimalORAM
from Compiler.permutation import cond_swap

from Compiler import library as lib
from Compiler.library import print_ln
from Compiler import util
from Compiler import types
import operator
from Compiler.script_utils import audit_function_utils as audit_utils
from Compiler.script_utils.timers import TIMER_AUDIT_SHAP_BUILD_COALITIONS, TIMER_AUDIT_SHAP_BUILD_Z_SAMPLES, \
    TIMER_AUDIT_SHAP_EVAL_SAMPLES, TIMER_AUDIT_SHAP_MARGINAL_CONTRIBUTION, TIMER_AUDIT_SHAP_LINREG

from Compiler.script_utils import audit_function_utils as audit_utils

from scipy.stats import norm

import ml
import numpy as np
from scipy.stats import binom
import copy
import itertools


def audit(input_loader, config, debug: bool):

    """
        config.type: "robustness" or "fairness
        config.seed: seed chosen by audit requestor to sample perturbations
        config.R: radius
        config.sigma: to calibrate the perturbation noise
        config.n: number of samples to sample around the audit trigger sample
        config.alpha: 1-alpha is the confidence level

    """


    audit_trigger_samples, audit_trigger_labels = input_loader.audit_trigger()
    n_audit_trigger_samples = audit_trigger_samples.sizes[0]
    assert n_audit_trigger_samples == 1, "Currently only supports one audit trigger sample"
    assert len(audit_trigger_labels) == 1, "Audit trigger samples need to be a list of length one"

    n_dimensions = audit_utils.flatten(audit_trigger_samples).sizes[1]


    if config.type == "robustness":
        cov_matrix = np.zeros((n_dimensions, n_dimensions))
        np.fill_diagonal(cov_matrix, config.sigma)


        tau = norm.cdf(config.radius/float(config.sigma), loc=0, scale=1)

    elif config.type == "fairness":
        Theta = np.zeros((n_dimensions, n_dimensions))
        np.fill_diagonal(Theta, config.theta)
        cov_matrix = np.linalg.inv(Theta)
        tau = norm.cdf(np.sqrt(1.0/config.L), loc=0, scale=1)

    else:
        raise ValueError(f"Type {config.type} not supported")


    print_ln("Creating %s audit trigger sample perturbations...", config.n)
    # sample config.n perturbations to apply to the audit trigger sample
    np.random.seed(config.seed)
    perturbations = np.random.default_rng(seed=config.seed).multivariate_normal(mean=n_dimensions * [0], cov=cov_matrix, size=config.n)
    shape = tuple([config.n]) + tuple(audit_trigger_samples.sizes[1:])
    perturbations = perturbations.reshape(shape)
    perturbations = audit_utils.from_numpy_to_multiarray(perturbations, sfix)

    # applying config.n perturbations to the audit trigger sample
    audit_trigger_sample_perturbations = MultiArray(perturbations.shape, sfix)
    for i in range(config.n):
        audit_trigger_sample_perturbations[i] = audit_trigger_samples[0] + perturbations[i]



    print_ln("Performing inference on %s audit trigger sample perturbations...", config.n)
    model = input_loader.model()
    prediction_results = model.eval(audit_trigger_sample_perturbations, batch_size=config.batch_size)


    # Count how many times the audit_trigger_label matches the prediction obtained on the permutations
    if  isinstance(audit_trigger_labels, Array):
        # binary classification
        print_ln("Processing inference predictions of Binary Classification...")
        correct_counts = [((prediction_results[i] >= 0.5) == audit_trigger_labels[0]).if_else(1, 0) for i in range(config.n)]

    elif isinstance(audit_trigger_labels, MultiArray):
        # multi-class classification
        print_ln("Processing inference predictions of Multi Class Classification...")

        audit_trigger_label_argmax = ml.argmax(audit_trigger_labels[0])
        correct_counts = [(audit_trigger_label_argmax == ml.argmax(prediction_results[i])).if_else(1, 0) for i in range(config.n)]
    else:
        raise ValueError(f"Audit trigger labels need to be either a list or a MultiArray   ({type(audit_trigger_labels)})")

    n_correct_count = lib.tree_reduce(operator.add, correct_counts)


    # finally, perform the binomial test to check if the model is robust/fair
    print_ln("Performing Binomial Test...")
    threshold = binary_search_min_n_correct_threshold(n_samples=config.n, tau=tau, alpha=config.alpha)
    print_ln("   with n_correct threshold %s/%s", threshold, config.n)

    # the bin-pv test is passed if the number of correct predictions is greater than the threshold
    is_certified = (n_correct_count >= threshold).if_else(True, False)

    result = {"is_certified": is_certified}

    # these are normally not released
    if config.debug:
        result["n_correct_count"] = n_correct_count
        #result["correct_counts"] = correct_counts
        #result["prediction_results"] = prediction_results

    return result




def binary_search_min_n_correct_threshold(n_samples, tau, alpha):

    def check_pv_test(n_correct):
        pv =  1 - binom.cdf(n_correct, n_samples, tau)
        return pv <= alpha

    # binary search

    left, right = 0, n_samples
    result = None

    while left <= right:
        mid = left + (right - left) // 2

        if check_pv_test(mid):
            result = mid
            right = mid - 1
        else:
            left = mid + 1

    if result is None:
        raise ValueError("No value satisfies the test")

    return result