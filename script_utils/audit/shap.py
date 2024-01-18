
from Compiler.types import MultiArray, Array, sfix, cfix, sint, cint, regint, MemValue, Matrix
from Compiler.dijkstra import HeapQ
from Compiler.oram import OptimalORAM
from Compiler.permutation import cond_swap

from Compiler import library as lib
from Compiler.library import print_ln
from Compiler import util
from Compiler import types

from Compiler.script_utils import audit_function_utils as audit_utils
from Compiler.script_utils.timers import TIMER_AUDIT_SHAP_BUILD_COALITIONS, TIMER_AUDIT_SHAP_BUILD_Z_SAMPLES, \
    TIMER_AUDIT_SHAP_EVAL_SAMPLES, TIMER_AUDIT_SHAP_MARGINAL_CONTRIBUTION, TIMER_AUDIT_SHAP_LINREG

import ml
import numpy as np
import copy
from scipy.special import binom
import itertools


def audit(input_loader, config, debug: bool):

    # step 1: create perturbed sample set, replace with elements from train set
    # step 2: run through model to get labels
    # step 3: train a lin reg

    # Sample coalitions
    audit_trigger_samples, _audit_trigger_mislabels = input_loader.audit_trigger()
    n_audit_trigger_samples = audit_trigger_samples.sizes[0]

    num_features = audit_trigger_samples.sizes[1]
    num_samples = 2 * 5

    if num_samples < num_features:
        print("WARNING: Number of samples is lower than number of features to explain, the explanations may not"
              "work correctly as the linear system is undetermined")

    train_samples, _train_labels = input_loader.train_dataset()
    if config.n_batches > 0:
        train_samples = train_samples.get_part(0, config.n_batches * config.batch_size)
        _train_labels = _train_labels.get_part(0, config.n_batches * config.batch_size)
        print("Using only first %s batches of training data" % config.n_batches)

    n_train_samples = len(train_samples)

    # @lib.for_range_opt(n_audit_trigger_samples)
    @lib.for_range_opt(1)
    def shap_loop(audit_trigger_idx):
        print_ln("  audit_trigger_idx=%s", audit_trigger_idx)

        audit_trigger_sample = audit_trigger_samples[audit_trigger_idx]
        audit_trigger_label = _audit_trigger_mislabels[audit_trigger_idx]

        # TODO: Can be completely parallel?
        # Optimization 1: clear int
        # Actually this may not need to be a MultiArray but can be bitwise operations of regint?
        # z_coalitions = MultiArray([num_samples, num_features], regint)
        lib.start_timer(TIMER_AUDIT_SHAP_BUILD_COALITIONS)
        z_coalitions, kernelWeights = build_subsets_order(num_samples, num_features)
        # print("z_coalitions", z_coalitions)

        # construct array num_samples * n_train_samples size
        # its so big because we take the marginal distribution
        z_samples = Matrix(num_samples * n_train_samples, num_features, audit_trigger_sample.value_type)
        # z_samples = np.zeros((num_samples * n_train_samples, num_features), dtype=np.int8)
        # print(z_samples)

        # regint style
        z_coalitions_runtime = Matrix(num_samples, num_features, regint)
        z_coalitions_list = z_coalitions.tolist()
        for i in range(len(z_coalitions_list)):
            z_coalitions_runtime[i].assign(z_coalitions_list[i])

        z_coalitions_kernelWeights_runtime = Matrix(num_samples, num_features, cfix)
        z_coalitions_kernelWeights_list = (np.expand_dims(kernelWeights, 1) * z_coalitions).tolist()
        for i in range(len(z_coalitions_list)):
            z_coalitions_kernelWeights_runtime[i].assign(z_coalitions_kernelWeights_list[i])
        lib.stop_timer(TIMER_AUDIT_SHAP_BUILD_COALITIONS)

        # kernelWeights_list = kernelWeights_runtime.tolist()
        # for i in range(len(kernelWeights_list)):
        #     kernelWeights_runtime[i].assign(kernelWeights_list[i])

        lib.start_timer(TIMER_AUDIT_SHAP_BUILD_Z_SAMPLES)
        @lib.for_range_opt(num_samples)
        def coalitions(z_i):
            @lib.for_range_opt(n_train_samples)
            def coalition_train_sample(train_idx):
                # TODO: Vectorize this when elementwise operations become available... can we update the whole column at once?
                @lib.for_range_opt(num_features)
                def ran(f_i):
                    z_samples[(z_i * n_train_samples) + train_idx][f_i] = audit_trigger_sample[f_i] * z_coalitions_runtime[z_i][f_i] \
                                                                          + (1 - z_coalitions_runtime[z_i][f_i]) * train_samples[train_idx][f_i]
        lib.stop_timer(TIMER_AUDIT_SHAP_BUILD_Z_SAMPLES)
        # compile-time
        # for z_i in range(num_samples):
        #     for train_idx in range(n_train_samples):
        #         # TODO: Vectorize this when elementwise operations become available... can we update the whole column at once?
        #         for f_i in range(num_features):
        #             # print(train_idx, f_i, z_coalitions.shape, z_i, z_coalitions[z_i][f_i], 1 - z_coalitions[z_i][f_i])
        #             # z_samples[(z_i * n_train_samples) + train_idx][f_i] = 0
        #             # print(audit_trigger_sample[f_i] * regint(int(z_coalitions[z_i][f_i])))
        #             # print((1 - z_coalitions[z_i][f_i]) * train_samples[train_idx][f_i])
        #             z_samples[(z_i * n_train_samples) + train_idx][f_i] = audit_trigger_sample[f_i] * convert_np_regint(z_coalitions[z_i][f_i]) \
        #                                                                   + (1 - convert_np_regint(z_coalitions[z_i][f_i])) * train_samples[train_idx][f_i]

                    # z_samples[(z_i * n_train_samples) + train_idx][f_i] = (audit_trigger_sample[f_i] == 1).if_else(z_coalitions[z_i][f_i], z_samples[(z_i * n_train_samples) + train_idx][f_i])
                                                                          # + (1 - z_coalitions[z_i][f_i]) * train_samples[train_idx][f_i]

            # print_ln("%s: %s", z_coalitions[z_i], z_samples[z_i].reveal())
        print("Done generating z_samples", z_samples.sizes)

        lib.start_timer(TIMER_AUDIT_SHAP_EVAL_SAMPLES)

        model = input_loader.model()
        prediction_results = model.eval(z_samples, batch_size=config.batch_size)
        prediction_results_ex = Array(num_samples, sfix)
        prediction_results_ex.assign_all(sfix(0))

        lib.stop_timer(TIMER_AUDIT_SHAP_EVAL_SAMPLES)
        lib.start_timer(TIMER_AUDIT_SHAP_MARGINAL_CONTRIBUTION)

        # integrate predictinos over marginal distribution
        @lib.for_range_opt(num_samples)
        def coalitions_marginal(z_i):
            @lib.for_range_opt(n_train_samples)
            def summer(i):
                prediction_results_ex[z_i] = prediction_results_ex[z_i] + prediction_results[(z_i * n_train_samples) + i]

            prediction_results_ex[z_i] = prediction_results_ex[z_i] * cfix(1. / n_train_samples)

        if debug:
            print_ln("Average prediction without feature: %s", prediction_results_ex.reveal())
        lib.stop_timer(TIMER_AUDIT_SHAP_MARGINAL_CONTRIBUTION)
        lib.start_timer(TIMER_AUDIT_SHAP_LINREG)

        # Now we train a lin reg on the samples

        # these can be constants
        invert_res = invert_compile_time(z_coalitions, kernelWeights)
        print("invert_res", invert_res.shape)

        # now we should compute X^T y
        runtime_xtxinv = Matrix(invert_res.shape[0], invert_res.shape[1], cfix)
        print("Compile-time", invert_res[0][0], invert_res[0][1])

        for i in range(num_samples):
            runtime_xtxinv[i].assign(invert_res[i].tolist())

        print("Compile-time", invert_res[0][0], invert_res[0][1])
        if debug:
            print_ln("Runtime %s %s", runtime_xtxinv[0][0], runtime_xtxinv[0][1])

        # Now finally some private-public multiplications are happening
        print("prediction_results_ex", prediction_results_ex)
        print("z_coalitions_runtime", z_coalitions_runtime)

        secret_xty = z_coalitions_kernelWeights_runtime.transpose().dot(prediction_results_ex)
        print("secret_xty", secret_xty)
        print("runtime_xtxinv", runtime_xtxinv)
        if debug:
            print_ln("runtime_xtxinv %s", runtime_xtxinv)
            print_ln("secret_xty %s", secret_xty.reveal_nested())
        secret_params = runtime_xtxinv.dot(secret_xty)
        if debug:
            print_ln("SECRET PARAMS %s", secret_params.reveal_nested())
            print_ln("SUM %s", sum(secret_params.reveal_nested()))
        lib.stop_timer(TIMER_AUDIT_SHAP_LINREG)

        # print_ln("SUM %s", )

    return {}, {}


            # TODO: Compute shapley distance?

        # TODO: Random subset weight vector calculation?


        # TODO: Compute distances?


        # for each coalition, compute a corresponding sample
        # @lib.for_range_opt(num_samples)
        # def coalitions(z_i):


def convert_np_regint(input):
    return regint(int(input))

def audit_shap(input_loader, config, debug: bool):
    # Load Training Dataset
    train_samples, _train_labels = input_loader.train_dataset()
    n_train_samples = len(train_samples)

    # _train_labels should be an array of integers for our sorting? not one_hot encoding!!
    # Idea for conversion?
    # TODO: Fix this for adult
    # TODO: Speed?
    lib.start_timer(104)
    _train_labels_idx = Array(len(train_samples), sint)
    @lib.for_range(len(train_samples))
    def _(i):
        _train_labels_idx[i] = ml.argmax(_train_labels[i])

    lib.stop_timer(104)
    # for each train samples -> build ownership array
    train_samples_ownership = Array(len(train_samples), sint)

    for party_id in range(input_loader.num_parties()):
        start, size = input_loader.train_dataset_region(party_id)
        train_samples_ownership.get_sub(start=start, stop=start+size).assign_all(party_id)

    # create array [0, 1, 2, 3, ... ]
    train_samples_idx = Array.create_from(cint(x) for x in range(len(train_samples)))

    # Load Audit Trigger Samples
    audit_trigger_samples, _audit_trigger_mislabels = input_loader.audit_trigger()


    #first = audit_trigger_samples.get_part(0, 1)
    #print_ln("first=%s", first.reveal_nested())
    #return {"red": None}

    model = input_loader.model()
    latent_space_layer, expected_latent_space_size = input_loader.model_latent_space_layer()
    latent_space_layer=None

    lib.start_timer(105)
    print_ln("Computing Latent Space for Training Set...")
    # Model.eval seems to take a really long time
    train_samples_latent_space = model.eval(train_samples, batch_size=config.batch_size, latent_space_layer=latent_space_layer)
    # assert train_samples_latent_space.sizes == (len(train_samples), expected_latent_space_size)

    print_ln("Computing Latent Space for Audit Trigger...")
    audit_trigger_samples_latent_space = model.eval(audit_trigger_samples, batch_size=config.batch_size, latent_space_layer=latent_space_layer)
    # assert  audit_trigger_samples_latent_space.sizes == (len(audit_trigger_samples), expected_latent_space_size)

    train_samples_latent_space = MultiArray([len(train_samples), expected_latent_space_size], sfix)
    train_samples_latent_space.assign_all(sfix(1))
    audit_trigger_samples_latent_space = MultiArray([len(audit_trigger_samples), expected_latent_space_size], sfix)
    audit_trigger_samples_latent_space.assign_all(sfix(1))

    lib.stop_timer(105)

    print(audit_trigger_samples_latent_space.sizes, train_samples_latent_space.sizes)
    print_ln("Computing L2 distance...")
    L2 = audit_utils.euclidean_dist_dot_product(A=train_samples_latent_space, B=audit_trigger_samples_latent_space, n_threads=config.n_threads)
    L2 = L2.transpose()
    assert L2.sizes == (len(audit_trigger_samples), len(train_samples)), f"L2 {L2.sizes}"


    n_audit_trigger_samples = len(audit_trigger_samples)

    if debug:
        knn_sample_id = MultiArray([n_audit_trigger_samples, config.K], sint)
    else:
        knn_sample_id = None


    knn_shapley_values = MultiArray([n_audit_trigger_samples, n_train_samples], sfix)
    knn_shapley_values.assign_all(-1)

    print_ln("Running knnshapley...")

    # precompute min(i, K) / i at compile time
    complex_division_at_compile_time = Array(n_train_samples, cfix)
    for i in range(1, n_train_samples):
        complex_division_at_compile_time[i] = min(config.K, i) / (float(i) * config.K)

    @lib.for_range_opt(n_audit_trigger_samples)
    def knn(audit_trigger_idx):
        print_ln("  audit_trigger_idx=%s", audit_trigger_idx)

        audit_trigger_label = _audit_trigger_mislabels[audit_trigger_idx]
        audit_trigger_label_idx = ml.argmax(audit_trigger_label)

        print(L2)
        print("L2")
        dist_arr = L2[audit_trigger_idx]

        print(_train_labels, train_samples_idx, dist_arr)
        # data = concatenate([dist_arr, train_samples_idx, _train_labels], axis=1)

        data = concatenate([dist_arr, train_samples_idx, _train_labels_idx], axis=1)
        # top k (min distances)
        # Note: I tried implementing top k with oblivious Heap (see dijkstra.py)
        #       However, I think there is a bug that requires setting max_size==n_train_samples instead of K
        #       and then performance is very bad
        lib.start_timer(timer_id=101)
        if debug:
            print_ln("Before sort: %s", data.get_part(0, 10).reveal_nested())
        data.sort()
        # pre_sort_sort(data, input_loader)
        # knn_naive(dist_arr, train_samples_idx, config.K)
        # if debug:
        #     print_ln("After sort: %s", data.get_part(0, 10).reveal_nested())
        #
        #     print_ln("Target sample %s", audit_trigger_samples[audit_trigger_idx].reveal_nested())
        #     closest_sample_idx = sint(data.get_part(0, 10)[0][1]).reveal()
        #     print_ln("cloeset sample idx %s", closest_sample_idx)
        #     print("compile sample", closest_sample_idx)
        #     closest_sample = train_samples[closest_sample_idx]
        #     print_ln("Closest sample %s %s", closest_sample_idx, closest_sample.reveal_nested() )

        lib.stop_timer(timer_id=101)

        assert data.sizes == (n_train_samples, 3), f"top k {data.sizes}"

        # TODO: Optimize comparison ?
        print(audit_trigger_idx, n_train_samples, knn_shapley_values.sizes, data.sizes)
        # TODO: uses idx instead of array
        knn_shapley_values[audit_trigger_idx][n_train_samples - 1] = \
            sfix(data[n_train_samples - 1][2] == audit_trigger_label_idx) / n_audit_trigger_samples

        lib.start_timer(timer_id=102)
        lib.start_timer(timer_id=103)

        precomputed_equality_array = Array(n_train_samples, sfix)
        @lib.for_range_opt(n_train_samples)
        def _(i):
            # PRECOMP
            precomputed_equality_array[i] = sfix(data[i][2] == audit_trigger_label_idx)
        lib.stop_timer(timer_id=103)

        @lib.for_range_opt(n_train_samples - 1, 1, -1)
        def _(iplusone):
            i = iplusone - 1
            # TODO: Optimize comparison?
            complex_part_one = precomputed_equality_array[i] - precomputed_equality_array[iplusone]
            compile_time_part_two = complex_division_at_compile_time[iplusone] # I think this is correct because its shifting

            knn_shapley_values[audit_trigger_idx][i] = knn_shapley_values[audit_trigger_idx][iplusone] + (complex_part_one * compile_time_part_two)

        lib.stop_timer(timer_id=102)

    # @lib.for_range(config.K)
        # def _(k):
        #
        #     # aggregate per party
        #     @lib.for_range(input_loader.num_parties())
        #     def _(party_id):
        #
        #         # top_k sample (based on distance) belongs to this party
        #         sel_party_id = data[k][2]
        #
        #         # update==0 if it does not match party_id
        #         # update==1 if it matches party_id
        #         comp = party_id == sel_party_id
        #         update = util.cond_swap(comp, 0, 1)[0]
        #         # Note: Conditional Swap becomes necessary because you cannot use the sel_party_id to lookup an array
        #
        #         # print_ln("trigger=%s   party_id=%s  sel_party_id=%s  update=%s", audit_trigger_idx, party_id, sel_party_id.reveal(), update.reveal())
        #
        #         # update the count +0 / +1
        #         knn_party_count[audit_trigger_idx][party_id] += update
        #
        #     # sample ids
        #     if debug:
        #         sel_sample_id = data[k][1]
        #         knn_sample_id[audit_trigger_idx][k] = sel_sample_id

    # assert knn_shapley_values.sizes == (n_audit_trigger_samples, n_train_samples), f"knn_party_count={knn_shapley_values.sizes}"

    result = {"shapley_values": knn_shapley_values}
    debug_output = {}
    # result = {"shapley_values": []}


    if debug:
    #     # result["knn_party_count"] = knn_party_count
    #     # result["knn_sample_id"] = knn_sample_id
    #     # result["mad_score_matrix"] = mad_score_matrix
    #     debug_output["l2_distances"] = L2.get_part(0, 3)
        pass

    return result, debug_output

def build_subsets_order(num_samples, num_features):
    # z_coalitions = MultiArray([num_samples, num_features], regint)
    z_coalitions = np.zeros((num_samples, num_features), dtype=np.int64)

    # These are constant and based on compile-time data such as the number of features
    num_subset_sizes = int(np.ceil((num_features - 1) / 2.0))
    num_paired_subset_sizes = int(np.floor((num_features - 1) / 2.0))

    # Also public compile-time constants
    weight_vector = np.array([(num_features - 1.0) / (i * (num_features - i)) for i in range(1, num_subset_sizes + 1)])
    # weight_vector = Array(num_subset_sizes, cfix)
    # @for_range_opt(num_subset_sizes):
    # def _(i):
    #     weight_vector[i] = (num_features - 1.0) / ((i + 1) * (num_features - i + 1))

    weight_vector[:num_paired_subset_sizes] *= 2
    weight_vector /= np.sum(weight_vector)

    # print(f"weight_vector = {weight_vector}")
    print(f"num_subset_sizes = {num_subset_sizes}")
    print(f"num_paired_subset_sizes = {num_paired_subset_sizes}")
    print(f"num_features = {num_features}")


    # fill out all the subset sizes we can completely enumerate
    # given nsamples*remaining_weight_vector[subset_size]
    # starts by doing all |z|-1, then all |z|-2 until |z|-(|z| / 2)
    num_full_subsets = 0
    num_samples_left = num_samples

    group_inds = np.arange(num_features, dtype='int64')
    mask = np.zeros(num_features, dtype=np.int64)
    remaining_weight_vector = copy.copy(weight_vector)

    n_samples_added = 0
    kernelWeights = np.zeros(num_samples)

    for subset_size in range(1, num_subset_sizes + 1):

        # determine how many subsets (and their complements) are of the current size
        nsubsets = binom(num_features, subset_size)
        if subset_size <= num_paired_subset_sizes:
            nsubsets *= 2
        print(f"subset_size = {subset_size}")
        print(f"nsubsets = {nsubsets}")
        print("self.nsamples*weight_vector[subset_size-1] = {}".format(
            num_samples_left * remaining_weight_vector[subset_size - 1]))
        print("self.nsamples*weight_vector[subset_size-1]/nsubsets = {}".format(
            num_samples_left * remaining_weight_vector[subset_size - 1] / nsubsets))

        # see if we have enough samples to enumerate all subsets of this size
        if num_samples_left * remaining_weight_vector[subset_size - 1] / nsubsets >= 1.0 - 1e-8:
            num_full_subsets += 1
            num_samples_left -= nsubsets

            # rescale what's left of the remaining weight vector to sum to 1
            if remaining_weight_vector[subset_size - 1] < 1.0:
                remaining_weight_vector /= (1 - remaining_weight_vector[subset_size - 1])

            # add all the samples of the current subset size
            w = weight_vector[subset_size - 1] / binom(num_features, subset_size)
            if subset_size <= num_paired_subset_sizes:
                w /= 2.0
            for inds in itertools.combinations(group_inds, subset_size):
                mask[:] = 0
                mask[np.array(inds, dtype='int64')] = 1
                # z_coalitions[n_samples_added].assign_vector(mask.tolist())
                z_coalitions[n_samples_added, :] = mask
                kernelWeights[n_samples_added] = w
                n_samples_added += 1
                if subset_size <= num_paired_subset_sizes:
                    # TODO: Not sure what this does but i guess its an optimization?
                    mask[:] = np.abs(mask - 1)
                    # z_coalitions[n_samples_added].assign_vector(mask.tolist())
                    z_coalitions[n_samples_added, :] = mask
                    kernelWeights[n_samples_added] = w
                    n_samples_added += 1
        else:
            break
    print(f"num_full_subsets = {num_full_subsets}")

    # add random samples from what is left of the subset space
    nfixed_samples = n_samples_added
    samples_left = num_samples - n_samples_added
    print(f"samples_left = {samples_left}")
    if num_full_subsets != num_subset_sizes:
        remaining_weight_vector = copy.copy(weight_vector)
        remaining_weight_vector[:num_paired_subset_sizes] /= 2 # because we draw two samples each below
        remaining_weight_vector = remaining_weight_vector[num_full_subsets:]
        remaining_weight_vector /= np.sum(remaining_weight_vector)
        print(f"remaining_weight_vector = {remaining_weight_vector}")
        print(f"num_paired_subset_sizes = {num_paired_subset_sizes}")
        np.random.seed(42)
        ind_set = np.random.choice(len(remaining_weight_vector), 4 * samples_left, p=remaining_weight_vector)
        ind_set_pos = 0
        used_masks = {}
        while samples_left > 0 and ind_set_pos < len(ind_set):
            mask.fill(0)
            ind = ind_set[ind_set_pos] # we call np.random.choice once to save time and then just read it here
            ind_set_pos += 1
            subset_size = ind + num_full_subsets + 1
            mask[np.random.permutation(num_features)[:subset_size]] = 1

            # only add the sample if we have not seen it before, otherwise just
            # increment a previous sample's weight
            mask_tuple = tuple(mask)
            new_sample = False
            if mask_tuple not in used_masks:
                new_sample = True
                used_masks[mask_tuple] = n_samples_added
                samples_left -= 1
                # z_coalitions[n_samples_added].assign_vector(mask.tolist())
                z_coalitions[n_samples_added, :] = mask
                kernelWeights[n_samples_added] = 1.0
                n_samples_added += 1
            else:
                kernelWeights[used_masks[mask_tuple]] += 1.0

            # add the compliment sample
            if samples_left > 0 and subset_size <= num_paired_subset_sizes:
                mask[:] = np.abs(mask - 1)

                # only add the sample if we have not seen it before, otherwise just
                # increment a previous sample's weight
                if new_sample:
                    samples_left -= 1
                    # z_coalitions[n_samples_added].assign_vector(mask.tolist())
                    z_coalitions[n_samples_added, :] = mask
                    kernelWeights[n_samples_added] = 1.0
                    n_samples_added += 1
                else:
                    # we know the compliment sample is the next one after the original sample, so + 1
                    kernelWeights[used_masks[mask_tuple] + 1] += 1.0

        # normalize the kernel weights for the random samples to equal the weight left after
        # the fixed enumerated samples have been already counted
        weight_left = np.sum(weight_vector[num_full_subsets:])
        print(f"weight_left = {weight_left}")
        print(f"kernelWeights {kernelWeights}")
        kernelWeights[nfixed_samples:] *= weight_left / kernelWeights[nfixed_samples:].sum()

        return z_coalitions, kernelWeights

def invert_compile_time(z_coalitions, kernelWeights):
    # eyAdj2 = eyAdj - self.maskMatrix[:, nonzero_inds[-1]] * (
    #     self.link.f(self.fx[dim]) - self.link.f(self.fnull[dim]))
    # (X^T - X)^T
    # etmp = np.transpose(np.transpose(z_coalitions) - z_coalitions)
    etmp = z_coalitions
    print(f"etmp[:4,:] {etmp[:4, :]}")
    print(f"kernelWeights {kernelWeights}")

    # solve a weighted least squares equation to estimate phi
    # wX^T X
    tmp = np.transpose(np.transpose(etmp) * np.transpose(kernelWeights))
    print(f"tmp {tmp}")
    etmp_dot = np.dot(np.transpose(tmp), etmp)
    print(f"etmp_dot {etmp_dot}")
    try:
        tmp2 = np.linalg.inv(etmp_dot)
    except np.linalg.LinAlgError:
        tmp2 = np.linalg.pinv(etmp_dot)
    # (wX^T X)-1
    return tmp2

