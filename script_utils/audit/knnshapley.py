
from Compiler.types import MultiArray, Array, sfix, cfix, sint, cint, regint, MemValue
from Compiler.dijkstra import HeapQ
from Compiler.oram import OptimalORAM
from Compiler.permutation import cond_swap

from Compiler import library as lib
from Compiler.library import print_ln
from Compiler import util
from Compiler import types

from Compiler.script_utils import audit_function_utils as audit_utils

import ml


# TODO [hly]: Something is still buggy here. The l2 distance squared is checked, but the problem may be in the forward pass to get the latent space or then later after computing the distances

# TODO [hly]: Check the optimizations that we apply here... are some of them

def audit(input_loader, config, debug: bool):

    # Load Training Dataset
    train_samples, _train_labels = input_loader.train_dataset()

    if config.n_batches > 0:
        print("Approximating with cfg.n_batches")
        train_samples = train_samples.get_part(0, config.n_batches * config.batch_size)
        _train_labels = _train_labels.get_part(0, config.n_batches * config.batch_size)
        print("Running on", len(train_samples), "samples")

    n_train_samples = len(train_samples)

    # _train_labels should be an array of integers for our sorting? not one_hot encoding!!
    # Idea for conversion?
    # TODO: Fix this for adult
    # TODO: Speed?
    lib.start_timer(104)
    _train_labels_idx = Array(len(train_samples), sint)
    @lib.for_range_opt_multithread(config.n_threads, len(train_samples))
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

    lib.start_timer(105)
    print_ln("Computing Latent Space for Training Set...")

    # Model.eval seems to take a really long time
    model.layers[-1].compute_loss = False

    if config.batch_size == 2 and config.n_batches == 1:
        # dont do the forward pass so we can properly extrapolate
        print("Skipping second forward pass")
        train_samples_latent_space = MultiArray([len(train_samples), expected_latent_space_size], sfix)
    else:
        train_samples_latent_space = model.eval(train_samples, batch_size=config.batch_size, latent_space_layer=latent_space_layer)
        assert train_samples_latent_space.sizes == (len(train_samples), expected_latent_space_size), f"{train_samples_latent_space.sizes} != {(len(train_samples), expected_latent_space_size)}"

    print_ln("Computing Latent Space for Audit Trigger...")
    audit_trigger_samples_latent_space = model.eval(audit_trigger_samples, batch_size=config.batch_size, latent_space_layer=latent_space_layer)
    assert  audit_trigger_samples_latent_space.sizes == (len(audit_trigger_samples), expected_latent_space_size), f"{audit_trigger_samples_latent_space.sizes} != {(len(audit_trigger_samples), expected_latent_space_size)}"

    lib.stop_timer(105)

    print(audit_trigger_samples_latent_space.sizes, train_samples_latent_space.total_size())
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
        print("DATA SHAPE", data.sizes)
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

def knn_naive(distance_array, idx_array, k):
    # O(nk * 3) swaps
    n = len(distance_array)
    k_nn = Array(k, sint)
    k_nn.assign(sint(10000))

    print(k, idx_array.value_type)

    # this seems reaaaally slwo ... how? are we doing something wrong with the bit composition?


    @lib.for_range(n)
    def _(i):
        val = distance_array[i]
        val_idx = idx_array[i]
        # use memvalue if we opt this
        val_was_placed = MemValue(sint(0))
        @lib.for_range(k)
        def _(j):
            val_is_lower = k_nn[j] > val # potentially reduce number of bits to compare here?
            print(val_is_lower, val_was_placed, j)
            val_should_be_placed = val_is_lower.bit_and(val_was_placed == 0)

            place_val = val_should_be_placed * val_idx + ((sint(1) - val_should_be_placed) * k_nn[j])
            print(place_val, val_should_be_placed, val, k_nn.reveal_nested())
            k_nn[j] = place_val

            # val_was_placed = val_was_placed or val_should_be_placed
            val_was_placed.write(val_was_placed.read().bit_or(val_should_be_placed))

    print_ln("k_nn %s", k_nn.reveal_nested())
    return k_nn


def pre_sort_sort(array, input_loader):
    # expects:         data = concatenate([dist_arr, train_samples_idx, _train_labels], axis=1)
    # sorted by party order
    # open data to each party based on train_samples_ownership
    for party_id in range(input_loader.num_parties()):
        start, size = input_loader.train_dataset_region(party_id)
        print(array, start, size)
        distance_party = array.get_part(start=start, size=start+size).reveal_nested() # distance_party is cfix
        print(len(distance_party), len(distance_party[0]))
        # sorted = bubble_sort(distance_party)
        sortedx = lib.sort([x[0] for x in distance_party])
        print(sortedx)

    # TODO: Make these values personal and verify the operations are not verified ?
    # TODO: verify each party sorted


# def bubble_sort(arr):
#     n = len(arr)
#
#     for i in range(n - 1):
#         for j in range(0, n - i - 1):
#             # Compare the first element of each sublist
#             # permutation.cond_swap(arr[j], arr[j + 1]
#             def cond(x,y):
#                 return x[0] > y[0]
#             cond_swap(arr[j], arr[j + 1], cond)
#             # if arr[j][0] > arr[j + 1][0]:
#             #     arr[j], arr[j + 1] = arr[j + 1], arr[j]


def concatenate(arrays, axis):
    """

    https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html

    Args:
        inputs [a1, a2, ...]: The arrays must have the same shape.
        axis (int): The axis along which the arrays will be joined.
    """

    if axis != 1:
        raise ValueError("not implemented yet")

    n_rows = arrays[0].length
    n_cols = len(arrays)
    out = MultiArray([n_rows, n_cols], arrays[0].value_type)

    for a in arrays:
        assert a.length == arrays[0].length
        #assert a.value_type == arrays[0].value_type, f"{a.value_type}   {arrays[0].value_type}"

    # TODO: Change to assign_vector ?
    @lib.for_range(n_rows)
    def _(i):
        for j, a in enumerate(arrays):
            out[i][j] = a[i]

    return out