
from Compiler.types import MultiArray, Array, Matrix, sfix, cfix, sint, cint, regint, MemValue

from Compiler import library as lib
from Compiler.library import print_ln

import ml


def cosine_distance(A: MultiArray, B: MultiArray, n_threads):
    """
    Computes the square of the euclidean distance

    Following: https://nenadmarkus.com/p/all-pairs-euclidean/

    Args:
        A (MultiArray): P x M
        B (MultiArray): R x M

    Returns:
        Pairwise euclidean distances between vectors in A and vectors in B
        (MultiArray) P x R
    """
    print_ln("cosine_distance")

    L2, _, _ = euclidean_distance_naive(A, B, n_threads)
    # aTa = Array(len(A), A.value_type)
    # bTb = Array(len(B), B.value_type)
    aTa = compute_magnitudes(A, n_threads)
    bTb = compute_magnitudes(B, n_threads)
    #


    # print("aTa", aTa)
    # print("bTb", bTb)
    print("L2", L2)

    lib.start_timer(109)
    @lib.for_range_opt_multithread(n_threads, len(A))
    def f(i):
        L2[i] = L2[i] / (aTa[i] * bTb)

    lib.stop_timer(109)

    return L2


def euclidean_distance(A: MultiArray, B: MultiArray, n_threads):
    """
    Computes the square of the euclidean distance

    Following: https://nenadmarkus.com/p/all-pairs-euclidean/

    Args:
        A (MultiArray): P x M
        B (MultiArray): R x M

    Returns:
        Pairwise euclidean distances between vectors in A and vectors in B
        (MultiArray) P x R
    """

    aTa = Array(len(A), A.value_type)
    @lib.for_range_multithread(n_threads, 1, len(A))
    def f(i):
        aTa[i] = A.value_type.dot_product(A[i], A[i])

    print(f"aTa={aTa.length}")

    # bTb = B.dot(B.transpose())
    bTb = Array(len(B) , B.value_type)
    @lib.for_range_multithread(n_threads, 1, len(B))
    def f(i):
        bTb[i] = B.value_type.dot_product(B[i], B[i])

    print_ln("  bTb done")

    print(f"bTb={bTb.length}")

    L2 = A.dot(B.transpose())
    print(f"L2={L2.sizes}")

    print_ln("  AB done")

    @lib.for_range_opt_multithread(n_threads, len(A))
    def f(i):
        L2[i] = L2[i] * -2 + bTb + aTa[i]

    return L2, aTa, bTb

def euclidean_distance_naive(A: MultiArray, B: MultiArray, n_threads, skip_reduce=True):
    """
    Computes the square of the euclidean distance

    Following: https://nenadmarkus.com/p/all-pairs-euclidean/

    Args:
        A (MultiArray): P x M
        B (MultiArray): R x M

    Returns:
        Pairwise euclidean distances between vectors in A and vectors in B
        (MultiArray) P x R
    """

    lib.start_timer(110)
    L2 = MultiArray([len(B), len(A)], A.value_type if not skip_reduce else sint)
    # @lib.for_range_opt(len(B))
    # def f(j):
    for j in range(len(B)):
        B_mat = Matrix(len(B[j]), 1, B.value_type if not skip_reduce else sint)
        B_mat[:] = B[j][:]

        res = A.dot(B_mat, n_threads=n_threads)
        print(A.shape, res.shape, L2[j].shape)
        L2[j] = res

        # @lib.for_range_multithread(n_threads, 1, len(A))
        # def g(i):
        #     L2[i][j] = A.value_type.dot_product(A[i], B[j])

    lib.stop_timer(110)
    return L2.transpose(), None, None


def compute_magnitudes(A: MultiArray, n_threads):
    """
    Computes the l2 norm of the vectors

    Following: https://nenadmarkus.com/p/all-pairs-euclidean/

    Args:
        A (MultiArray): P x M

    Returns:
        Array of l2 norms
    """
    assert len(A.shape) == 2, "Only 2D arrays supported"

    print("Compute Magnitudes")
    aTa = Array(len(A) , A.value_type)
    @lib.for_range_opt_multithread(n_threads, 1, len(A))
    def f(i):
        # aTa[i] = A[i].dot(A[i])
        aTa[i] = A.value_type.dot_product(A[i], A[i])

    # print(A, "A type", A.shape)

    # aTa = A.dot(A.transpose())
    # aTa = Matrix(len(A), len(A), A.value_type)
    # print(aTa, "ata")
    # # print(aTa.diag())
    #
    # arr = Array.create_from(aTa.diag())
    # arr = Array(len(A), A.value_type)

    return aTa



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
    lib.start_timer(108)
    # _train_labels_idx = Array(len(train_samples), sint)
    # @lib.for_range_opt_multithread(config.n_threads, len(train_samples))
    # def _(i):
    #     _train_labels_idx[i] = ml.argmax(_train_labels[i])

    lib.stop_timer(108)
    # for each train samples -> build ownership array
    train_samples_ownership = Array(len(train_samples), sint)

    for party_id in range(input_loader.num_parties()):
        start, size = input_loader.train_dataset_region(party_id)
        train_samples_ownership.get_sub(start=start, stop=start+size).assign_all(party_id)

    # Load Audit Trigger Samples
    audit_trigger_samples, _audit_trigger_mislabels = input_loader.audit_trigger()


    #first = audit_trigger_samples.get_part(0, 1)
    #print_ln("first=%s", first.reveal_nested())
    #return {"red": None}

    model = input_loader.model()
    latent_space_layer, expected_latent_space_size = input_loader.model_latent_space_layer()

    lib.start_timer(101)
    print_ln("Computing Latent Space for Training Set...")

    # Model.eval seems to take a really long time
    model.layers[-1].compute_loss = False

    train_samples_latent_space = MultiArray([config.n_checkpoints, len(train_samples), expected_latent_space_size], sfix)
    audit_trigger_samples_latent_space = MultiArray([config.n_checkpoints, len(audit_trigger_samples), expected_latent_space_size], sfix)
    train_samples_latent_space.assign_all(sfix(0))
    audit_trigger_samples_latent_space.assign_all(sfix(0))

    if not config.skip_inference:
        print_ln("Computing inference")
        # do this n_checkpoints - 1 more times
        @lib.for_range_opt(config.n_checkpoints)
        def f(i):
            train_samples_latent_space[i] = model.eval(train_samples, batch_size=config.batch_size, latent_space_layer=latent_space_layer)
            assert train_samples_latent_space.sizes == (config.n_checkpoints, len(train_samples), expected_latent_space_size), f"{train_samples_latent_space.sizes} != {(len(train_samples), expected_latent_space_size)}"

            print_ln("Computing Latent Space for Audit Trigger...")
            audit_trigger_samples_latent_space[i] = model.eval(audit_trigger_samples, batch_size=config.batch_size, latent_space_layer=latent_space_layer)
            assert  audit_trigger_samples_latent_space.sizes == (config.n_checkpoints, len(audit_trigger_samples), expected_latent_space_size), f"{audit_trigger_samples_latent_space.sizes} != {(len(audit_trigger_samples), expected_latent_space_size)}"

    lib.stop_timer(101)
    lib.start_timer(102)
    # Score computation

    print(audit_trigger_samples_latent_space.sizes, train_samples_latent_space.total_size())
    print_ln("Computing scores...")
    thetas = Array(config.n_checkpoints, sfix)
    thetas.assign_all(sfix(1))

    total_scores, train_samples_idx = compute_score(config.score_method, config.n_checkpoints, train_samples_latent_space, audit_trigger_samples_latent_space, audit_trigger_samples, train_samples, thetas, config)

    L2 = total_scores

    lib.stop_timer(102)


    n_audit_trigger_samples = len(audit_trigger_samples)

    knn_shapley_values = MultiArray([n_audit_trigger_samples, n_train_samples], sfix)
    knn_shapley_values.assign_all(-1)

    @lib.for_range_opt(n_audit_trigger_samples)
    def sort(audit_trigger_idx):
        print_ln("  audit_trigger_idx=%s", audit_trigger_idx)

        audit_trigger_label = _audit_trigger_mislabels[audit_trigger_idx]
        audit_trigger_label_idx = ml.argmax(audit_trigger_label)

        print("L2")
        dist_arr = L2[audit_trigger_idx]

        print(dist_arr, train_samples_idx[audit_trigger_idx])

        data = concatenate([dist_arr, train_samples_idx[audit_trigger_idx]], axis=1)
        print("DATA SHAPE", data.sizes)

        lib.start_timer(timer_id=103)
        if debug:
            print_ln("Before sort: %s", data.get_part(0, 10).reveal_nested())
        data.sort()

        lib.stop_timer(timer_id=103)

        assert data.sizes == (total_scores.sizes[1], 2), f"top k {data.sizes}"

    # From here now select the top-k or something


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
    #
    # assert knn_shapley_values.sizes == (n_audit_trigger_samples, n_train_samples), f"knn_party_count={knn_shapley_values.sizes}"

    result = {"scores": knn_shapley_values}
    # result = {"shapley_values": []}


    if debug:
    #     # result["knn_party_count"] = knn_party_count
    #     # result["knn_sample_id"] = knn_sample_id
    #     # result["mad_score_matrix"] = mad_score_matrix
    #     debug_output["l2_distances"] = L2.get_part(0, 3)
        pass

    return result

def compute_score(score_method, n_checkpoints, train_samples_latent_space, audit_trigger_samples_latent_space, audit_trigger_samples, train_samples, thetas, config):
    if score_method == "l2":
        total_scores = Matrix(len(audit_trigger_samples), len(train_samples), sfix)
        total_scores.assign_all(sfix(0))
        compute_score_euclid(n_checkpoints, train_samples_latent_space, audit_trigger_samples_latent_space, audit_trigger_samples, train_samples, total_scores, thetas, config)
        train_samples_idx = Array.create_from(cint(x) for x in range(len(train_samples)))
        train_samples_idx_matrix = Matrix.create_from([train_samples_idx for _ in range(len(audit_trigger_samples))])
        return total_scores, train_samples_idx_matrix
    elif score_method == "cosine":
        total_scores = Matrix(len(audit_trigger_samples), len(train_samples), sfix)
        total_scores.assign_all(sfix(0))
        compute_score_cosine(n_checkpoints,train_samples_latent_space, audit_trigger_samples_latent_space, audit_trigger_samples, train_samples, total_scores, thetas, config)
        train_samples_idx = Array.create_from(cint(x) for x in range(len(train_samples)))
        train_samples_idx_matrix = Matrix.create_from([train_samples_idx for _ in range(len(audit_trigger_samples))])
        return total_scores, train_samples_idx_matrix
    elif score_method == "cosine_opt_defer_div":
        total_scores = Matrix(len(audit_trigger_samples), len(train_samples), sfix)
        total_scores.assign_all(sfix(0))
        compute_score_cosine_opt_defer_div(n_checkpoints, train_samples_latent_space, audit_trigger_samples_latent_space, audit_trigger_samples, train_samples, total_scores, thetas, config)
        train_samples_idx = Array.create_from(cint(x) for x in range(len(train_samples)))
        train_samples_idx_matrix = Matrix.create_from([train_samples_idx for _ in range(len(audit_trigger_samples))])
        return total_scores, train_samples_idx_matrix
    elif score_method == "cosine_presort_l2":
        assert config.pre_score_select_k is not None
        total_scores = Matrix(len(audit_trigger_samples), config.pre_score_select_k, sfix)
        total_scores.assign_all(sfix(0))
        total_scores_out, samples_idx = compute_score_cosine_opt_presort_l2(n_checkpoints, train_samples_latent_space, audit_trigger_samples_latent_space, audit_trigger_samples, train_samples, total_scores, thetas, config)
        print("total_scores_out", total_scores_out.sizes, total_scores_out.reveal_nested())
        return total_scores_out, samples_idx
    elif score_method == "cosine_presort_l2_compute":
        assert config.pre_score_select_k is not None
        total_scores = Matrix(len(audit_trigger_samples), config.pre_score_select_k, sfix)
        total_scores.assign_all(sfix(0))
        total_scores_out, samples_idx = compute_score_cosine_opt_presort_l2(n_checkpoints, train_samples_latent_space, audit_trigger_samples_latent_space, audit_trigger_samples, train_samples, total_scores, thetas, config)
        print("total_scores_out", total_scores_out.sizes, total_scores_out.reveal_nested())
        return total_scores_out, samples_idx
    else:
        raise ValueError(f"Score method {score_method} not supported!")


def compute_score_euclid(n_checkpoints, train_samples_latent_space, audit_trigger_samples_latent_space, audit_trigger_samples, train_samples, total_scores, thetas, config):

    lib.start_timer(timer_id=105)
    # TODO: Runtime loop
    for checkpoint_id in range(n_checkpoints):
    # @lib.for_range_opt(n_checkpoints)
    # def s(checkpoint_id):
        score, aTa, bTb = euclidean_distance_naive(A=train_samples_latent_space[checkpoint_id],
                                                         B=audit_trigger_samples_latent_space[checkpoint_id],
                                                         n_threads=config.n_threads)
        score = score.transpose()
        assert score.sizes == (len(audit_trigger_samples), len(train_samples)), f"L2 {score.sizes}"
        @lib.for_range_opt_multithread(config.n_threads, total_scores.sizes[0])
        def f(i):
            total_scores[i] = total_scores[i] + (score[i] * thetas[checkpoint_id])
    lib.stop_timer(timer_id=105)

def compute_score_cosine(n_checkpoints, train_samples_latent_space, audit_trigger_samples_latent_space, audit_trigger_samples, train_samples, total_scores, thetas, config):
    lib.start_timer(timer_id=105)
    # TODO: Runtime loop
    for checkpoint_id in range(n_checkpoints):
    # @lib.for_range_opt(n_checkpoints)
    # def s(checkpoint_id):
        score = cosine_distance(A=train_samples_latent_space[checkpoint_id], B=audit_trigger_samples_latent_space[checkpoint_id], n_threads=config.n_threads)
        score = score.transpose()
        assert score.sizes == (len(audit_trigger_samples), len(train_samples)), f"L2 {score.sizes}"

        @lib.for_range_opt_multithread(config.n_threads, total_scores.sizes[0])
        def f(i):
            total_scores[i] = total_scores[i] + (score[i] * thetas[checkpoint_id])
    lib.stop_timer(timer_id=105)




def compute_score_cosine_opt_presort_l2(n_checkpoints, train_samples_latent_space, audit_trigger_samples_latent_space, audit_trigger_samples, train_samples, total_scores, thetas, config):
    # TODO: Correctness?

    defer_truncate = False

    all_scores_l2 = MultiArray([n_checkpoints, len(audit_trigger_samples), len(train_samples)], sfix)
    all_multipliers = MultiArray([n_checkpoints, len(audit_trigger_samples), len(train_samples)], sfix)

    total_scores_l2 = Matrix(len(audit_trigger_samples), len(train_samples), sfix)
    total_scores_l2.assign_all(sfix(0))

    lib.start_timer(timer_id=105)

    # @lib.for_range_opt(n_checkpoints)
    # def s(checkpoint_id):
    for checkpoint_id in range(n_checkpoints):
        score, _, _ = euclidean_distance_naive(A=train_samples_latent_space[checkpoint_id],
                                             B=audit_trigger_samples_latent_space[checkpoint_id],
                                             n_threads=config.n_threads)
        score = score.transpose()
        assert score.sizes == (len(audit_trigger_samples), len(train_samples)), f"L2 {score.sizes}"
        all_scores_l2[checkpoint_id] = score

        # Do compute those here because its probably slow
        aTa = compute_magnitudes(train_samples_latent_space[checkpoint_id], config.n_threads)
        bTb = compute_magnitudes(audit_trigger_samples_latent_space[checkpoint_id], config.n_threads)

        @lib.for_range_opt_multithread(config.n_threads, total_scores_l2.sizes[0])
        def f(j):
            total_scores_l2[:] = total_scores_l2[:] + (score[j] * thetas[checkpoint_id]) # TODO: I think?
            all_multipliers[checkpoint_id][j] = aTa[j] * bTb

        print_ln("Done score transpose after")
    print_ln("Done score for range checkpoints")

    lib.stop_timer(timer_id=105)

    n_train_samples = len(train_samples)
    train_samples_idx = Array.create_from(cint(x) for x in range(n_train_samples))

    # we need to flatten all_scores_l2 and all_multipliers
    all_scores_l2_flat = [all_scores_l2[i] for i in range(n_checkpoints)]
    all_multipliers_flat = [all_multipliers[i] for i in range(n_checkpoints)]

    presort_idx = Matrix(len(audit_trigger_samples), config.pre_score_select_k, sfix)
    presort_idx.assign_all(sfix(0))

    print_ln("Prescoring l2 done")

    # sort by total_scores and select k
    # @lib.for_range_opt(len(audit_trigger_samples))
    # def sort(audit_trigger_idx):
    for audit_trigger_idx in range(len(audit_trigger_samples)):
        print_ln("  audit_trigger_idx=%s", audit_trigger_idx)

        print("L2")
        dist_arr = total_scores_l2[audit_trigger_idx]

        # for now we let the sort also sort our precomputed l2s and multipliers
        all_scores_l2_flat_trigger = [all_scores_l2_flat[i][audit_trigger_idx] for i in range(n_checkpoints)]
        all_multipliers_flat_trigger = [all_multipliers_flat[i][audit_trigger_idx] for i in range(n_checkpoints)]

        # If we want to sort aLL!
        # train_sample_latent_len = train_samples_latent_space.shape[2]
        # print(train_sample_latent_len, "latent len")
        # matrix_sample_by_latent = Matrix(n_train_samples, train_sample_latent_len * n_checkpoints, sfix)
        # @lib.for_range_opt(n_checkpoints)
        # def f(checkpoint_id):
        #     matrix_sample_by_latent.assign(train_samples_latent_space[checkpoint_id], len(train_samples) * checkpoint_id * train_sample_latent_len)

        # all_latent_flat = [train_samples_latent_space[i] for i in range(n_checkpoints)]
        # all_latent_flat = [[all_latent_flat[i] for j in range(train_sample_latent_len)] for i in range(n_checkpoints)]
        # print("matrix_sample_by_latent", matrix_sample_by_latent)
        # print("tyoes", all_scores_l2_flat_trigger, all_multipliers_flat_trigger)

        data = concatenate([dist_arr, train_samples_idx] + all_scores_l2_flat_trigger + all_multipliers_flat_trigger, axis=1)
        # data = data.transpose().concat(matrix_sample_by_latent.transpose()).transpose()
        print("DATA SHAPE", data.sizes)

        lib.start_timer(timer_id=106)
        if config.debug:
            print_ln("Before sort: %s", data.get_part(0, 10).reveal_nested())
        data.sort()

        lib.stop_timer(timer_id=106)

        print_ln("Sort done")

        # we want to go over checkpoints to compute cosine, and then add to l2_scores
        scores_top_pre_k = data.get_part(0, config.pre_score_select_k)

        print("scores_top_pre_k", scores_top_pre_k.sizes, scores_top_pre_k[2 + 1])

        lib.start_timer(timer_id=107)

        @lib.for_range_opt_multithread(config.n_threads, config.pre_score_select_k)
        def f(j):
            # @lib.map_sum_opt(config.n_threads, n_checkpoints, [sfix])
            # def summer(i):
            #     return scores_top_pre_k[j][2 + i] / scores_top_pre_k[j][n_checkpoints + 2 + i]
            #
            sum = MemValue(sfix(0))
            @lib.for_range_opt(n_checkpoints)
            def s(i):
                # now we need to compute the latent space distance in each checkpoint
                sum.iadd(scores_top_pre_k[j][2 + i] / scores_top_pre_k[j][n_checkpoints + 2 + i])

            total_scores[audit_trigger_idx][j] = sum
            presort_idx[audit_trigger_idx][j] = scores_top_pre_k[j][1]

        lib.stop_timer(timer_id=107)

    # Somehow this reveal here causes non-determinism in compilation
    # print(total_scores)
    # print_ln(total_scores.reveal_nested())

    return total_scores, presort_idx




def compute_score_cosine_opt_defer_div(n_checkpoints, train_samples_latent_space, audit_trigger_samples_latent_space, audit_trigger_samples, train_samples, total_scores, thetas, config):
    # TODO: Correctness?

    all_scores = MultiArray([n_checkpoints, len(audit_trigger_samples), len(train_samples)], sfix)
    all_multipliers = MultiArray([n_checkpoints, len(audit_trigger_samples), len(train_samples)], sfix)

    @lib.for_range_opt(n_checkpoints)
    def s(i):
        score, aTa, bTb = euclidean_distance(A=train_samples_latent_space,
                                             B=audit_trigger_samples_latent_space,
                                             n_threads=config.n_threads)
        score = score.transpose()
        assert score.sizes == (len(audit_trigger_samples), len(train_samples)), f"L2 {score.sizes}"

        all_scores[i].assign(score)

        @lib.for_range_opt_multithread(config.n_threads, total_scores.sizes[0])
        def f(j):
            all_multipliers[i][j] = aTa[j] * bTb

    cumulative_multiplier = MultiArray([len(audit_trigger_samples), len(train_samples)], sfix)
    cumulative_multiplier.assign_all(sfix(1))
    # now we look over the checkpoints and apply to each row i all multipliers from the other rows
    @lib.for_range_opt(n_checkpoints)
    def s(i):
        @lib.for_range_opt_multithread(config.n_threads, all_multipliers[i].sizes[0])
        def f(k):
            cumulative_multiplier[k] = cumulative_multiplier[k] * all_multipliers[i][k]

        for j in range(n_checkpoints):
            # if i != j:
            @lib.if_(i != j)
            def f():
                @lib.for_range_opt_multithread(config.n_threads, all_scores[i].sizes[0])
                def f(k):
                    all_scores[i][k] = all_scores[i][k] * all_multipliers[j][k]

        # now we add all
        @lib.for_range_opt_multithread(config.n_threads, all_scores[i].sizes[0])
        def f(k):
            total_scores[k] = total_scores[k] + (all_scores[i][k] * thetas[i])

    # and now we divide all by cumulative multiplier
    @lib.for_range_opt_multithread(config.n_threads, total_scores.sizes)
    def f(i, j):
        print(total_scores[i], cumulative_multiplier[i])
        total_scores[i][j] = total_scores[i][j] / cumulative_multiplier[i][j]

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