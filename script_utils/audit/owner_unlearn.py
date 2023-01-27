from typing import List, Any, Union

from Compiler.types import MultiArray, Array
from Compiler.types import sfix, cfix, MemValue

from Compiler import ml


from Compiler.library import print_ln
from Compiler.library import for_range_opt, for_range


from Compiler.script_utils import audit_function_utils as audit_utils



def audit(input_loader, params, debug: bool):

    # TODO [nku] should reflect the debug

    model = input_loader.model()
    train_samples, train_labels = input_loader.train_dataset()

    input_loader.train_dataset()

    audit_trigger_samples, audit_trigger_mislabels = input_loader.audit_trigger()


    batch_size = 128 # TODO [nku] this needs to come from somewhere
    n_samples = len(audit_trigger_samples) # TODO [nku] the number of samples (triggers) needs to come from somewhere
    audit_trigger_batch_size = min(batch_size, n_samples)


    backup_variables = backup_trainable_vars(model)

    # extract audit function parameters
    learning_rate = float(params.get('learning_rate', 0.01))
    #n_thread_num = int(params.get('n_num_threads', 16))
    n_data_owners = int(params.get('n_data_owners', 4))

    # TODO [nku] also consider this: get_number_of_players() -> difference between parties and owners I suppose?

    sample_cutoff_size = int(params.get('sample_cutoff_size', -1))


    # preparing unlearning optimizer
    train_optimizer = model.opt
    train_optimizer.n_epochs = 1 # TODO [nku] should probably be also params?
    train_optimizer.report_losses = True
    train_optimizer.print_losses = True
    train_optimizer.gamma = MemValue(cfix(learning_rate))


    # TODO [nku] at the moment, this is mnist specific + does not respect the actual data distribution
    unlearn_size = train_labels.sizes[0] // n_data_owners
    assert train_labels.sizes[0] % n_data_owners == 0, "Must be cleanly divisible"

    unlearn_labels = MultiArray([unlearn_size, 10], sfix)
    unlearn_labels.assign_all(sfix(1.0 / 10))

    unlearned_predictions = MultiArray([n_data_owners, n_samples, 10], sfix)

    # for party_id range(n_data_owners):
    @for_range_opt(n_data_owners)  # TODO: could we use for_range_multithread and create a separate copy of each model
    def unlearn_party(data_owner_id):

        # TODO: unlearning method changed in analysis => change should be reflected here also

        print_ln("Unlearning data from owner %s", data_owner_id)

        # TODO [nku] need to update once every party has own data
        modified_samples, modified_labels = audit_utils.get_unlearn_data_for_party(data_owner_id,
                                                                                train_samples, train_labels,
                                                                                    unlearn_labels, unlearn_size)

        # restore the initial variable state -> could skip for first data owner but not sure how it works

        for i, (model_var, initial_var) in  enumerate(zip(model.trainable_variables, backup_variables, strict=True)):

            print_ln("  restoring trainable_variable %s", i)
            #assert type(model_var) == type(initial_var), f"{type(model_var)}     {type(initial_var)}" # TODO [nku] not sure how this works in mpc?

            model_var.assign(initial_var) # TODO [nku] check if this actually is enough to restore


        # [bugfix]
        # In ml.py on line 2674 build sets a static value, (called by `print_predict_accuracy_layers`)
        # which messes up the batch_batch_size for an already created model.
        ml.Layer.back_batch_size = batch_size


        # run an epoch of unlearning
        train_optimizer.layers[-1].Y.address = modified_labels.address
        train_optimizer.layers[0].X.address = modified_samples.address
        train_optimizer.run(batch_size=batch_size)

        guesses = audit_utils.predict_on_model_copy(input_loader.model_layers(), model, audit_trigger_batch_size,
                                                            audit_trigger_samples, audit_trigger_mislabels)

        if debug:
            audit_utils.reveal_accuracy(preds=guesses, labels=audit_trigger_mislabels)

        unlearned_predictions[data_owner_id] = guesses


    loss_array = audit_utils.compute_loss(unlearned_predictions, audit_trigger_mislabels)

    if debug:
        print_ln("Losses %s", loss_array.reveal_nested())



    # Compute MAD Scores
    mad_score_matrix = compute_MAD_matrix(loss_array)


    # TODO : MAD matrxi aggregation for thresholding

    # TODO: output adjustment + debug flag

    return {
        "loss_matrix": loss_array,
        "mad_score_matrix": mad_score_matrix,
        "audit_trigger_labels": audit_trigger_mislabels,
        #"unmodified_prediction_results": original_prediction_results,
        "unlearned_prediction_results": unlearned_predictions
    }


def compute_MAD_matrix(loss_matrix):
    MAD_array = MultiArray(list(loss_matrix.sizes), sfix)

    @for_range(start=0, stop=loss_matrix.sizes[0], step=1)
    def _(i):
        loss_array = loss_matrix.get_part(i, 1)[0]
        score_array = audit_utils.MAD_Score(loss_array)
        MAD_array.get_part(i, 1).assign(score_array)

    return MAD_array








def backup_trainable_vars(model):
    backup: List[Union[MultiArray, Array]] = list()

    for i, var in enumerate(model.trainable_variables):

        copy_container = var.same_shape()
        copy_container.assign(var)
        backup.append(copy_container)
        ## TODO: could maybe var.same_shape()n to get a copy container
        #if isinstance(var, MultiArray):
        #    copy_container = MultiArray(var.sizes, var.value_type)
        #    copy_container.assign(var)
        #    backup.append(copy_container)
        #elif isinstance(var, Array):
        #    copy_container = Array(var.length, var.value_type)
        #    copy_container.assign(var)
        #    backup.append(copy_container)
        #else:
        #    raise ValueError(f"Error, var {type(var)} not supported for copy!")
    return backup
