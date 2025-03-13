import numpy as np
import pandas as pd

#################### WEIGHT PRUNING ####################

# We consider the *item embedding* layer and build a dictionary where each entry is like ((row, column): weight).
# The dictionary is converted to a list of tuples then sorted in ascending order w.r.t. the weights absolut values. 
# Depending on the pruning percentage specified, the first weights of the sorted list are set to 0.
# The others instead are kept as they were before.
def weight_pruning(layer, pruning_percentage):
    emb_weights = (pd.DataFrame(layer.weight.data).stack()).to_dict() 
    emb_weights = { (k[0], k[1]): v for k, v in emb_weights.items() }

    emb_weights_list = list(emb_weights.items())
    all_weights_sorted = sorted(emb_weights_list, key=lambda item: abs(item[1]))

    total_emb_weights = len(all_weights_sorted) 
    trained_weights = layer.weight.data
    prune_fraction = pruning_percentage/100

    number_of_weights_to_be_pruned = int(prune_fraction*total_emb_weights)
    weights_to_be_pruned = [(k) for k, v in all_weights_sorted[:number_of_weights_to_be_pruned]]

    for (k, v) in weights_to_be_pruned:
        trained_weights[k, v] = 0

    layer.weight.data = trained_weights  



#################### NEURON PRUNING ####################

# We consider the item embedding layer and build a dictionary where each entry is like
# (column_index: column_values_array), i.e. (neuron, neuron weights). 
# The dictionary is converted to a list of tuples and then sorted in ascending order w.r.t. the L2 norm of each neuron weight vector.
# Depending on the pruning percentage specified, all the weights of the first neurons of the sorted list
# are set to 0. The weights of the other neurons instead are kept as they were before.
def neuron_pruning(layer, pruning_percentage):
    emb_neurons = {}
    emb_neurons_df = pd.DataFrame(layer.weight.data)

    # every column index of the emb_matrix becomes a key in the emb_neurons dict and the associated value is the corresponding column
    for i in range(len(emb_neurons_df.columns)):
        emb_neurons.update({ i :np.array(emb_neurons_df.iloc[:,i] ) })  

    emb_neurons_list = list(emb_neurons.items())

    emb_neurons_sorted = sorted(emb_neurons.items(), key=lambda item: np.linalg.norm(item[1], ord=2, axis=0))

    total_neurons = len(emb_neurons_sorted)
    trained_weights = layer.weight.data
    prune_fraction = pruning_percentage/100
    number_of_neurons_to_be_pruned = int(prune_fraction*total_neurons)

    neurons_to_be_pruned = [(k) for k, v in emb_neurons_sorted[:number_of_neurons_to_be_pruned]]  

    for k in neurons_to_be_pruned:
        trained_weights[:, k] = 0

    layer.weight.data = trained_weights

    return neurons_to_be_pruned



#################### LAZY NEURON PRUNING ####################

# We set to zero the last emb_size/2 columns of the item embedding matrix
def lazy_neuron_pruning(layer, pruning_percentage):
    emb_neurons = {}
    emb_neurons_df = pd.DataFrame(layer.weight.data)

    # every column index of the emb_matrix becomes a key in the emb_neurons dict and the associated value is the corresponding column
    for i in range(len(emb_neurons_df.columns)):
        emb_neurons.update({ i :np.array(emb_neurons_df.iloc[:,i] ) })  

    emb_neurons_list = list(emb_neurons.items())

    total_neurons = len(emb_neurons_list)
    trained_weights = layer.weight.data
    prune_fraction = pruning_percentage/100
    number_of_neurons_to_be_pruned = int(prune_fraction*total_neurons)

    neurons_to_be_pruned = [(k) for k, v in emb_neurons_list[-number_of_neurons_to_be_pruned:]]  

    for k in neurons_to_be_pruned:
        trained_weights[:, k] = 0

    layer.weight.data = trained_weights

    return neurons_to_be_pruned