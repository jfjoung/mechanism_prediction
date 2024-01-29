
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.backend as K
from WLN.layers import WLN_Layer, WLN_Edit, Global_Attention, WL_DiffNet

class WLN_Regressor(tf.keras.Model):
    '''
    A simple NN regressor that uses the WLN graph convolution procedure as the embedding layer

    '''
    def __init__(self, hidden_size, depth, max_nb=10):
        super(WLN_Regressor, self).__init__()
        self.WLN = WLN_Layer(hidden_size, depth, max_nb)
        self.layer1 = layers.Dense(1)

    def call(self, inputs):
        x = self.WLN(inputs)
        x = K.sum(x, axis=-2, keepdims=False)
        x = self.layer1(x)
        return x

class WLNPairwiseAtomClassifier(tf.keras.Model):
    '''
     A Keras class for implementation ICML paper Predicting Organic Reaction Outcomes with Weisfeiler-Lehman Network

    Init
    hidden_size: The hidden size of the dense layers
    depth: How many iterations that a new representation of each atom is computed. Each iteration goes one atom further away from the
           initial starting point.  The number of distinct labels from the WLN grows ponentially with the number of iterations
    output_dim: the dimention corresponding to bond type changes. Usually 5 and should equal the N_BOND_CLASS in ioutils_direct
    max_nb: Max number of bonds. Generally set at 10 and is specified by the graph generation procedure for the inputs

    Inputs
    inputs: molecular graph that has atom features, bond features, the atom attachments, bond attachments
                  number of bonds for each atom, and a node mask since batches have to be padded

    Output
    score: Tensor of flattened atom scores
    '''
    def __init__(self, hidden_size, depth, output_dim=5, H_dim=2, FC_dim=4, max_nb=10):  # Changed to have H_dim and FC_dim
        super(WLNPairwiseAtomClassifier, self).__init__()
        initializer = tf.random_normal_initializer(stddev=0.01)
        self.hidden_size = hidden_size
        self.WLN = WLN_Layer(hidden_size, depth, max_nb)
        self.attention = Global_Attention(hidden_size)
        self.atom_feature = layers.Dense(hidden_size,kernel_initializer=tf.random_normal_initializer(stddev=0.01), use_bias=False)
        self.bin_feature = layers.Dense(hidden_size, kernel_initializer=tf.random_normal_initializer(stddev=0.01), use_bias=False)
        self.ctx_feature = layers.Dense(hidden_size,kernel_initializer=tf.random_normal_initializer(stddev=0.01))
        self.score = layers.Dense(output_dim,kernel_initializer=tf.random_normal_initializer(stddev=0.01))
        self.H_score=layers.Dense(H_dim,kernel_initializer=tf.random_normal_initializer(stddev=0.01))
        self.FC_score=layers.Dense(FC_dim,kernel_initializer=tf.random_normal_initializer(stddev=0.01))
        self.reshape1 = layers.Reshape((1,-1,hidden_size))
        self.reshape2 = layers.Reshape((-1,1,hidden_size))
        self.reshape_pair = layers.Reshape((-1, hidden_size))
        self.reshape_score = layers.Reshape((-1,))

    @tf.function(input_signature=[[
        tf.TensorSpec(shape=[None, None, 102], dtype=tf.float32, name="input_1"),  # pack2D(fatom_list)  ##82 is changed to 102
        tf.TensorSpec(shape=[None, None, 6], dtype=tf.float32, name="input_2"),   # pack2D(fbond_list)
        tf.TensorSpec(shape=[None, None, 10, 2], dtype=tf.float32, name="input_3"), # pack2D_withidx(gatom_list)
        tf.TensorSpec(shape=[None, None, 10, 2], dtype=tf.float32, name="input_4"), # pack2D_withidx(gbond_list)
        tf.TensorSpec(shape=[None, None], dtype=tf.float32, name="input_5"),        # pack1D(nb_list)
        tf.TensorSpec(shape=[None, None], dtype=tf.float32, name="input_6"),        # get_mask(fatom_list)
        tf.TensorSpec(shape=[None, None, None, 11], dtype=tf.float32, name="input_7") #  binary_features_batch(smiles_list)
    ]])
    def call(self, inputs):
        atom_hidden,_ = self.WLN(inputs)
        att_context, atom_pair = self.attention(atom_hidden, inputs[-1])
        att_context1 = self.reshape1(att_context)
        att_context2 = self.reshape2(att_context)
        att_pair = att_context1 + att_context2
        pair_hidden = self.atom_feature(atom_pair) + self.bin_feature(inputs[-1]) + self.ctx_feature(att_pair)
        pair_hidden = K.relu(pair_hidden)
        pair_hidden = self.reshape_pair(pair_hidden)
        bond_score = self.score(pair_hidden)
        bond_score = self.reshape_score(bond_score)

        H_score = self.H_score(atom_hidden)
        H_score = self.reshape_score(H_score)
        FC_score = self.FC_score(atom_hidden)
        FC_score = self.reshape_score(FC_score)

        score=K.concatenate([bond_score, H_score, FC_score])
        return score

    #added for convenience. This model does not need explicit shapes defined when saving model
    @tf.function
    def save_model(self, inputs):
        return self.call(inputs)


class WLNCandidateRanker(tf.keras.Model):
    '''
     A Keras class for implementation ICML paper Predicting Organic Reaction Outcomes with Weisfeiler-Lehman Network

    Init
    hidden_size: The hidden size of the dense layers
    depth: How many iterations that a new representation of each atom is computed. Each iteration goes one atom further away from the
           initial starting point.  The number of distinct labels from the WLN grows ponentially with the number of iterations
    max_nb: Max number of bonds. Generally set at 10 and is specified by the graph generation procedure for the inputs

    Inputs
    inputs: molecular graph that has atom features, bond features, the atom attachments, bond attachments
                  number of bonds for each atom

    Output
    score: Tensor of scores for the Candidates
    '''
    def __init__(self, hidden_size, depth, max_nb=10):
        super(WLNCandidateRanker, self).__init__()
        self.hidden_size = hidden_size
        self.WLN = WLN_Edit(hidden_size, depth, max_nb)
        self.WL_DiffNet = WL_DiffNet(hidden_size, depth=1, max_nb=max_nb)
        self.rex_hidden = layers.Dense(hidden_size, activation=K.relu, kernel_initializer=tf.random_normal_initializer(stddev=0.01))
        self.score = layers.Dense(1, kernel_initializer=tf.random_normal_initializer(stddev=0.01))

    @tf.function(input_signature=[[
        tf.TensorSpec(shape=[None, None, None, 102], dtype=tf.float32, name="input_1"),
        tf.TensorSpec(shape=[None, None, None, 5], dtype=tf.float32, name="input_2"),
        tf.TensorSpec(shape=[None, None, None, 10, 2], dtype=tf.float32, name="input_3"),
        tf.TensorSpec(shape=[None, None, None, 10, 2], dtype=tf.float32, name="input_4"),
        tf.TensorSpec(shape=[None, None, None], dtype=tf.int32, name="input_5"),
        tf.TensorSpec(shape=[None, None], dtype=tf.float32, name="input_6")
    ]])
    def call(self, inputs):
        #unpack to make easier
        core_bias = inputs[5]
        inputs = (K.squeeze(inputs[0], axis=0), K.squeeze(inputs[1], axis=0),K.squeeze(inputs[2], axis=0),K.squeeze(inputs[3], axis=0),K.squeeze(inputs[4], axis=0))#,K.transpose(inputs[5]))
        graph_inputs = inputs[:5]
        fp_all_atoms = self.WLN(inputs)
        reactant = fp_all_atoms[0:1,:]
        candidates = fp_all_atoms[1:,:]
        candidates = candidates - reactant
        candidates = tf.concat([reactant, candidates], axis=0)
        reaction_fp = self.WL_DiffNet(graph_inputs, candidates)
        reaction_fp = self.rex_hidden(reaction_fp[1:])
        scores = K.squeeze(self.score(reaction_fp), axis=1) + core_bias

        return scores

    #Workaround because tf.function decorator on call makes the training slower by reconstructing
    # the graph on every step. Need tf.function to save the model and define shapes of tensors
    @tf.function
    def save_model(self, inputs):
        return self.call(inputs)
