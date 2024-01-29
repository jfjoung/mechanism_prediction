import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K

class WLN_Layer(tf.keras.layers.Layer):
    '''
    A Keras class for implementation ICML paper Predicting Organic Reaction Outcomes with Weisfeiler-Lehman Network

    Init
    hidden_size: The hidden size of the dense layers
    depth: How many iterations that a new representation of each atom is computed. Each iteration goes one atom further away from the
           initial starting point.  The number of distinct labels from the WLN grows ponentially with the number of iterations
    max_nb: Max number of bonds. Generally set at 10 and is specified by the graph generation procedure for the inputs

    Inputs
    graph_inputs: molecular graph that has atom features, bond features, the atom attachments, bond attachments
                  number of bonds for each atom, and a node mask since batches have to be padded

    Output
    kernels: The WLN graph kernal which is the updated representation of each atom
    '''
    def __init__(self, hidden_size, depth, max_nb=10):
        super(WLN_Layer, self).__init__()
        self.hidden_size = hidden_size
        self.depth = depth
        self.max_nb = max_nb

    def build(self, input_shape):
        self.atom_features = layers.Dense(self.hidden_size, kernel_initializer=tf.random_normal_initializer(stddev=0.01), use_bias=False, input_shape=(83,))
        self.nei_atom = layers.Dense(self.hidden_size, kernel_initializer=tf.random_normal_initializer(stddev=0.01), use_bias=False)
        self.nei_bond = layers.Dense(self.hidden_size, kernel_initializer=tf.random_normal_initializer(stddev=0.01), use_bias=False)
        self.self_atom = layers.Dense(self.hidden_size, kernel_initializer=tf.random_normal_initializer(stddev=0.01), use_bias=False)
        self.label_U2 = layers.Dense(self.hidden_size, activation=K.relu, kernel_initializer=tf.random_normal_initializer(stddev=0.01))
        self.label_U1 = layers.Dense(self.hidden_size, activation=K.relu, kernel_initializer=tf.random_normal_initializer(stddev=0.01))
        self.node_reshape = layers.Reshape((-1,1))
        super(WLN_Layer, self).build(input_shape)

    def get_config(self):
        return {"atom_features": self.atom_features,
                'nei_atom': self.nei_atom,
                'nei_bond' : self.nei_bond,
                'self_atom': self.self_atom,
                'label_U2': self.label_U2,
                'label_U1': self.label_U1
        }

    def call(self, graph_inputs):
        input_atom, input_bond, atom_graph, bond_graph, num_nbs, node_mask,_ = graph_inputs
        #calculate the initial atom features using only its own features (no neighbors)
        atom_features = self.atom_features(input_atom)
        layers = []
        for i in range(self.depth):
            fatom_nei = tf.gather_nd(atom_features, tf.dtypes.cast(atom_graph,tf.int64)) #(batch, #atoms, max_nb, hidden)
            fbond_nei = tf.gather_nd(input_bond, tf.dtypes.cast(bond_graph, tf.int64)) #(batch, #atoms, max_nb, #bond features)
            # print('atom', fatom_nei.shape)
            # print('bond', fbond_nei.shape)
            fatom_nei.set_shape([None,None,self.max_nb, self.hidden_size])
            fbond_nei.set_shape([None,None,self.max_nb,6])
            h_nei_atom = self.nei_atom(fatom_nei) #(batch, #atoms, max_nb, hidden)
            h_nei_bond = self.nei_bond(fbond_nei) #(batch, #atoms, max_nb, hidden)
            h_nei = h_nei_atom * h_nei_bond #(batch, #atoms, max_nb, hidden)
            mask_nei = K.reshape(tf.sequence_mask(K.reshape(num_nbs, [-1]), self.max_nb, dtype=tf.float32), [K.shape(input_atom)[0],-1, self.max_nb,1])
            f_nei = K.sum(h_nei * mask_nei, axis=-2, keepdims=False) #(batch, #atoms, hidden) sum across atoms
            f_self = self.self_atom(atom_features) #(batch, #atoms, hidden)
            layers.append(f_nei * f_self * self.node_reshape(node_mask))#, -1))
            l_nei = K.concatenate([fatom_nei, fbond_nei], axis=3) #(batch, #atoms, max_nb, )
            pre_label = self.label_U2(l_nei)
            nei_label = K.sum(pre_label * mask_nei, axis=-2, keepdims=False)
            new_label = K.concatenate([atom_features, nei_label], axis=2)
            atom_features = self.label_U1(new_label)
        kernels = layers[-1]
        return kernels, atom_features

class WLN_Edit(tf.keras.layers.Layer):
    '''
    A Keras class for implementation ICML paper Predicting Organic Reaction Outcomes with Weisfeiler-Lehman Network

    Init
    hidden_size: The hidden size of the dense layers
    depth: How many iterations that a new representation of each atom is computed. Each iteration goes one atom further away from the
           initial starting point.  The number of distinct labels from the WLN grows ponentially with the number of iterations
    max_nb: Max number of bonds. Generally set at 10 and is specified by the graph generation procedure for the inputs

    Inputs
    graph_inputs: molecular graph that has atom features, bond features, the atom attachments, bond attachments
                  number of bonds for each atom, and a node mask since batches have to be padded

    Output
    kernels: The WLN graph kernal which is the updated representation of each atom
    '''
    def __init__(self, hidden_size, depth, max_nb=10):
        super(WLN_Edit, self).__init__()
        self.hidden_size = hidden_size
        self.depth = depth
        self.max_nb = max_nb

    def build(self, input_shape):
        self.atom_features = layers.Dense(self.hidden_size, kernel_initializer=tf.random_normal_initializer(stddev=0.01), use_bias=False, input_shape=(None,None,89,))
        self.label_U2 = layers.Dense(self.hidden_size, activation=K.relu, kernel_initializer=tf.random_normal_initializer(stddev=0.01))
        self.label_U1 = layers.Dense(self.hidden_size, activation=K.relu, kernel_initializer=tf.random_normal_initializer(stddev=0.01))
        self.node_reshape = layers.Reshape((-1,1))
        super(WLN_Edit, self).build(input_shape)

    def get_config(self):
        return {"atom_features": self.atom_features,
                'label_U2': self.label_U2,
                'label_U1': self.label_U1
        }

    def call(self, graph_inputs):
        input_atom, input_bond, atom_graph, bond_graph, num_nbs = graph_inputs
        #calculate the initial atom features using only its own features (no neighbors)
        atom_features = self.atom_features(input_atom)
        layers = []
        for i in range(self.depth):
            fatom_nei = tf.gather_nd(atom_features, tf.dtypes.cast(atom_graph,tf.int64)) #(batch, #atoms, max_nb, hidden)
            fbond_nei = tf.gather_nd(input_bond, tf.dtypes.cast(bond_graph, tf.int64)) #(batch, #atoms, max_nb, #bond features)
            mask_nei = tf.sequence_mask(K.reshape(num_nbs, [-1]), self.max_nb, dtype=tf.float32)
            target_shape = tf.concat([tf.shape(num_nbs), [self.max_nb, 1]], 0)
            mask_nei = tf.reshape(mask_nei, target_shape)
            mask_nei.set_shape([None, None, self.max_nb, 1])
            l_nei = K.concatenate([fatom_nei, fbond_nei], axis=3) #(batch, #atoms, max_nb, )
            l_nei.set_shape([None,None,self.max_nb, self.hidden_size + 5])
            pre_label = self.label_U2(l_nei)
            nei_label = K.sum(pre_label * mask_nei, axis=-2, keepdims=False)
            new_label = K.concatenate([atom_features, nei_label], axis=2)
            atom_features = self.label_U1(new_label)

        return atom_features


class WL_DiffNet(tf.keras.layers.Layer):
    '''
    A Keras class for implementation ICML paper Predicting Organic Reaction Outcomes with Weisfeiler-Lehman Network

    Init
    hidden_size: The hidden size of the dense layers
    depth: How many iterations that a new representation of each atom is computed. Each iteration goes one atom further away from the
           initial starting point.  The number of distinct labels from the WLN grows ponentially with the number of iterations
    max_nb: Max number of bonds. Generally set at 10 and is specified by the graph generation procedure for the inputs

    Inputs
    graph_inputs: molecular graph that has atom features, bond features, the atom attachments, bond attachments
                  number of bonds for each atom, and a node mask since batches have to be padded
    atom_features: The atom feature embedding from the WLN_Edit layer

    Output
    kernels: The WLN graph kernal which is the updated representation of each atom
    '''
    def __init__(self, hidden_size, depth, max_nb=10):
        super(WL_DiffNet, self).__init__()
        self.hidden_size = hidden_size
        self.depth = depth
        self.max_nb = max_nb

    def build(self, input_shape):

        self.label_U2 = layers.Dense(self.hidden_size, activation=K.relu, kernel_initializer=tf.random_normal_initializer(stddev=0.01))
        self.label_U1 = layers.Dense(self.hidden_size, activation=K.relu, kernel_initializer=tf.random_normal_initializer(stddev=0.01))
        super(WL_DiffNet, self).build(input_shape)

    def call(self, graph_inputs, atom_features):
        input_atom, input_bond, atom_graph, bond_graph, num_nbs = graph_inputs
        for i in range(self.depth):
            fatom_nei = tf.gather_nd(atom_features, tf.dtypes.cast(atom_graph,tf.int64)) #(batch, #atoms, max_nb, hidden)
            fbond_nei = tf.gather_nd(input_bond, tf.dtypes.cast(bond_graph, tf.int64)) #(batch, #atoms, max_nb, #bond features)
            mask_nei = tf.sequence_mask(K.reshape(num_nbs, [-1]), self.max_nb, dtype=tf.float32)
            target_shape = tf.concat([tf.shape(num_nbs), [self.max_nb, 1]], 0)
            mask_nei = tf.reshape(mask_nei, target_shape)
            mask_nei.set_shape([None, None, self.max_nb, 1])
            l_nei = K.concatenate([fatom_nei, fbond_nei], axis=3) #(batch, #atoms, max_nb, )
            l_nei.set_shape([None,None,self.max_nb, self.hidden_size + 5])
            pre_label = self.label_U2(l_nei)
            nei_label = K.sum(pre_label * mask_nei, axis=-2, keepdims=False)
            new_label = K.concatenate([atom_features, nei_label], axis=2)
            atom_features = self.label_U1(new_label)

        return K.sum(atom_features, axis=-2, keepdims=False)


class Global_Attention(tf.keras.layers.Layer):
    '''
    A Keras class for implementation global attention mechanism

    Init
    hidden_size: The hidden size of the dense layers

    Inputs
    inputs: Tensor of (batch, #atoms, hidden) usually from embedding of WLN_Layer
    bin_features: binary features of atom-atom relationsips from function get_bin_feature in ioutils_direct.py

    Output
    attention score: the global attention scores for all atom pairs
    '''

    def __init__(self, hidden_size):
        super(Global_Attention, self).__init__()
        self.hidden_size = hidden_size

    def build(self, input_shape):
        self.att_atom_feature = layers.Dense(self.hidden_size, kernel_initializer='random_uniform', use_bias=False, input_shape=(self.hidden_size,))
        self.att_bin_feature = layers.Dense(self.hidden_size, kernel_initializer='random_uniform')
        self.att_score = layers.Dense(1, activation=K.sigmoid, kernel_initializer='random_uniform')
        self.reshape1 = layers.Reshape((1,-1,self.hidden_size))
        self.reshape2 = layers.Reshape((-1,1,self.hidden_size))
        super(Global_Attention, self).build(input_shape)

    def call(self, inputs, bin_features):
        atom_hiddens1 = self.reshape1(inputs)
        atom_hiddens2 = self.reshape2(inputs)
        atom_pair = atom_hiddens1 + atom_hiddens2
        att_hidden = K.relu(self.att_atom_feature(atom_pair) + self.att_bin_feature(bin_features))
        att_score = self.att_score(att_hidden)
        att_context = att_score * atom_hiddens1
        return K.sum(att_context, axis=2, keepdims=False), atom_pair
