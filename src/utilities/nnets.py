from torch import nn

_activation_functions_without_positional_arguments = {
    #Non-linear activations (weighted sum, nonlinearity) -> https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity
    'elu':nn.ELU(), 'hardshrink':nn.Hardshrink, 'hardsigmoid':nn.Hardsigmoid(), 'hardtanh':nn.Hardtanh(), \
    'hardswish':nn.Hardswish(), 'leakyrelu':nn.LeakyReLU(), 'logsigmoid':nn.LogSigmoid(), \
    'prelu':nn.PReLU(), 'relu':nn.ReLU(), \
    'relu6':nn.ReLU6(), 'rrelu':nn.RReLU(), 'selu':nn.SELU(), 'celu':nn.CELU(), 'gelu':nn.GELU(), \
    'sigmoid':nn.Sigmoid(), 'silu':nn.SiLU(), 'mish':nn.Mish(), 'softplus':nn.Softplus(), 'softshrink':nn.Softshrink(), \
    'softsign':nn.Softsign(), 'tanh':nn.Tanh(), 'tanhshrink':nn.Tanhshrink(), 'glu':nn.GLU(), \
    #Non-linear activations (other) -> https://pytorch.org/docs/stable/nn.html#non-linear-activations-other
    'softmin':nn.Softmin(), 'softmax':nn.Softmax(), 'softmax2d':nn.Softmax2d(), 'logsoftmax':nn.LogSoftmax() 
}

_activation_functions_with_positional_arguments_set = {'multiheadattention', 'threshold', 'adaptivelogsoftmaxwithloss'}

_linear_layers_set = {'linear', 'bilinear', 'identity', 'lazylinear'}


def _define_activation_function(activation_function:str, *args:list, **kwargs:dict):
    """ Given a lowered activation function name, return the corresponding activation function object from torch.nn.

    Args:
        activation_function (str): Lowered activation function name.

    Raises:
        ValueError: If the activation function does not exist.
        ValueError: If the activation function requires positional arguments, but none were passed.
        ValueError: If the activation function requires positional arguments, but they were not correctly given.

    Returns:
        torch.nn child class: An activation function object from torch.nn.
    """    
    if activation_function not in _activation_functions_with_positional_arguments_set and activation_function not in _activation_functions_without_positional_arguments.keys():
        raise ValueError(f'Activation function {activation_function} does not exist, check your spelling.')
    if activation_function in _activation_functions_with_positional_arguments_set:
        #With positional arguments
        if len(kwargs) == 0 and len(args) == 0: raise ValueError(f'Activation function {activation_function} requires positional arguments')
        if activation_function == 'multiheadattention' and ('embed_dim' not in kwargs or 'num_heads' not in kwargs) and len(args) != 2: raise ValueError(f'Activation function {activation_function} requires positional arguments embed_dim and num_heads')
        else:
            return nn.MultiheadAttention(*args, **kwargs)
        if activation_function == 'threshold' and ('threshold' not in kwargs or 'value' not in kwargs) and len(args) != 2: raise ValueError(f'Activation function {activation_function} requires positional arguments threshold and value')
        else:
            return nn.Threshold(*args, **kwargs)
        if activation_function == 'adaptivelogsoftmaxwithloss' and ('in_features' not in kwargs or 'n_classes' not in kwargs) and len(args) != 2: raise ValueError(f'Activation function {activation_function} requires positional arguments in_features and n_classes')
        else:
            return nn.AdaptiveLogSoftmaxWithLoss(*args, **kwargs)

    else: 
        #Without positional arguments. #TODO: Add non compulsory arguments to the remaining activation functions.
        return _activation_functions_without_positional_arguments[activation_function]


def _define_linear_layer(linear_layer:str, *args:list, **kwargs:dict):
    """ Given a lowered linear layer name, return the corresponding linear layer object from torch.nn.

    Args:
        linear_layer (str): Lowered linear layer name.

    Raises:
        ValueError: If the linear layer requires positional arguments, but they were not correctly given.

    Returns:
        torch.nn child class: A linear layer object from torch.nn.
    """    
    if linear_layer not in _linear_layers_set: raise ValueError(f'The module {linear_layer} is not a linear layer. Check https://pytorch.org/docs/stable/nn.html#linear-layers for the full list of supported linear layers.')
    if linear_layer == 'linear' and ('in_features' not in kwargs or 'out_features' not in kwargs) and (len(args) != 2):
        raise ValueError(f'Linear layer {linear_layer} requires positional arguments in_features and out_features')
    else:
        return nn.Linear(*args, **kwargs)
    if linear_layer == 'bilinear' and ('in1_features' not in kwargs or 'in2_features' not in kwargs or 'out_features' not in kwargs) and (len(args) != 3):
        raise ValueError(f'Linear layer {linear_layer} requires positional arguments in1_features, in2_features and out_features.')
    else:
        return nn.Bilinear(*args, **kwargs)
    if linear_layer == 'lazylinear' and ('out_features' not in kwargs) and (len(args) != 1): raise ValueError(f'Linear layer {linear_layer} requires positional argument out_features')
    else:
        return nn.LazyLinear(*args, **kwargs)
    if linear_layer == 'identity':
        return nn.Identity(*args, **kwargs)


def mlp_from_kwargs(**kwargs:dict) -> nn.Sequential:
    """ Create a Multi Layer Perceptron (also called ANN) from a dictionary.
    It only accepts linear layers.

    Args:
        kwargs (dict): The dictionary describing the neural network to be created.
        For example: 
        The dictionary {'linear_1':[5, 10], 'relu_1':[], 'linear_2':[10, 20], 'relu_2':[], 'linear_3':[20, 3], 'softmax_1':[]}
        would create the object 
        Sequential(
            (linear_1): Linear(in_features=5, out_features=10, bias=True)
            (relu_1): ReLU()
            (linear_2): Linear(in_features=10, out_features=20, bias=True)
            (relu_2): ReLU()
            (linear_3): Linear(in_features=20, out_features=3, bias=True)
            (softmax_1): Softmax(dim=None)
        )

    Returns:
        nn.Sequential: The neural network object.
    """    
    model = nn.Sequential()
    for module_id, module_parameters in kwargs.items():
        module_id_lowered = module_id.split('_')[0].lower()
        if not len(module_parameters) == 0 and isinstance(module_parameters[-1], dict):
            module_args = module_parameters[:-1]
            module_kwargs = module_parameters[-1]
        else:
            module_args = module_parameters
            module_kwargs = {}

        if module_id_lowered not in _linear_layers_set:
            #It is an activation function
            model.add_module(module_id, _define_activation_function(module_id_lowered, *module_args, **module_kwargs))
        else:
            #It is a linear layer
            model.add_module(module_id, _define_linear_layer(module_id_lowered, *module_args, **module_kwargs))

    return model
