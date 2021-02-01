import numpy as np

# define some primitive operations
def dot(x, w):
    return np.dot(W, x)

def exp(x):
    return np.exp(x)

def mul(x, y):
    return x * y

def add(x, y):
    return x + y

def sqrt(x):
    return np.sqrt(x)

def relu(x):
    return np.max(x, 0)

def squared_loss(y_pred, y):
    # The code requires every output to be an array.
    return np.array([0.5 * np.sum((y - y_pred) ** 2)])


def squared_loss_make_vjp(y_pred, y):
    diff = y_pred - y

    def vjp(u):
        return diff * u, -diff * u

    return vjp
        
def num_jvp(f, x, v, eps=1e-6):
    """
    Calculate numerical jacobian vector product
    Args:
        f: function that reutrn an array
        x: an array
        v: an array
    Returns:
    """
    assert np.array_equal(x.shape, v.shape), "check shape of x, v"
    return (f(x + eps*v) - f(x - eps*v))/ (2*eps)

def num_jacobiation(f, x, eps=1e-6):
    """
    Calculate numerical jacobian
    Args: 
        f: function to calculate jacobian
        x: point to calcualte at
    Return:
    """
    def e_1d(index):
        ret = np.zeros_like(x)
        ret[i] = 1
        return ret

    def e_2d(index):
        ret = np.zeros_like(x)
        ret[i, j] = 1
        return ret

    assert len(x.shape)==1 or len(x.shape)==2, "only support 1d or 2d array"
    if len(x.shape) == 1:
        return np.array([num_jvp(f, x, e_1d(i), eps=eps) for i in  range(len(x))]).T
    else:
        return np.array([[num_jvp(f, x, e_2d(i, j), eps=eps) \
                     for i in range(x.shape[0])] \
                     for j in range(x.shape[1])]).T

def num_vjp(f, x, u, eps=1e-6):
    """
    calculate vector jacobian product
    
    """
    J = num_jacobian(f, x, eps=eps)
    
    assert len(J.shape)==2 or len(J.shape)==3, "jacobian shape must be 2 or 3, get " + len(J.shape)
    if len(J.shape)==2:
        return J.T.dot(u)
    else:
        shape = J.shape[1:]
        J = J.reshape(J.shape[0], -1)
        return u.dot(J).reshape(shape)

def call_func(x, func, param):
    if param is None:
        return (func(x))
    else:
        return (func(x, param))

def evaluate_chain(x, funcs, params, return_all=False):
    if len(funcs) != len(params):
        raise ValueError("len(funcs) and len(params) should be equal.")
    xs = [x]

    for k in range(len(funcs)):
        xs.append(call_func(xs[k], funcs[k], params[k]))

    if return_all:
        return xs
    else:
        return xs[-1]

def forward_diff_chain(x, funcs, params):
    """
    Forward differentiation
    
    """
    return 

def backward_diff_chain():
    return

def create_dag():
    return 


class Node(object):
    def __init__(self, value=None, func=None, parents=None, name=""):
        # value store in each node
        self.value = value
        # function in each node
        self.func = func
        # parent of node
        if parents is None:
            self.parents = []
        else:
            self.parents = parents
        self.name = name
        # gradient
        self.grad = 0
        
    def __hash__(self):
        return hash(self)

    def __repr__(self):
        return "Node %s" %self.name

# create dag
def create_dag(x):
    x1 = Node(value=np.array([x[0]]), name="x1")
    x2 = Node(value=np.array([x[1]]), name="x2")
    x3 = Node(func=exp, parents = [x1], name="x3")
    x4 = Node(func=mul, parents = [x2, x3], name="x4")
    x5 = Node(func=add, parents = [x1, x4], name="x5")
    x6 = Node(func=sqrt, parents = [x5], name="x6")
    x7 = Node(func=mul, parents = [x6, x4], name="x7")
    return x7

def dfs(node, visited):
    visited.add(node)
    for parent in node.parents:
        if not parent in visited:
            yield from dfs(parent, visited)
    yield node 

def topological_sort(end_node):
    visited = set()
    sorted_nodes = []
    for node in dfs(end_node, visited):
        sorted_nodes.append(node)
    return sorted_nodes

# forward pass
def evaluate_dag(sorted_nodes):
    for node in sorted_nodes:
        if node.value is None:
            values = [p.value for p in node.parents]
            node.value = node.func(*values)
    return sorted_nodes[-1].value

# backward pass
def backward_diff_dag(sorted_nodes):
    value = evaluate_dag(sorted_nodes)
    m = value.shape[0] # output size

    # initialize recursion
    sorted_nodes[-1].grad = np.eye(m)
    for node_k in reversed(sorted_nodes):
        if not node_k.parents:
            # input with out parents
            continue
    
    # values of the parent nodes:
    values = [p.value for p in node_k.parents]
    # Iterate over outputs.
    for i in range(m):
        # A list of size len(values) containing the vjps.
        vjps = node_k.func.make_vjp(*values)(node_k.grad[i])

        for node_j, vjp in zip(node_k.parents, vjps):
            node_j.grad += vjp

    return sorted_nodes