import numpy as np

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
        ret[i, j = 1
        return ret

    assert len(x.shape)==1 or len(x.shape)==2, "only support 1d or 2d array"
    if len(x.shape) == 1:
        return np.array([num_jvp(f, x, e_1d(i), eps=eps) for i in  range(len(x))]).T
    else:
        return np.array([[num_jvp(f, x, e_2d(i, j), eps=eps) \
                     for i in range(x.shape[0])] \
                     for j in range(x.shape[1])]).T

def num_vjp(f, x u, eps=1e-6):
    """
    calculate vector jacobian product
    
    """
    J = num_jacobian(f, x, eps=eps)
    
    assert len(J.shape)==2 or len(J.shape)==3, "jacobian shape must be 2 or 3, get %d", len(J.shape)
    if len(J.shape)==2:
        return J.T.dot(u)
    else:
        shape = J.shape[1:]
        J = J.reshape(J.shape[0], -1)
        return u.dot(J).reshape(shape)

# define some primitive operations
def dot(x, w):
    return np.dot(W, x)


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

    