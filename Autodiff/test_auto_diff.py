import numpy as np
from autodiff import dot, relu, mul, add, exp, sqrt, squared_loss
from autodiff import evaluate_chain, forward_diff_chain, backward_diff_chain
from autodiff import Node, evaluate_dag, backward_diff_dag, topological_sort

def test_dag():
    x7 = create_dag([0.5, 1.3])
    sorted_nodes = topological_sort(x7)
    node_names = [node.name for node in sorted_nodes]
    names = ["x2", "x1", "x3", "x4", "x5", "x6", "x7"]
    assert_array_equal(node_names, names)

    value = evaluate_dag(sorted_nodes)

    def f(x):
        return x[1] * np.exp(x[0]) * np.sqrt(x[0] + x[1] * np.exp(x[0]))

    x = np.array([0.5, 1.3])
    value2 = f(x)
    assert_array_almost_equal(value, value2)

    num_jac = num_jacobian(f, x)
    backward_diff_dag(sorted_nodes)
    # x2 is before x1 in the topological order
    jac = np.concatenate([sorted_nodes[1].grad, sorted_nodes[0].grad])
    assert_array_almost_equal(num_jac, jac)

test_dag()