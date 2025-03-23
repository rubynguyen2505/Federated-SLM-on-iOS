import numpy as np

def aggregate_weights(weights_list):
    return np.mean(weights_list, axis=0)

if __name__ == "__main__":
    test_weights = [np.random.rand(10), np.random.rand(10)]
    print(aggregate_weights(test_weights))
