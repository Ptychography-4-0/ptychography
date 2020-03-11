import cppBinding
import numpy as np

def main():
    array_size = 12
    a = cppBinding.AlgoFI(array_size)

    # allocate memory
    a.init()

    # create numpy arrays to share memory between python and the C++ implementation
    # copy=False is really important, otherwise the memory is just copied
    input = np.array(a.get_input(), copy=False)
    output = np.array(a.get_output(), copy=False)

    for k in range(a.get_size()):
        input[k] = k

    a.compute()

    if(compare_result(output)):
        print('test passed')

    # release memory
    a.deinit()

def compare_result(result : np.ndarray) -> bool:
    """
    return true, if the result is correct
    """
    size = len(result)
    expected_res = np.zeros(size, dtype=np.float32)

    for k in range(size):
        expected_res[k] = k + (k%3)

    for k in range(size):
        if result[k] != expected_res[k]:
            print('test failed')
            print('expected: ' + str(expected_res))
            print('result: ' + str(result))
            return False

    return True


if __name__ == "__main__":
    main()
