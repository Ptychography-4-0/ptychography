import cuBinding
import numpy as np

def main():
    array_size = 12
    a = cuBinding.CuAlgoFI(array_size)
    b = cuBinding.CuUmemAlgoFI(array_size)

    print("test CuAlgoFI")
    if(test(a)):
        print("test passed")

    print("\ntest CuUmemAlgoFI")
    if(test(b)):
        print("test passed")


def test(algoObj) -> bool:
    """
    return true, if the result is correct
    """

    # allocate memory
    algoObj.init()

    # return a numpy array object, which is view of the memory, which managed by
    # the C++ class
    input = algoObj.get_input_view()
    output = algoObj.get_output_view()

    for k in range(algoObj.get_size()):
        input[k] = k

    algoObj.compute()

    correct = compare_result(output)

    # release memory
    algoObj.deinit()

    return correct

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
