import cppyy
import numpy as np

cppyy.include("wrapper.hpp")
cppyy.load_library("wrapper.so")

class AlgoFloat:
    def __init__(self, size : int):
        self.cpp_obj = cppyy.gbl.get_algo_float(size)

    def init(self):
        """
        allocate memory
        """
        return self.cpp_obj.init()

    def deinit(self):
        """
        deallocate memory
        """
        return self.cpp_obj.deinit()

    def get_size(self):
        return self.cpp_obj.get_size()

    def get_input(self) -> np.ndarray:
        """
        return a reference to the input buffer as numpy.ndarry
        """
        self.input = self.cpp_obj.get_input_memory()
        self.input.reshape((self.get_size(),))
        return np.frombuffer(self.input, dtype=np.float32, count=self.get_size())

    def get_output(self) -> np.ndarray:
        """
        return a reference to the output buffer as numpy.ndarry
        """
        self.output = self.cpp_obj.get_output_memory()
        self.output.reshape((self.get_size(),))
        return np.frombuffer(self.output, dtype=np.float32, count=self.get_size())

    def algo(self):
        """
        run a algorithm, which use the input buffer as source and the output buffer as destination
        """
        return self.cpp_obj.algo()

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

def test():
    size = 12
    a = AlgoFloat(size)
    a.init()

    input = a.get_input()
    output = a.get_output()

    for k in range(a.get_size()):
        input[k] = k

    a.algo()

    if(compare_result(output)):
        print('test passed')

    a.deinit()

if __name__ == '__main__':
    test()
