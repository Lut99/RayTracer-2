/* CPUGPUHYBRID.cpp
 *   by Lut99
 *
 * Created:
 *   7/4/2020, 4:55:23 PM
 * Last edited:
 *   7/4/2020, 5:20:10 PM
 * Auto updated?
 *   Yes
 *
 * Description:
 *   This file allows me to test how it works to use CUDA's __host__
 *   __device__ function.
**/

#include <iostream>

using namespace std;


#ifdef __CUDACC__
#define CUDAPREFIX __host__ __device__
#else
#define CUDAPREFIX
#endif


/* Test class which will be hybrid, and copyable to the GPU. */
class TestClass {
protected:
    size_t size;
    double* data;
    bool is_local;

    CUDAPREFIX TestClass() {
        // Does nothing
    }

public:
    TestClass(size_t n) {
        this->size = n;
        this->data = new double[n];
        this->is_local = true;
    }

    CUDAPREFIX TestClass(const TestClass& other) {
        this->size = other.size;
        this->is_local = true;
        this->data = new double[this->size];
        for (size_t i = 0; i < this->size; i++) {
            this->data[i] = other.data[i];
        }
    }

    CUDAPREFIX ~TestClass() {
        delete[] this->data;
    }

    CUDAPREFIX double& operator[](size_t index) {
        return this->data[index];
    }

    CUDAPREFIX double sum() const {
        double total = 0;
        for (size_t i = 0; i < this->size; i++) {
            total += this->data[i];
        }
        return total;
    }

    friend class GTestClass;
};


/* Test class which handles the GPU side. */
class GTestClass : public TestClass {
public:
    __device__ GTestClass(size_t n, double* data)
    {
        this->size = n;
        this->data = data;
        this->is_local = false;
    }

    static void* toGPU(const TestClass& test) {
        void* ptr;
        cudaMalloc(&ptr, sizeof(double) * test.size);
        cudaMemcpy(ptr, (void*) test.data, sizeof(double) * test.size, cudaMemcpyHostToDevice);
        return ptr;
    }

    static TestClass fromGPU(size_t size, void* ptr) {
        // Allocate memory
        TestClass result(size);
        // Copy back
        cudaMemcpy((void*) result.data, ptr, sizeof(double) * size, cudaMemcpyDeviceToHost);
        cudaFree(ptr);
        return result;
    }
};


__global__ void test_kernel(size_t size, void* test_ptr) {
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i == 0) {
        GTestClass test(size, (double*) test_ptr);

        printf("GPU: %f\n", test.sum());
    }
}


int main() {
    // Create a test class and fill it with some data
    size_t n = 100;
    TestClass test(n);
    for (size_t i = 0; i < n; i++) {
        test[i] = i;
    }

    cout << "CPU: " << test.sum() << endl;

    // Copy to the GPU
    void* test_ptr = GTestClass::toGPU(test);

    // Run the kernel
    test_kernel<<<1, 32>>>(n, test_ptr);
    cudaDeviceSynchronize();

    // Copy back to the CPU
    TestClass result = GTestClass::fromGPU(n, test_ptr);

    // Test if the same
    for (size_t i = 0; i < n; i++) {
        if (test[i] != result[i]) {
            cerr << "Mismatch @ " << i << ": " << test[i] << " is not " << result[i] << "." << endl;
            return EXIT_FAILURE;
        }
    }

    return EXIT_SUCCESS;
}
