/* GPUCOPY.cu
 *   by Lut99
 *
 * Created:
 *   07/07/2020, 16:49:06
 * Last edited:
 *   07/07/2020, 17:02:08
 * Auto updated?
 *   Yes
 *
 * Description:
 *   This file allows us to test if we can copy nested structs easily or if
 *   it is going to require some extreme level wizardry.
**/

#include <iostream>

using namespace std;

#define TOSTR(S) _TOSTR(S)
#define _TOSTR(S) #S
#define CUDA_ASSERT(ID) \
    if (cudaPeekAtLastError() != cudaSuccess) { \
        cout << "[FAIL]" << endl << endl; \
        cerr << "ERROR: " TOSTR(ID) ": " << cudaGetErrorString(cudaGetLastError()) << endl << endl; \
        return false; \
    }


/* Test struct. */
struct TestStruct {
    int a;
    int b;

    TestStruct* GPU_create(void* ptr) {
        cudaMemcpy(ptr, &(this->a), sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy((void*) ((int*) ptr + 1), &(this->b), sizeof(int), cudaMemcpyHostToDevice);
        return (TestStruct*) ptr;
    }

    static TestStruct GPU_copy(TestStruct* ptr) {
        int a, b;
        cudaMemcpy(&a, ptr, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&b, (void*) ((int*) ptr + 1), sizeof(int), cudaMemcpyDeviceToHost);
        return TestStruct({a, b});
    }
};


/* Test class with a nested teststruct. */
class TestClass {
public:
    TestStruct test;

    TestClass(int a, int b) :
        test({a, b})
    {}

    __host__ __device__ void print() const {
        printf("TestClass : {a = %d, b = %d}\n", this->test.a, this->test.b);
    }

    static TestClass* GPU_create(const TestClass& other) {
        // Create space for the testclass on the GPU
        TestClass* ptr;
        cudaMalloc((void**) &ptr, sizeof(TestClass));

        // Inject the struct
        other.test.GPU_copy((TestStruct*) ptr);

        // Return the pointer
        return ptr;
    }

    static TestClass GPU_copy(TestClass* ptr) {
        // Create a new TestClass
        TestClass result(-1, -1);

        // Copy the struct back
        result.test = TestStruct::GPU_copy((TestStruct*) ptr);

        // Return the class
        return result;
    }

    static void GPU_free(TestClass* ptr) {
        cudaFree(ptr);
    }
};


/* Kernel which we test with. */
__global__ void testKernel(TestClass* test) {
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i == 0) {
        // Print it lmao
        test->print();
    }
}


int main() {
    // Create a test class
    TestClass test(50, 100);

    // Print it
    test.print();

    // Next, copy it to the GPU
    TestClass* test_gpu = TestClass::GPU_create(test);

    // Run the kernel with that pointer
    testKernel<<<1, 32>>>(test_gpu);
    cudaDeviceSynchronize();
    CUDA_ASSERT(testKernel);

    // Copy back & free
    TestClass result = TestClass::GPU_copy(test_gpu);
    TestClass::GPU_free(test_gpu);

    // Also print
    result.print();
    
    // Done
    return 0;
}

