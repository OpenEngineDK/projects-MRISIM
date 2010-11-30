

__kernel void test1(__global float* input, __global float* output) {
    uint idx = get_global_id(0);
    output[idx] = input[idx] + 1;
}
