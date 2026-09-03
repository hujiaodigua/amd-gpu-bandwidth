#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define DEFAULT_SIZE_MB 256
#define ITERATIONS      5

static const char *kernels_src =
    "__kernel void copy_k(__global const float *__restrict a,\n"
    "                     __global float *__restrict b) {\n"
    "    size_t i = get_global_id(0);\n"
    "    b[i] = a[i];\n"
    "}\n"
    "__kernel void scale_k(__global const float *__restrict a,\n"
    "                      __global float *__restrict b, float s) {\n"
    "    size_t i = get_global_id(0);\n"
    "    b[i] = s * a[i];\n"
    "}\n"
    "__kernel void add_k(__global const float *__restrict a,\n"
    "                    __global const float *__restrict b,\n"
    "                    __global float *__restrict c) {\n"
    "    size_t i = get_global_id(0);\n"
    "    c[i] = a[i] + b[i];\n"
    "}\n"
    "__kernel void triad_k(__global float *__restrict a,\n"
    "                      __global const float *__restrict b,\n"
    "                      __global const float *__restrict c, float s) {\n"
    "    size_t i = get_global_id(0);\n"
    "    a[i] = b[i] + s * c[i];\n"
    "}\n";

static void die(const char *msg, cl_int err) {
    fprintf(stderr, "%s (error=%d)\n", msg, err);
    exit(1);
}

static double elapsed_s(cl_event ev) {
    cl_ulong start = 0, end = 0;
    clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START,
                            sizeof(start), &start, NULL);
    clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END,
                            sizeof(end), &end, NULL);
    return (double)(end - start) * 1e-9;
}

static char *dev_str(cl_device_id d, cl_device_info p) {
    size_t sz = 0;
    clGetDeviceInfo(d, p, 0, NULL, &sz);
    char *s = (char *)malloc(sz);
    clGetDeviceInfo(d, p, sz, s, NULL);
    return s;
}

static cl_ulong dev_ulong(cl_device_id d, cl_device_info p) {
    cl_ulong v = 0;
    clGetDeviceInfo(d, p, sizeof(v), &v, NULL);
    return v;
}

static double gbs(double bytes, double seconds) {
    return bytes / seconds / 1e9;
}

static void report(const char *label, double bytes_moved, const double *times, int n) {
    double best = times[0], sum = 0.0;
    for (int i = 0; i < n; i++) {
        if (times[i] < best) best = times[i];
        sum += times[i];
    }
    double avg = sum / n;
    printf("  %-14s best %8.2f GB/s   avg %8.2f GB/s\n",
           label, gbs(bytes_moved, best), gbs(bytes_moved, avg));
}

/* --- VRAM: device-to-device copy --- */
static void bench_d2d_copy(cl_command_queue q, cl_mem a, cl_mem b, size_t bytes) {
    cl_int err;
    cl_event ev;
    double times[ITERATIONS];

    err = clEnqueueCopyBuffer(q, a, b, 0, 0, bytes, 0, NULL, &ev);
    if (err != CL_SUCCESS) die("clEnqueueCopyBuffer (warmup)", err);
    clWaitForEvents(1, &ev);
    clReleaseEvent(ev);

    for (int i = 0; i < ITERATIONS; i++) {
        err = clEnqueueCopyBuffer(q, a, b, 0, 0, bytes, 0, NULL, &ev);
        if (err != CL_SUCCESS) die("clEnqueueCopyBuffer", err);
        clWaitForEvents(1, &ev);
        times[i] = elapsed_s(ev);
        clReleaseEvent(ev);
    }
    /* copy reads + writes => move 2x bytes */
    report("D2D copy", 2.0 * bytes, times, ITERATIONS);
}

/* --- VRAM: STREAM kernels --- */
static void bench_stream(cl_command_queue q, cl_kernel k, size_t n_floats,
                         double mult, const char *label) {
    cl_int err;
    cl_event ev;
    double times[ITERATIONS];
    size_t gws = n_floats;

    err = clEnqueueNDRangeKernel(q, k, 1, NULL, &gws, NULL, 0, NULL, &ev);
    if (err != CL_SUCCESS) die("clEnqueueNDRangeKernel (warmup)", err);
    clWaitForEvents(1, &ev);
    clReleaseEvent(ev);

    for (int i = 0; i < ITERATIONS; i++) {
        err = clEnqueueNDRangeKernel(q, k, 1, NULL, &gws, NULL, 0, NULL, &ev);
        if (err != CL_SUCCESS) die("clEnqueueNDRangeKernel", err);
        clWaitForEvents(1, &ev);
        times[i] = elapsed_s(ev);
        clReleaseEvent(ev);
    }
    report(label, mult * (double)n_floats * sizeof(float), times, ITERATIONS);
}

/* --- PCIe: host<->device using pinned (mapped) host memory --- */
static void bench_h2d(cl_command_queue q, cl_mem d_buf, void *h_ptr, size_t bytes) {
    cl_int err;
    cl_event ev;
    double times[ITERATIONS];

    err = clEnqueueWriteBuffer(q, d_buf, CL_FALSE, 0, bytes, h_ptr, 0, NULL, &ev);
    if (err != CL_SUCCESS) die("clEnqueueWriteBuffer (warmup)", err);
    clWaitForEvents(1, &ev);
    clReleaseEvent(ev);

    for (int i = 0; i < ITERATIONS; i++) {
        err = clEnqueueWriteBuffer(q, d_buf, CL_FALSE, 0, bytes, h_ptr, 0, NULL, &ev);
        if (err != CL_SUCCESS) die("clEnqueueWriteBuffer", err);
        clWaitForEvents(1, &ev);
        times[i] = elapsed_s(ev);
        clReleaseEvent(ev);
    }
    report("H2D (write)", (double)bytes, times, ITERATIONS);
}

static void bench_d2h(cl_command_queue q, cl_mem d_buf, void *h_ptr, size_t bytes) {
    cl_int err;
    cl_event ev;
    double times[ITERATIONS];

    err = clEnqueueReadBuffer(q, d_buf, CL_FALSE, 0, bytes, h_ptr, 0, NULL, &ev);
    if (err != CL_SUCCESS) die("clEnqueueReadBuffer (warmup)", err);
    clWaitForEvents(1, &ev);
    clReleaseEvent(ev);

    for (int i = 0; i < ITERATIONS; i++) {
        err = clEnqueueReadBuffer(q, d_buf, CL_FALSE, 0, bytes, h_ptr, 0, NULL, &ev);
        if (err != CL_SUCCESS) die("clEnqueueReadBuffer", err);
        clWaitForEvents(1, &ev);
        times[i] = elapsed_s(ev);
        clReleaseEvent(ev);
    }
    report("D2H (read)", (double)bytes, times, ITERATIONS);
}

int main(int argc, char **argv) {
    cl_int err;
    cl_uint nplat = 0;
    size_t requested = (size_t)DEFAULT_SIZE_MB * 1024 * 1024;

    if (argc > 1) {
        long mb = strtol(argv[1], NULL, 10);
        if (mb <= 0) {
            fprintf(stderr, "usage: %s [size_mb]\n", argv[0]);
            return 1;
        }
        requested = (size_t)mb * 1024 * 1024;
    }

    err = clGetPlatformIDs(0, NULL, &nplat);
    if (err != CL_SUCCESS || nplat == 0) die("no OpenCL platform", err);

    cl_platform_id *plats = (cl_platform_id *)malloc(nplat * sizeof(cl_platform_id));
    clGetPlatformIDs(nplat, plats, NULL);

    cl_device_id dev = NULL;
    cl_uint nde = 0;
    for (cl_uint i = 0; i < nplat && !dev; i++) {
        clGetDeviceIDs(plats[i], CL_DEVICE_TYPE_GPU, 0, NULL, &nde);
        if (nde > 0) {
            cl_device_id *devs = (cl_device_id *)malloc(nde * sizeof(cl_device_id));
            clGetDeviceIDs(plats[i], CL_DEVICE_TYPE_GPU, nde, devs, NULL);
            dev = devs[0];
            free(devs);
        }
    }
    if (!dev) die("no GPU device found", CL_DEVICE_NOT_FOUND);

    char *name = dev_str(dev, CL_DEVICE_NAME);
    char *vendor = dev_str(dev, CL_DEVICE_VENDOR);
    char *ver = dev_str(dev, CL_DEVICE_VERSION);
    cl_ulong global_mem = dev_ulong(dev, CL_DEVICE_GLOBAL_MEM_SIZE);
    cl_ulong max_alloc = dev_ulong(dev, CL_DEVICE_MAX_MEM_ALLOC_SIZE);

    printf("Device: %s (%s)  OpenCL %s\n", name, vendor, ver);
    printf("Global mem: %.1f MB   Max alloc: %.1f MB\n",
           global_mem / 1048576.0, max_alloc / 1048576.0);

    /* clamp sizes so buffers fit on-device */
    size_t max_safe = (size_t)(max_alloc / 3);           /* fits 3 stream buffers */
    if (max_safe > global_mem / 4) max_safe = global_mem / 4;
    if (requested > max_safe) {
        printf("clamping %zu MB -> %zu MB (max safe for 3 buffers)\n",
               requested >> 20, max_safe >> 20);
        requested = max_safe;
    }
    size_t pcie_bytes = requested;
    size_t stream_bytes = requested;

    cl_context ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err);
    if (err != CL_SUCCESS) die("clCreateContext", err);

    cl_command_queue q = clCreateCommandQueue(ctx, dev, CL_QUEUE_PROFILING_ENABLE, &err);
    if (err != CL_SUCCESS) die("clCreateCommandQueue", err);

    cl_program prog = clCreateProgramWithSource(ctx, 1, &kernels_src, NULL, &err);
    if (err != CL_SUCCESS) die("clCreateProgramWithSource", err);
    err = clBuildProgram(prog, 1, &dev, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t logsz;
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &logsz);
        char *log = (char *)malloc(logsz);
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, logsz, log, NULL);
        fprintf(stderr, "build log:\n%s\n", log);
        free(log);
        die("clBuildProgram", err);
    }

    cl_kernel copy_k = clCreateKernel(prog, "copy_k", &err);
    cl_kernel scale_k = clCreateKernel(prog, "scale_k", &err);
    cl_kernel add_k = clCreateKernel(prog, "add_k", &err);
    cl_kernel triad_k = clCreateKernel(prog, "triad_k", &err);
    if (!copy_k || !scale_k || !add_k || !triad_k) die("clCreateKernel", err);

    /* stream buffers */
    cl_mem sA = clCreateBuffer(ctx, CL_MEM_READ_WRITE, stream_bytes, NULL, &err);
    cl_mem sB = clCreateBuffer(ctx, CL_MEM_READ_WRITE, stream_bytes, NULL, &err);
    cl_mem sC = clCreateBuffer(ctx, CL_MEM_READ_WRITE, stream_bytes, NULL, &err);
    if (err != CL_SUCCESS) die("clCreateBuffer (stream)", err);

    cl_float one = 1.0f;
    clEnqueueFillBuffer(q, sA, &one, sizeof(one), 0, stream_bytes, 0, NULL, NULL);
    clEnqueueFillBuffer(q, sB, &one, sizeof(one), 0, stream_bytes, 0, NULL, NULL);
    clEnqueueFillBuffer(q, sC, &one, sizeof(one), 0, stream_bytes, 0, NULL, NULL);
    clFinish(q);

    size_t n_floats = stream_bytes / sizeof(float);

    clSetKernelArg(copy_k, 0, sizeof(cl_mem), &sA);
    clSetKernelArg(copy_k, 1, sizeof(cl_mem), &sB);

    clSetKernelArg(scale_k, 0, sizeof(cl_mem), &sA);
    clSetKernelArg(scale_k, 1, sizeof(cl_mem), &sB);
    clSetKernelArg(scale_k, 2, sizeof(float), &one);

    clSetKernelArg(add_k, 0, sizeof(cl_mem), &sA);
    clSetKernelArg(add_k, 1, sizeof(cl_mem), &sB);
    clSetKernelArg(add_k, 2, sizeof(cl_mem), &sC);

    clSetKernelArg(triad_k, 0, sizeof(cl_mem), &sA);
    clSetKernelArg(triad_k, 1, sizeof(cl_mem), &sB);
    clSetKernelArg(triad_k, 2, sizeof(cl_mem), &sC);
    clSetKernelArg(triad_k, 3, sizeof(float), &one);

    /* PCIe buffers: device buffer + pinned host buffer */
    cl_mem d_buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, pcie_bytes, NULL, &err);
    cl_mem h_buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                  pcie_bytes, NULL, &err);
    if (err != CL_SUCCESS) die("clCreateBuffer (pcie)", err);

    void *h_ptr = clEnqueueMapBuffer(q, h_buf, CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION,
                                     0, pcie_bytes, 0, NULL, NULL, &err);
    if (err != CL_SUCCESS) die("clEnqueueMapBuffer", err);
    memset(h_ptr, 0xAB, pcie_bytes);

    clEnqueueFillBuffer(q, d_buf, &one, sizeof(one), 0, pcie_bytes, 0, NULL, NULL);
    clFinish(q);

    printf("\n== VRAM bandwidth (buffer=%zu MB) ==\n", stream_bytes >> 20);
    bench_d2d_copy(q, sA, sB, stream_bytes);
    bench_stream(q, copy_k, n_floats, 2.0, "STREAM copy");
    bench_stream(q, scale_k, n_floats, 2.0, "STREAM scale");
    bench_stream(q, add_k, n_floats, 3.0, "STREAM add");
    bench_stream(q, triad_k, n_floats, 3.0, "STREAM triad");

    printf("\n== PCIe bandwidth (buffer=%zu MB) ==\n", pcie_bytes >> 20);
    bench_h2d(q, d_buf, h_ptr, pcie_bytes);
    bench_d2h(q, d_buf, h_ptr, pcie_bytes);

    clFinish(q);

    free(name); free(vendor); free(ver); free(plats);
    return 0;
}
