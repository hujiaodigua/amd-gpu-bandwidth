#include <vulkan/vulkan.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#define NITER 10
#define DEFAULT_MB 256

typedef struct {
    VkDevice device;
    VkQueue queue;
    uint32_t qfamily;
    VkCommandPool cmdpool;
    VkCommandBuffer cmd;
    VkPhysicalDeviceMemoryProperties memprops;
} Ctx;

typedef struct {
    VkBuffer buffer;
    VkDeviceMemory mem;
    VkDeviceSize size;
} Buf;

static void vk_check(VkResult r, const char *msg) {
    if (r != VK_SUCCESS) {
        fprintf(stderr, "%s: VkResult %d\n", msg, r);
        exit(1);
    }
}

static uint32_t find_mem_type(const VkPhysicalDeviceMemoryProperties *mp,
                              uint32_t type_bits, VkMemoryPropertyFlags props) {
    for (uint32_t i = 0; i < mp->memoryTypeCount; i++) {
        if ((type_bits & (1u << i)) &&
            (mp->memoryTypes[i].propertyFlags & props) == props)
            return i;
    }
    return UINT32_MAX;
}

static uint64_t device_local_heap_size(const VkPhysicalDeviceMemoryProperties *mp) {
    uint64_t max = 0;
    for (uint32_t i = 0; i < mp->memoryHeapCount; i++) {
        if (mp->memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
            if (mp->memoryHeaps[i].size > max)
                max = mp->memoryHeaps[i].size;
    }
    return max;
}

static void create_buffer(Ctx *c, VkDeviceSize size,
                          VkBufferUsageFlags usage, VkMemoryPropertyFlags props,
                          Buf *out) {
    VkBufferCreateInfo bci;
    memset(&bci, 0, sizeof(bci));
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = size;
    bci.usage = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vk_check(vkCreateBuffer(c->device, &bci, NULL, &out->buffer), "vkCreateBuffer");

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(c->device, out->buffer, &req);
    uint32_t mti = find_mem_type(&c->memprops, req.memoryTypeBits, props);
    if (mti == UINT32_MAX) {
        fprintf(stderr, "no matching memory type for props 0x%x\n", props);
        exit(1);
    }
    VkMemoryAllocateInfo mai;
    memset(&mai, 0, sizeof(mai));
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = mti;
    vk_check(vkAllocateMemory(c->device, &mai, NULL, &out->mem), "vkAllocateMemory");
    vk_check(vkBindBufferMemory(c->device, out->buffer, out->mem, 0), "vkBindBufferMemory");
    out->size = size;
}

static void destroy_buffer(Ctx *c, Buf *b) {
    vkDestroyBuffer(c->device, b->buffer, NULL);
    vkFreeMemory(c->device, b->mem, NULL);
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void bench_copy(Ctx *c, Buf *src, Buf *dst, VkDeviceSize size,
                       double eff_bytes, const char *label) {
    int K = (int)(512 * 1024 * 1024 / size);
    if (K < 1) K = 1;
    if (K > 64) K = 64;

    vk_check(vkResetCommandPool(c->device, c->cmdpool, 0), "vkResetCommandPool");

    VkCommandBufferAllocateInfo cai;
    memset(&cai, 0, sizeof(cai));
    cai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cai.commandPool = c->cmdpool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    vk_check(vkAllocateCommandBuffers(c->device, &cai, &cmd), "vkAllocateCommandBuffers");

    VkCommandBufferBeginInfo bi;
    memset(&bi, 0, sizeof(bi));
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vk_check(vkBeginCommandBuffer(cmd, &bi), "vkBeginCommandBuffer");

    VkBufferCopy region;
    region.srcOffset = 0;
    region.dstOffset = 0;
    region.size = size;
    for (int i = 0; i < K; i++)
        vkCmdCopyBuffer(cmd, src->buffer, dst->buffer, 1, &region);

    vk_check(vkEndCommandBuffer(cmd), "vkEndCommandBuffer");

    VkSubmitInfo si;
    memset(&si, 0, sizeof(si));
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;

    vk_check(vkQueueSubmit(c->queue, 1, &si, VK_NULL_HANDLE), "vkQueueSubmit (warmup)");
    vk_check(vkQueueWaitIdle(c->queue), "vkQueueWaitIdle (warmup)");

    double times[NITER];
    for (int i = 0; i < NITER; i++) {
        double t0 = now_s();
        vk_check(vkQueueSubmit(c->queue, 1, &si, VK_NULL_HANDLE), "vkQueueSubmit");
        vk_check(vkQueueWaitIdle(c->queue), "vkQueueWaitIdle");
        times[i] = now_s() - t0;
    }

    double best = times[0], sum = 0.0;
    for (int i = 0; i < NITER; i++) {
        if (times[i] < best) best = times[i];
        sum += times[i];
    }
    double total = eff_bytes * K;
    printf("  %-16s best %8.2f GB/s   avg %8.2f GB/s\n",
           label, total / best / 1e9, total / (sum / NITER) / 1e9);
}

static void bench_device(VkPhysicalDevice pd, VkDeviceSize req_size) {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(pd, &props);
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);

    uint64_t heap = device_local_heap_size(&mp);
    VkDeviceSize size = req_size;
    VkDeviceSize max_safe = heap / 4; /* two device-local buffers live at once */
    if (size > max_safe) size = max_safe;

    printf("\n== %s ==\n", props.deviceName);
    printf("  driverVersion %u.%u.%u   device-local heap %.1f MB\n",
           VK_VERSION_MAJOR(props.driverVersion),
           VK_VERSION_MINOR(props.driverVersion),
           VK_VERSION_PATCH(props.driverVersion),
           heap / 1048576.0);

    /* queue family: prefer transfer-only, else any transfer-capable */
    uint32_t qcount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qcount, NULL);
    VkQueueFamilyProperties *qprops = malloc(qcount * sizeof(*qprops));
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qcount, qprops);

    uint32_t qf = UINT32_MAX;
    for (uint32_t i = 0; i < qcount; i++)
        if (qprops[i].queueFlags & VK_QUEUE_TRANSFER_BIT) { qf = i; break; }
    if (qf == UINT32_MAX)
        for (uint32_t i = 0; i < qcount; i++)
            if (qprops[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) { qf = i; break; }
    free(qprops);
    if (qf == UINT32_MAX) { fprintf(stderr, "no usable queue\n"); return; }

    float priority = 1.0f;
    VkDeviceQueueCreateInfo dqci;
    memset(&dqci, 0, sizeof(dqci));
    dqci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    dqci.queueFamilyIndex = qf;
    dqci.queueCount = 1;
    dqci.pQueuePriorities = &priority;

    VkDeviceCreateInfo dci;
    memset(&dci, 0, sizeof(dci));
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &dqci;

    Ctx c;
    memset(&c, 0, sizeof(c));
    c.memprops = mp;
    c.qfamily = qf;
    vk_check(vkCreateDevice(pd, &dci, NULL, &c.device), "vkCreateDevice");
    vkGetDeviceQueue(c.device, qf, 0, &c.queue);

    VkCommandPoolCreateInfo cpci;
    memset(&cpci, 0, sizeof(cpci));
    cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpci.queueFamilyIndex = qf;
    vk_check(vkCreateCommandPool(c.device, &cpci, NULL, &c.cmdpool), "vkCreateCommandPool");

    printf("  buffer size %zu MB\n", (size_t)(size >> 20));

    /* VRAM D2D copy */
    Buf a, b;
    create_buffer(&c, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &a);
    create_buffer(&c, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &b);

    printf("\n  -- VRAM bandwidth --\n");
    bench_copy(&c, &a, &b, size, 2.0 * size, "D2D copy");

    destroy_buffer(&c, &a);
    destroy_buffer(&c, &b);

    /* PCIe H2D / D2H */
    Buf dev, host;
    create_buffer(&c, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &dev);
    create_buffer(&c, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &host);

    void *mapped;
    vk_check(vkMapMemory(c.device, host.mem, 0, VK_WHOLE_SIZE, 0, &mapped), "vkMapMemory");
    memset(mapped, 0xAB, size);

    printf("\n  -- PCIe bandwidth --\n");
    bench_copy(&c, &host, &dev, size, (double)size, "H2D (write)");
    bench_copy(&c, &dev, &host, size, (double)size, "D2H (read)");

    vkUnmapMemory(c.device, host.mem);
    destroy_buffer(&c, &dev);
    destroy_buffer(&c, &host);

    vkDestroyCommandPool(c.device, c.cmdpool, NULL);
    vkDestroyDevice(c.device, NULL);
}

static void usage(const char *prog) {
    printf(
        "Usage: %s [options]\n"
        "\n"
        "Measure GPU VRAM and PCIe bandwidth using Vulkan (vkCmdCopyBuffer).\n"
        "\n"
        "Options:\n"
        "  -h, --help          show this help and exit\n"
        "  -s, --size <mb>     buffer size in MB (default %d, clamped to 1/4 of VRAM)\n"
        "  -d, --device <str>  only test devices whose name contains <str>\n"
        "  -l, --list          list discrete GPUs and exit\n"
        "\n"
        "Examples:\n"
        "  %s                    test all discrete GPUs at default size\n"
        "  %s -s 512             use 512 MB buffers\n"
        "  %s -d 5300            only test the RX 5300\n"
        "  %s -l                 list GPUs without running benchmarks\n",
        prog, DEFAULT_MB, prog, prog, prog, prog);
}

static void list_devices(VkInstance inst) {
    uint32_t nd = 0;
    vkEnumeratePhysicalDevices(inst, &nd, NULL);
    VkPhysicalDevice *pds = malloc(nd * sizeof(*pds));
    vkEnumeratePhysicalDevices(inst, &nd, pds);
    for (uint32_t i = 0; i < nd; i++) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(pds[i], &p);
        if (p.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) continue;
        VkPhysicalDeviceMemoryProperties mp;
        vkGetPhysicalDeviceMemoryProperties(pds[i], &mp);
        uint64_t heap = device_local_heap_size(&mp);
        printf("  %s  (device-local heap %.1f MB)\n", p.deviceName, heap / 1048576.0);
    }
    free(pds);
}

int main(int argc, char **argv) {
    VkDeviceSize req = (VkDeviceSize)DEFAULT_MB * 1024 * 1024;
    const char *name_filter = NULL;
    int list_only = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            usage(argv[0]);
            return 0;
        } else if (!strcmp(argv[i], "-l") || !strcmp(argv[i], "--list")) {
            list_only = 1;
        } else if (!strcmp(argv[i], "-s") || !strcmp(argv[i], "--size")) {
            if (i + 1 >= argc) { usage(argv[0]); return 1; }
            long mb = strtol(argv[++i], NULL, 10);
            if (mb <= 0) { fprintf(stderr, "invalid size: %s\n", argv[i]); return 1; }
            req = (VkDeviceSize)mb * 1024 * 1024;
        } else if (!strcmp(argv[i], "-d") || !strcmp(argv[i], "--device")) {
            if (i + 1 >= argc) { usage(argv[0]); return 1; }
            name_filter = argv[++i];
        } else {
            fprintf(stderr, "unknown option: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    VkApplicationInfo ai;
    memset(&ai, 0, sizeof(ai));
    ai.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    ai.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo ici;
    memset(&ici, 0, sizeof(ici));
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &ai;

    VkInstance inst;
    vk_check(vkCreateInstance(&ici, NULL, &inst), "vkCreateInstance");

    if (list_only) {
        list_devices(inst);
        vkDestroyInstance(inst, NULL);
        return 0;
    }

    uint32_t nd = 0;
    vk_check(vkEnumeratePhysicalDevices(inst, &nd, NULL), "vkEnumeratePhysicalDevices");
    VkPhysicalDevice *pds = malloc(nd * sizeof(*pds));
    vkEnumeratePhysicalDevices(inst, &nd, pds);

    for (uint32_t i = 0; i < nd; i++) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(pds[i], &p);
        if (p.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) continue;
        if (name_filter && !strstr(p.deviceName, name_filter)) continue;
        bench_device(pds[i], req);
    }

    free(pds);
    vkDestroyInstance(inst, NULL);
    return 0;
}
