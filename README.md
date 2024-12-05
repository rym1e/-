# 王楚哲的第二周学习报告 
- [ ] 未完成的任务{#nodone}
- [x] 已完成的任务 {#done}
- [ ] 碰到的问题{#questions}
# 一. 学习的项目
· 项目1 .学习了markdown的基础语法
-
- [x] 已完成的任务 {#done}

### **参考链接**[Markdown 语法大全详解](https://blog.csdn.net/w11111xxxl/article/details/140783343?ops_request_misc=%257B%2522request%255Fid%2522%253A%252265f09e652a4209b03eec91bfbceae0bd%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=65f09e652a4209b03eec91bfbceae0bd&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-2-140783343-null-null.142^v100^pc_search_result_base1&utm_term=markdown%E8%AF%AD%E6%B3%95&spm=1018.2226.3001.4187)

### 自己的学习笔记
### [markdown基础语法大全学习笔记.md](../../../../PycharmProjects/PythonProject11/markdown%E5%9F%BA%E7%A1%80%E8%AF%AD%E6%B3%95%E5%A4%A7%E5%85%A8%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0.md)




学习难度|知识点量|
-|-
**|****

--------

· 项目2 对异构计算进行调研
=======

- [x] 已完成的任务 {#done}

学习难度|知识点量|
-|-
*|****
### [对异构计算的调研笔记.docx](%E5%AF%B9%E5%BC%82%E6%9E%84%E8%AE%A1%E7%AE%97%E7%9A%84%E8%B0%83%E7%A0%94.docx)
### [CUDA学习入门](https://blog.csdn.net/weixin_44222088/article/details/135716596)
# ·项目3 mnist手写数字识别的代码熟悉与了解
### 学习到的知识点：
- 1.各种函数的声明与定义

        def __init__(self):
                super(MnistModel, self).__init__()
                self.conv1 = Conv2d(in_channels=1, out_channels=10, kernel_size=5, stride=1, padding=0)
                self.maxpool1 = MaxPool2d(2)
                self.conv2 = Conv2d(in_channels=10, out_channels=20, kernel_size=5, stride=1, padding=0)
                self.maxpool2 = MaxPool2d(2)
                self.linear1 = Linear(320, 128)
                self.linear2 = Linear(128, 64)
                self.linear3 = Linear(64, 10)
                self.relu = ReLU()

- 2.如何进行前进操作


            def forward(self, x):
                    x = self.relu(self.maxpool1(self.conv1(x)))
                    x = self.relu(self.maxpool2(self.conv2(x)))
                    x = x.view(x.size(0), -1)
                    x = self.linear1(x)
                    x = self.linear2(x)
                    x = self.linear3(x)
            
                return x
    



- [ ] 未完成的任务{#nodone}
- [ ] 碰到的问题{#questions}


> 1 .缺乏mnist数据集
>
>


学习难度|知识点量|
-|-
****|****

### 参考代码：  [handwriting.py](../../../../PycharmProjects/pythonProject5/handwriting.py)
# 二. 遇到的困难和问题
### 技术难点：
#### 1.没学过指针，内存分配等计算机底层原理，所以对cuda的并行计算程序并不是很了解
例如：

        __global__ void VectorAddGPU(const float *a, const float *b,float *c, const int n) {
            int i = blockDim.x * blockIdx.x + threadIdx.x; // 线程ID
            if (i < n) {
                 c[i] = a[i] + b[i]; //每个线程需要做的事情
            }
    }
----
这是一个我让ai生成的关于cuda程序。这段 CUDA 程序的主要功能是实现一个简单的模板操作，针对输入数组的每个元素，将其周围的值（受到一个指定半径的影响）进行求和，并将结果存储到输出数组中。这种处理类似于图像处理中的卷积运算。

    #include <iostream>  
    #include <algorithm>  
    
    using namespace std;  
    
    #define N 1024  
    #define RADIUS 3  
    #define BLOCK_SIZE 16  
    
    __global__ void idnt(int *in, int *out) {   
        __shared__ int temp[BLOCK_SIZE + 2 * RADIUS]; // 共享内存，大小为 BLOCK_SIZE 加上两个半径  
        int index = threadIdx.x + blockIdx.x * blockDim.x; // 计算当前线程输送的全局索引  
        int tempIndex = threadIdx.x + RADIUS; // 计算临时数组的索引  
    
        // 将元素读入共享内存  
        temp[tempIndex] = in[index];   
        // 边界条件处理  
        if (threadIdx.x < RADIUS) {   
            temp[tempIndex - RADIUS] = (index >= RADIUS) ? in[index - RADIUS] : 0; // 处理左边界  
            temp[tempIndex + BLOCK_SIZE] = (index + RADIUS < N) ? in[index + RADIUS] : 0; // 处理右边界  
        }  
    
        // 同步以确保数据读取完成  
        __syncthreads();   
    
        // 应用模板  
        int result = 0;   
        for (int offset = -RADIUS; offset <= RADIUS; offset++)   
            result += temp[tempIndex + offset]; // 计算结果  
        // 存储结果  
        out[index] = result;   
    }  
    
    void fill_ints(int *x, int n) {   
        fill_n(x, n, 1); // 初始化数组为1  
    }  
    
    int main(void) {   
        int *in, *out; // 主机上的输入和输出指针  
        int *d_in, *d_out; // 设备上的输入和输出指针  
        int size = N + 2 * RADIUS; // 计算大小  
    
        // 为主机分配内存并设置值  
        in = (int *)malloc(size * sizeof(int));   
        fill_ints(in, N + 2*RADIUS);   
        out = (int *)malloc(size * sizeof(int));   
    
        // 为设备拷贝分配内存  
        cudaMalloc((void **)&d_in, size * sizeof(int));   
        cudaMalloc((void **)&d_out, size * sizeof(int));   
    
        // 拷贝到设备  
        cudaMemcpy(d_in, in, size * sizeof(int), cudaMemcpyHostToDevice);   
        cudaMemcpy(d_out, out, size * sizeof(int), cudaMemcpyHostToDevice);   
    
        // 在 GPU 上启动内核  
        idnt<<<N/BLOCK_SIZE, BLOCK_SIZE>>>(d_in, d_out);   
    
        // 拷贝结果回主机  
        cudaMemcpy(out, d_out, size * sizeof(int), cudaMemcpyDeviceToHost);   
    
        // 清理  
        free(in);   
        free(out);   
        cudaFree(d_in);   
        cudaFree(d_out);   
        return 0;   
    }  
#### 尚且无法了解计算机内部的指针环境和内存分配



任务难度|学习时长
-|-
*****|***

# 四. 下周工作计划
 ## 主要目标：
- 1. 学习c++语言：学校老师没教但是发现很多地方都要用到 3h
- 2. 通过c语言了解计算机的指针与内存分配 1h
- 3. 继续调研异构模型（可以询问学长学姐们） 1h


#### 时间安排：一周共计5小时


### 5. 需要的支持与资源

**资源需求： 需要minst数据集。**
