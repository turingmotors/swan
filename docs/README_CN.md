<div align="center">

# Swan
**使用FPGA的轻量级语言模型执行环境**

[英文](../README.md) | [日文](./README_JP.md) | 中文

Swan是一个用C++实现的开源软件项目。
它的目标是利用高级综合（HLS）在通用FPGA上高效地运行语言模型。
</div>

<div align="center">
<img src="../images/swan_image.png" width="50%">
</div>

该项目旨在FPGA上实现语言模型推理，支持边缘设备和资源有限环境中的AI应用。

## 特点

- 通用性: 支持常见的FPGA板，如KV260。
- 可扩展性: 源代码用C++编写，便于定制和扩展。
- 轻量级: 考虑到语言模型大小的限制，采用了高效的架构。

## 依赖

构建和运行Swan需要以下工具和库。

- CMake
- g++
- HLS工具（如Vivado HLS）

## 克隆和下载

要克隆Swan仓库，请运行以下命令:

```bash
$ git clone git@github.com:turingmotors/swan.git
$ cd swan
```

从[huggingface.co/karpathy/tinyllamas](https://huggingface.co/karpathy/tinyllamas/tree/main)下载15M参数模型:
```bash
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin -O model/stories15M.bin
```

## 构建

### FPGA环境

有关在FPGA环境中构建Swan的详细信息，请参阅[技术博客](https://zenn.dev/turing_motors/articles/82505880d27d65)。

### CPU环境

```bash
$ mkdir -p build && cd build
$ cmake ..
$ make && cd ..
```
构建完成后，可以用以下命令运行Swan。

```bash
$ ./build/swan
```

## 命令行选项

Swan支持以下选项。

```bash
Usage: ./build/swan [options]
Options:
  --weight_path   : 权重文件路径
  --vocab_path    : 词汇表文件路径
  --max_seq       : 最大序列长度
  --temp          : 采样温度
  --color         : 启用彩色输出
  --log           : 启用日志输出
  --help, -h      : 显示此帮助信息
```

## 参考项目
该项目参考了[llama2.c](https://github.com/karpathy/llama2.c)。

## 许可证
该项目根据[Apache License 2.0](../LICENSE)许可证公开。

## 贡献
欢迎为Swan做出贡献。通过Issue或Pull Request提供反馈和改进建议。
[Turing株式会社](https://www.turing-motors.com/en)支持Swan的开发。
