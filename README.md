<div align="center">

# Swan
**A Lightweight Language Model Execution Environment Using FPGA**

English | [日本語](./docs/README_JP.md) | [中文](./docs/README_CN.md)

Swan is an OSS project implemented in C++.  
Its goal is to efficiently run language models on general-purpose FPGAs using High-Level Synthesis (HLS).
</div>

<div align="center">
<img src="./images/swan_image.png" width="50%">
</div>

This project aims to enable language model inference on FPGAs, supporting AI applications in edge devices and environments with limited resources.  

## Features

- Versatility: Supports common FPGA boards such as the KV260.
- Scalability: The source code is written in C++, making customization and extension easy.
- Lightweight: Considers the size constraints of language models and adopts an efficient architecture.

## Dependencies

To build and run Swan, the following tools and libraries are required:

- CMake
- g++
- HLS tools (e.g., Vivado HLS)

## Clone & Download Weight Files

To clone the Swan repository, run the following command:
```bash
$ git clone git@github.com:turingmotors/swan.git
$ cd swan
```

Download 15M parameter model from [huggingface.co/karpathy/tinyllamas](https://huggingface.co/karpathy/tinyllamas/tree/main):
```
mkdir model
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin -O model/stories15M.bin
wget https://raw.githubusercontent.com/leloykun/llama2.cpp/master/tokenizer.bin -O model/tokenizer.bin
```

## Building

### FPGA Environment

See [technical blog](https://zenn.dev/turing_motors/articles/82505880d27d65) for details on building Swan in an FPGA environment.

### CPU Environment

```bash
$ mkdir -p build && cd build
$ cmake ..
$ make && cd ..
```
Once the build is complete, you can run Swan with the following command:

```bash
$ ./build/swan
```

## Command Line Options

Swan supports the following options:

```bash
Usage: ./build/swan [options]
Options:
  --weight_path   : Weight file path
  --vocab_path    : Tokenizer file path
  --max_seq       : Maximum sequence length
  --temp          : Temperature for sampling
  --color         : Enable color output
  --log           : Enable log output
  --help, -h      : Show this help message
```

## Reference Projects
This project is inspired by [llama2.c](https://github.com/karpathy/llama2.c).

## License
This project is released under the [Apache License 2.0](./LICENSE).

## Contributions
Contributions to Swan are highly welcome. Please submit feedback and improvement suggestions through Issues and Pull Requests.  
[Turing Inc.](https://www.turing-motors.com/en) is supporting the development of Swan.
