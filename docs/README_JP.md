<div align="center">

# Swan
**FPGAを使用した軽量言語モデルの実行環境**

[English](../README.md) | 日本語 | [中文](./README_CN.md)

SwanはC++で実装されたOSSプロジェクトです。  
HLS(High-Level Synthesis)を利用して汎用FPGA上で言語モデルを動かすことを目的としています。
</div>

<div align="center">
<img src="../images/swan_image.png" width="50%">
</div>

このプロジェクトは、言語モデルの推論をFPGA上で実現し、エッジデバイスやリソースが限られた環境でのAIアプリケーションをサポートすることを目指します。

## 特徴

- **汎用性**: KV260などの一般的なFPGAボードに対応しています。
- **拡張性**: ソースコードはC++で記述されており、カスタマイズや拡張が容易です。
- **軽量**: 言語モデルのサイズに対する制約を考慮し、効率的なアーキテクチャを採用しています。

## 依存関係

Swanをビルドおよび実行するには、以下のツールやライブラリが必要です。

- CMake
- g++
- HLSツール（Vivado HLSなど）

## 環境構築

Swanリポジトリをクローンするには、以下のコマンドを実行します:
```bash
$ git clone git@github.com:turingmotors/swan.git
$ cd swan
```

[huggingface.co/karpathy/tinyllamas](https://huggingface.co/karpathy/tinyllamas/tree/main)からTinyStories datasetで学習されたパラメータファイルをダウンロードします:
```bash
mkdir model
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin -O model/stories15M.bin
wget https://raw.githubusercontent.com/leloykun/llama2.cpp/master/tokenizer.bin -O model/tokenizer.bin
```

## ビルド

### FPGA環境

FPGA環境でSwanをビルドする方法は、[こちらのテックブログ](https://zenn.dev/turing_motors/articles/82505880d27d65)をご覧ください。

### CPU環境

```bash
$ mkdir -p build && cd build
$ cmake ..
$ make && cd ..
```
ビルドが完了したら、以下のコマンドでSwanを実行できます。

```bash
$ ./build/swan
```

## コマンドラインオプション

Swanは以下のオプションをサポートしています。

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

## 参考プロジェクト
このプロジェクトは[llama2.c](https://github.com/karpathy/llama2.c)を参考にしています。

## ライセンス
このプロジェクトは[Apache License 2.0](../LICENSE)のもとで公開されています。

## コントリビューション
Swanへのコントリビューションは大歓迎です。IssueやPull Requestを通じて、フィードバックや改善提案をお寄せください。  
[Turing株式会社](https://turing-motors.com/)は、Swanの開発を支援しています。
