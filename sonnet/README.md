**Try it live here:**  
👉 [https://huggingface.co/spaces/alzaemaliq/NanoGPT-Shakespeare](https://huggingface.co/spaces/alzaemaliq/NanoGPT-Shakespeare)

# NanoGPT-Shakespeare

This is a basic reimplementation of the "Attention Is All You Need" paper, focused on the decoder-only transformer architecture used in GPT models. The model is trained from scratch on the Tiny Shakespeare dataset using character-level tokenization.

The code is written for readability and learning. It avoids high-level libraries or frameworks and builds the model components (multi-head attention, feedforward layers, positional embeddings, etc.) using raw PyTorch.

## Features

- Decoder-only transformer (GPT-style)
- Character-level tokenizer (custom encode/decode logic)
- Multi-head self-attention with causal masking
- Learned positional embeddings
- LayerNorm and residual connections
- Simple training loop using AdamW
- Generates text one character at a time

## Model details

- Embedding dimension: 128
- Number of heads: 4
- Number of layers: 4
- Context window (block size): 64
- Batch size: 16
- Dropout: 0.1
- Training steps: 9000

## Data

The model is trained on the Tiny Shakespeare dataset (a small corpus of Shakespeare text). Character-level tokenization is used instead of word or subword tokenization.

## Notes

The `train.py` file contains the full model implementation and training loop. It also includes many inline comments and side notes where I recorded my thoughts while learning and building the model. Some of these comments may be messy or unintelligible at times, as they were written in the moment during the learning process.

**ライブデモはこちら：**  
👉 [https://huggingface.co/spaces/alzaemaliq/NanoGPT-Shakespeare](https://huggingface.co/spaces/alzaemaliq/NanoGPT-Shakespeare)

# NanoGPT-Shakespeare

このプロジェクトは「Attention Is All You Need」論文の基本的な再実装であり、GPTモデルで使われているデコーダ専用のトランスフォーマーアーキテクチャに焦点を当てています。  
Tiny Shakespeareデータセットを使用し、文字単位のトークン化でゼロから学習を行っています。

コードは学習と可読性を重視しており、高レベルのライブラリを避け、マルチヘッドアテンション、フィードフォワード層、位置埋め込みなどのモデル構成要素をPyTorchで低レベルから構築しています。

## 特徴

- デコーダ専用トランスフォーマー（GPTスタイル）
- 文字単位のトークナイザー（独自のエンコード／デコード）
- マルチヘッド自己注意機構（因果マスキング付き）
- 学習された位置埋め込み
- LayerNormと残差接続
- AdamWによるシンプルな学習ループ
- テキストを1文字ずつ生成

## モデル詳細

- 埋め込み次元数：128  
- ヘッド数：4  
- レイヤー数：4  
- コンテキストウィンドウ（ブロックサイズ）：64  
- バッチサイズ：16  
- ドロップアウト：0.1  
- 学習ステップ数：9000

## データ

Tiny Shakespeare（シェイクスピアの小規模コーパス）を使用。  
単語やサブワードではなく、文字単位のトークン化を行っています。

## 補足

`train.py` にモデルの実装と学習ループがすべて含まれています。  
多くのインラインコメントやメモがあり、学習中に思いついたことをそのまま書き留めています。  
そのため、雑だったり読みづらい部分があるかもしれませんがご了承ください。
