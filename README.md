# CS 336 Excercise

Stanford CS 336: Language Modeling from Scratch (Spring 2025)

This course is designed to provide sufficient knowledge of language models by walking students through the entire process of building one on their own.

## Schedule (From original course website, for reference only)

| 周数   | 主题                                | 主讲人         | 课程材料          | 关键事件                     |
|--------|-------------------------------------|----------------|-------------------|------------------------------|
| 第1周  | Overview, tokenization              | Percy          | `lecture_01.py`   | Assignment 1 out             |
|        | PyTorch, resource accounting        | Percy          | `lecture_02.py`   |                              |
| 第2周  | Architectures, hyperparameters      | Tatsu          | `lecture 3.pdf`   |                              |
|        | Mixture of experts                  | Tatsu          | `lecture 4.pdf`   |                              |
| 第3周  | GPUs                                | Tatsu          | `lecture 5.pdf`   | Assignment 1 due<br>Assignment 2 out |
|        | Kernels, Triton                     | Tatsu          | `lecture_06.py`   |                              |
| 第4周  | Parallelism                         | Tatsu          | `lecture 7.pdf`   |                              |
|        | Parallelism                         | Percy          | `lecture_08.py`   |                              |
| 第5周  | Scaling laws                        | Tatsu          | `lecture 9.pdf`   | Assignment 3 out             |
|        | —                                   | —              | —                 | Assignment 2 due             |
|        | Inference                           | Percy          | `lecture_10.py`   |                              |
| 第6周  | Scaling laws                        | Tatsu          | `lecture 11.pdf`  | Assignment 3 due<br>Assignment 4 out |
|        | Evaluation                          | Percy          | `lecture_12.py`   |                              |
| 第7周  | Data                                | Percy          | `lecture_13.py`   |                              |
|        | Data                                | Percy          | `lecture_14.py`   |                              |
| 第8周  | Alignment - SFT/RLHF                | Tatsu          | `lecture 15.pdf`  |                              |
|        | Alignment - RL                      | Tatsu          | `lecture 16.pdf`  |                              |
|        | —                                   | —              | —                 | Assignment 4 due<br>Assignment 5 out |
| 第9周  | Alignment - RL                      | Percy          | `lecture_17.py`   |                              |
|        | Guest Lecture                       | Junyang Lin    | —                 |                              |
|        | Guest Lecture                       | Mike Lewis     | —                 |                              |
|        | —                                   | —              | —                 | *(likely Assignment 5 due)*  |

## Learning Logs

- Update on 2025-12-06: Initial commit and finished lecture 1.
- Update on 2025-12-07: Finished lecture 2.
    - Memory accounting
        - Basic ideas of data types. fp32, fp16, bf16, fp8, etc.
    - Compute accouting
        - Basic ideas of FLOP/s
        - Raw estimation of time consumption of training models of different weights.
    - Pytorch basics and tensor operations
        - Device inspection. GPU, cpu, etc.
        - Element-wise operations
        - Matrix multiplication
            - A lot of optimizations under the hood. Looking forward to it!
    - jaxtyping and einops
- Update on 2025-12-07: Basic implementation of BPE Tokenizer but requires optimization.
- Update on 2025-12-15: Bug fix and optimization of BPE Tokenizer; Finished Lecture 3.
    - A working version of BPE Tokenizer with some optimizations implemented. (parallelization not implemented though)
