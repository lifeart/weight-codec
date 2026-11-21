#### Qwen2.5-0.5B: real file bytes vs the phase-1 entropy estimate (same algorithms, same seeds)

| phase-2 file | phase-1 config | phase-1 estimate bpw | phase-2 REAL bpw | real vs estimate | MB | quality metric | phase-1 | phase-2 (decoded file) | phase-1 PPL | phase-2 PPL |
|---|---|---|---|---|---|---|---|---|---|---|
| B_gptq4_ans | gptq_int4_g128 | 3.7568 | 3.7570 | 0.01 % | 168.04 | wiki-KL | 0.0746 | 0.0746 | 14.033 | 14.033 |
| B_ecsq_c0.35 | gptq_ecsq_row_c0.35 | 3.5784 | 3.5782 | -0.01 % | 160.05 | wiki-KL | 0.0599 | 0.0599 | 13.838 | 13.838 |
| B_ecsq_c0.5 | gptq_ecsq_row_c0.5 | 3.0748 | 3.0747 | -0.00 % | 137.53 | wiki-KL | 0.1215 | 0.1215 | 14.740 | 14.740 |
| T_gptq4_ans | indep_gptq_int4_g128 | 3.7571 | 3.7573 | 0.01 % | 168.06 | chat-KL | 0.1482 | 0.1482 | 15.313 | 15.313 |
| T_ecsq_c0.35 | indep_gptq_ecsq_c0.35 | 3.5785 | 3.5783 | -0.01 % | 160.05 | chat-KL | 0.1173 | 0.1173 | 15.089 | 15.089 |
| T_ecsq_c0.5 | indep_gptq_ecsq_c0.5 | 3.0750 | 3.0749 | -0.00 % | 137.54 | chat-KL | 0.2562 | 0.2562 | 16.061 | 16.061 |
| T_pframe_c0.35 | pframe_gptq2_c0.35 | 1.1137 | 1.1009 | -1.15 % | 49.24 | chat-KL | 0.1279 | 0.1279 | 15.072 | 15.072 |
| T_pframe_c0.5 | pframe_gptq2_c0.5 | 0.9781 | 0.9670 | -1.13 % | 43.25 | chat-KL | 0.2556 | 0.2558 | 16.049 | 16.049 |
| T_delta_c0.06 | delta_gptq_c0.06 | 2.5685 | 2.5685 | 0.00 % | 114.88 | chat-KL | 0.0031 | 0.0031 | 14.255 | 14.255 |
| T_delta_c0.12 | delta_gptq_c0.12 | 1.6865 | 1.6866 | 0.01 % | 75.44 | chat-KL | 0.0105 | 0.0105 | 14.269 | 14.269 |
| T_delta_c0.25 | delta_gptq_c0.25 | 0.8425 | 0.8427 | 0.02 % | 37.69 | chat-KL | 0.0375 | 0.0375 | 14.237 | 14.237 |
| T_lowrank_r32 | delta_lowrank_r32 | 0.7868 | 0.7869 | 0.01 % | 35.20 | chat-KL | 0.0874 | 0.0874 | 13.636 | 13.636 |
| T_lowrank_r128 | delta_lowrank_r128 | 3.1473 | 3.1473 | 0.00 % | 140.77 | chat-KL | 0.0522 | 0.0522 | 13.675 | 13.675 |

#### Qwen2.5-0.5B {base, Instruct} — linears (block nn.Linear), real files (MPS fp16, WikiText-2 PPL on all 146 windows)

| model | format | MB | real bpw | phase-1 proxy bpw | real vs proxy | zstd-19 bpw | PPL | wiki-KL | chat-KL | decode ms CPU constriction 1-thr | decode ms CPU rans16 parallel | decode ms MPS (rans16 CPU-par + GPU recon) | decode ms GPU (Metal rANS + GPU recon) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | fp16 (raw) | 715.7 | 16.000 | 16.000 | 0.00 % | — | 13.07 | 0.0000 | — | — | — | — | — |
| base | GPTQ-int4 g128 fixed-rate (packed nibbles) | 185.9 | 4.156 | 4.156 | 0.00 % | — | 14.03 | 0.0746 | — | 635 | 623 | 597 | 615 |
| base | GPTQ-int4 g128 + rANS | 168.0 | 3.757 | 3.757 | 0.01 % | 3.859 | 14.03 | 0.0746 | — | 6957 | 1100 | 1182 | 591 |
| base | GPTQ-ECSQ c=0.35 + rANS | 160.0 | 3.578 | 3.578 | -0.01 % | 3.628 | 13.84 | 0.0599 | — | 9482 | 1130 | 1154 | 597 |
| base | GPTQ-ECSQ c=0.5 + rANS | 137.5 | 3.075 | 3.075 | -0.00 % | 3.144 | 14.74 | 0.1215 | — | — | — | — | — |
| fine-tune | fp16 (raw) | 715.7 | 16.000 | 16.000 | 0.00 % | — | 14.24 | 0.0000 | 0.0000 | — | — | — | — |
| fine-tune | GPTQ-int4 g128 fixed-rate (packed nibbles) | 185.9 | 4.156 | 4.156 | 0.00 % | — | 15.31 | 0.0799 | 0.1482 | 584 | 533 | 541 | 548 |
| fine-tune | GPTQ-int4 g128 + rANS | 168.1 | 3.757 | 3.757 | 0.01 % | 3.845 | 15.31 | 0.0799 | 0.1482 | 6848 | 1052 | 1124 | 693 |
| fine-tune | GPTQ-ECSQ c=0.35 + rANS | 160.1 | 3.578 | 3.579 | -0.01 % | 3.628 | 15.09 | 0.0645 | 0.1173 | 9161 | 1133 | 1067 | 618 |
| fine-tune | LoRA-style Δ rank 128 (fp16 factors) | 140.8 | 3.147 | 3.147 | 0.00 % | — | 13.67 | 0.0608 | 0.0522 | 448 | 446 | 285 | 289 |
| fine-tune | GPTQ-ECSQ c=0.5 + rANS | 137.5 | 3.075 | 3.075 | -0.00 % | — | 16.06 | 0.1254 | 0.2562 | — | — | — | — |
| fine-tune | full-rank Δ vs fp16 base, GPTQ c=0.06 + rANS | 114.9 | 2.568 | 2.568 | 0.00 % | 2.723 | 14.25 | 0.0018 | 0.0031 | — | — | — | — |
| fine-tune | full-rank Δ vs fp16 base, GPTQ c=0.12 + rANS | 75.4 | 1.687 | 1.686 | 0.01 % | 1.853 | 14.27 | 0.0058 | 0.0105 | 7165 | 1202 | 1126 | 608 |
| fine-tune | LoRA-style Δ rank 64 (fp16 factors) | 70.4 | 1.574 | 1.574 | 0.00 % | — | 13.62 | 0.0773 | 0.0706 | — | — | — | — |
| fine-tune | P-frame on base I-frame lattice (c=0.35) | 49.2 | 1.101 | 1.101 | 0.01 % | 1.280 | 15.07 | 0.0657 | 0.1279 | 15353 | 2345 | 2147 | 956 |
| fine-tune | P-frame on base lattice (c=0.5) | 43.3 | 0.967 | 0.967 | 0.02 % | 1.132 | 16.05 | 0.1320 | 0.2558 | — | — | — | — |
| fine-tune | full-rank Δ vs fp16 base, GPTQ c=0.25 + rANS | 37.7 | 0.843 | 0.843 | 0.02 % | 1.029 | 14.24 | 0.0205 | 0.0375 | 6246 | 1172 | 1055 | 579 |
| fine-tune | LoRA-style Δ rank 32 (fp16 factors) | 35.2 | 0.787 | 0.787 | 0.01 % | — | 13.64 | 0.0880 | 0.0874 | 378 | 383 | 220 | 221 |


**Pair {base, fine-tune} — every tensor, real MB** (Qwen2.5-0.5B {base, Instruct})

| scenario | pair MB | base PPL | fine-tune chat-KL | fine-tune wiki-KL |
|---|---|---|---|---|
| both fp16 (raw) | 1976.1 | 13.07 | 0.0000 | 0.0000 |
| both GPTQ-int4 fixed-rate (standard int4 files), rest fp16 | 916.6 | 14.03 | 0.1482 | 0.0799 |
| both GPTQ-int4 + rANS, rest lossless | 694.3 | 14.03 | 0.1482 | 0.0799 |
| both independent GPTQ-ECSQ c=0.35, rest lossless | 678.3 | 13.84 | 0.1173 | 0.0645 |
| FAMILY: base I-frame + fine-tune P-frame (c=0.35), rest lossless (+XOR P) | 531.7 | 13.84 | 0.1279 | 0.0657 |
| FAMILY: base I-frame + P-frame (c=0.5) | 503.2 | 14.74 | 0.2558 | 0.1320 |
| lossless fp16 base + full-rank Δ c=0.06 | 1152.9 | 13.07 | 0.0031 | 0.0018 |
| lossless fp16 base + full-rank Δ c=0.12 | 1113.5 | 13.07 | 0.0105 | 0.0058 |
| lossless fp16 base + full-rank Δ c=0.25 | 1075.7 | 13.07 | 0.0375 | 0.0205 |
| lossless fp16 base + LoRA-style Δ r=32 | 1073.2 | 13.07 | 0.0874 | 0.0880 |
| lossless fp16 base + LoRA-style Δ r=64 | 1108.4 | 13.07 | 0.0706 | 0.0773 |
| lossless fp16 base + LoRA-style Δ r=128 | 1178.8 | 13.07 | 0.0522 | 0.0608 |


Rest tensors (embeddings + norms + biases, lossless): B_rest_raw16: 272.4 MB (16.00 bit/param), B_rest_ll16: 179.0 MB (10.51 bit/param), T_rest_raw16: 272.4 MB (16.00 bit/param), T_rest_ll16: 179.2 MB (10.52 bit/param), T_rest_xor_base: 143.4 MB (8.42 bit/param)

#### SmolLM2-360M {base, Instruct} — linears (block nn.Linear), real files (local MPS fp16, PPL on all windows (HF jobs were cancelled externally))

| model | format | MB | real bpw | phase-1 proxy bpw | real vs proxy | zstd-19 bpw | PPL | wiki-KL | chat-KL | decode ms CPU constriction 1-thr | decode ms CPU rans16 parallel | decode ms MPS (rans16 CPU-par + GPU recon) | decode ms GPU (Metal rANS + GPU recon) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | fp16 (raw) | 629.1 | 16.000 | 16.000 | 0.00 % | — | 11.47 | 0.0000 | — | — | — | — | — |
| base | GPTQ-int4 g64 fixed-rate (packed nibbles) | 169.6 | 4.313 | 4.312 | 0.00 % | — | 12.51 | 0.0856 | — | 481 | 473 | 591 | 599 |
| base | GPTQ-int4 g64 + rANS | 159.4 | 4.052 | 4.052 | 0.01 % | 4.097 | 12.51 | 0.0856 | — | 5976 | 960 | 1263 | 594 |
| base | GPTQ-ECSQ c=0.35 + rANS | 140.9 | 3.582 | 3.583 | -0.01 % | 3.614 | 12.66 | 0.0906 | — | 7927 | 952 | 1148 | 493 |
| base | GPTQ-ECSQ c=0.5 + rANS | 121.0 | 3.078 | 3.079 | -0.00 % | 3.146 | 14.18 | 0.2030 | — | — | — | — | — |
| fine-tune | fp16 (raw) | 629.1 | 16.000 | 16.000 | 0.00 % | — | 12.74 | 0.0000 | 0.0000 | — | — | — | — |
| fine-tune | GPTQ-int4 g64 fixed-rate (packed nibbles) | 169.6 | 4.313 | 4.312 | 0.00 % | — | 13.90 | 0.0974 | 0.1184 | 417 | 421 | 564 | 558 |
| fine-tune | GPTQ-int4 g64 + rANS | 159.4 | 4.053 | 4.052 | 0.01 % | 4.090 | 13.90 | 0.0974 | 0.1184 | 5911 | 897 | 1161 | 585 |
| fine-tune | GPTQ-ECSQ c=0.35 + rANS | 140.9 | 3.582 | 3.583 | -0.01 % | 3.614 | 14.08 | 0.1080 | 0.1375 | 7783 | 904 | 1097 | 508 |
| fine-tune | LoRA-style Δ rank 128 (fp16 factors) | 138.9 | 3.533 | 3.533 | 0.00 % | — | 12.37 | 0.0541 | 0.0885 | — | — | — | — |
| fine-tune | GPTQ-ECSQ c=0.5 + rANS | 121.0 | 3.078 | 3.078 | -0.00 % | 3.146 | 15.82 | 0.2343 | 0.3029 | — | — | — | — |
| fine-tune | full-rank Δ vs fp16 base, GPTQ c=0.06 + rANS | 97.6 | 2.482 | 2.482 | 0.00 % | 2.652 | 12.75 | 0.0029 | 0.0035 | — | — | — | — |
| fine-tune | LoRA-style Δ rank 64 (fp16 factors) | 69.5 | 1.767 | 1.767 | 0.00 % | — | 12.33 | 0.0751 | 0.1448 | — | — | — | — |
| fine-tune | full-rank Δ vs fp16 base, GPTQ c=0.12 + rANS | 63.0 | 1.602 | 1.602 | 0.01 % | 1.773 | 12.77 | 0.0097 | 0.0134 | 5972 | 1055 | 1158 | 498 |
| fine-tune | P-frame on base I-frame lattice (c=0.35) | 41.5 | 1.056 | 1.056 | 0.02 % | 1.220 | 14.04 | 0.1071 | 0.1379 | 13152 | 1864 | 2087 | 930 |
| fine-tune | P-frame on base lattice (c=0.5) | 36.3 | 0.922 | 0.922 | 0.02 % | 1.084 | 15.77 | 0.2287 | 0.3018 | — | — | — | — |
| fine-tune | LoRA-style Δ rank 32 (fp16 factors) | 34.7 | 0.883 | 0.883 | 0.01 % | — | 12.34 | 0.0915 | 0.1874 | 327 | 311 | 254 | 254 |
| fine-tune | full-rank Δ vs fp16 base, GPTQ c=0.25 + rANS | 30.6 | 0.777 | 0.777 | 0.03 % | 0.953 | 12.63 | 0.0274 | 0.0681 | — | — | — | — |


**Pair {base, fine-tune} — every tensor, real MB** (SmolLM2-360M {base, Instruct})

| scenario | pair MB | base PPL | fine-tune chat-KL | fine-tune wiki-KL |
|---|---|---|---|---|
| both fp16 (raw) | 1447.3 | 11.47 | 0.0000 | 0.0000 |
| both GPTQ-int4 fixed-rate (standard int4 files), rest fp16 | 528.2 | 12.51 | 0.1184 | 0.0974 |
| both GPTQ-int4 + rANS, rest lossless | 442.9 | 12.51 | 0.1184 | 0.0974 |
| both independent GPTQ-ECSQ c=0.35, rest lossless | 405.9 | 12.66 | 0.1375 | 0.1080 |
| FAMILY: base I-frame + fine-tune P-frame (c=0.35), rest lossless (+XOR P) | 295.3 | 12.66 | 0.1379 | 0.1071 |
| FAMILY: base I-frame + P-frame (c=0.5) | 270.2 | 14.18 | 0.3018 | 0.2287 |
| lossless fp16 base + full-rank Δ c=0.06 | 839.6 | 11.47 | 0.0035 | 0.0029 |
| lossless fp16 base + full-rank Δ c=0.12 | 805.0 | 11.47 | 0.0134 | 0.0097 |
| lossless fp16 base + full-rank Δ c=0.25 | 772.6 | 11.47 | 0.0681 | 0.0274 |
| lossless fp16 base + LoRA-style Δ r=32 | 776.8 | 11.47 | 0.1874 | 0.0915 |
| lossless fp16 base + LoRA-style Δ r=64 | 811.5 | 11.47 | 0.1448 | 0.0751 |
| lossless fp16 base + LoRA-style Δ r=128 | 881.0 | 11.47 | 0.0885 | 0.0541 |


Rest tensors (embeddings + norms + biases, lossless): B_rest_raw16: 94.5 MB (16.00 bit/param), B_rest_ll16: 62.1 MB (10.51 bit/param), T_rest_raw16: 94.5 MB (16.00 bit/param), T_rest_ll16: 62.1 MB (10.51 bit/param), T_rest_xor_base: 50.8 MB (8.60 bit/param)

#### SmolLM2-135M {base, Instruct} (Space demo pair) — linears (block nn.Linear), real files (MPS fp16, PPL on all windows)

| model | format | MB | real bpw | phase-1 proxy bpw | real vs proxy | zstd-19 bpw | PPL | wiki-KL | chat-KL | decode ms CPU constriction 1-thr | decode ms CPU rans16 parallel | decode ms MPS (rans16 CPU-par + GPU recon) | decode ms GPU (Metal rANS + GPU recon) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | fp16 (raw) | 212.3 | 16.000 | 16.000 | 0.00 % | — | 15.43 | 0.0000 | — | — | — | — | — |
| base | GPTQ-int4 g64 fixed-rate (packed nibbles) | 57.2 | 4.313 | 4.312 | 0.01 % | — | 20.69 | 0.2813 | — | 258 | 290 | 306 | 304 |
| base | GPTQ-int4 g64 + rANS | 53.8 | 4.051 | 4.050 | 0.01 % | 4.088 | 20.69 | 0.2813 | — | 2165 | 498 | 559 | 364 |
| base | GPTQ-ECSQ c=0.35 + rANS | 47.7 | 3.594 | 3.594 | -0.01 % | 3.623 | 20.93 | 0.2833 | — | 2723 | 442 | 532 | 295 |
| base | GPTQ-ECSQ c=0.5 + rANS | 41.0 | 3.090 | 3.090 | -0.01 % | 3.156 | 29.51 | 0.6096 | — | 2610 | 441 | 521 | 289 |
| fine-tune | fp16 (raw) | 212.3 | 16.000 | 16.000 | 0.00 % | — | 17.36 | 0.0000 | 0.0000 | — | — | — | — |
| fine-tune | LoRA-style Δ rank 128 (fp16 factors) | 78.2 | 5.889 | 5.889 | 0.00 % | — | 17.10 | 0.0418 | 0.0707 | 234 | 234 | 234 | 251 |
| fine-tune | GPTQ-int4 g64 fixed-rate (packed nibbles) | 57.2 | 4.313 | 4.312 | 0.01 % | — | 23.40 | 0.2882 | 0.4000 | 228 | 248 | 321 | 313 |
| fine-tune | GPTQ-int4 g64 + rANS | 53.8 | 4.051 | 4.050 | 0.01 % | 4.087 | 23.40 | 0.2882 | 0.4000 | 2128 | 440 | 574 | 391 |
| fine-tune | GPTQ-ECSQ c=0.35 + rANS | 47.7 | 3.594 | 3.594 | -0.01 % | 3.623 | 24.84 | 0.3567 | 0.5049 | 2671 | 402 | 538 | 297 |
| fine-tune | GPTQ-ECSQ c=0.5 + rANS | 41.0 | 3.090 | 3.090 | -0.01 % | 3.156 | 32.75 | 0.6413 | 1.0362 | 2553 | 418 | 538 | 293 |
| fine-tune | LoRA-style Δ rank 64 (fp16 factors) | 39.1 | 2.945 | 2.944 | 0.01 % | — | 17.09 | 0.0706 | 0.1443 | 235 | 234 | 215 | 214 |
| fine-tune | full-rank Δ vs fp16 base, GPTQ c=0.06 + rANS | 34.5 | 2.596 | 2.596 | 0.00 % | 2.751 | 17.45 | 0.0069 | 0.0101 | 2395 | 488 | 549 | 307 |
| fine-tune | full-rank Δ vs fp16 base, GPTQ c=0.12 + rANS | 22.6 | 1.707 | 1.706 | 0.02 % | 1.874 | 17.58 | 0.0181 | 0.0299 | 2166 | 477 | 553 | 298 |
| fine-tune | LoRA-style Δ rank 32 (fp16 factors) | 19.5 | 1.472 | 1.472 | 0.02 % | — | 17.10 | 0.0965 | 0.2204 | 214 | 218 | 206 | 207 |
| fine-tune | P-frame on base I-frame lattice (c=0.35) | 14.6 | 1.098 | 1.098 | 0.03 % | 1.262 | 24.09 | 0.3232 | 0.5378 | 4504 | 757 | 1088 | 484 |
| fine-tune | P-frame on base lattice (c=0.5) | 12.7 | 0.956 | 0.956 | 0.04 % | 1.119 | 34.06 | 0.6854 | 1.0116 | 4420 | 739 | 1073 | 479 |
| fine-tune | full-rank Δ vs fp16 base, GPTQ c=0.25 + rANS | 11.5 | 0.870 | 0.870 | 0.07 % | 1.036 | 17.59 | 0.0427 | 0.1018 | 1942 | 489 | 553 | 292 |


**Pair {base, fine-tune} — every tensor, real MB** (SmolLM2-135M {base, Instruct} (Space demo pair))

| scenario | pair MB | base PPL | fine-tune chat-KL | fine-tune wiki-KL |
|---|---|---|---|---|
| both fp16 (raw) | 538.1 | 15.43 | 0.0000 | 0.0000 |
| both GPTQ-int4 fixed-rate (standard int4 files), rest fp16 | 227.9 | 20.69 | 0.4000 | 0.2882 |
| both GPTQ-int4 + rANS, rest lossless | 182.6 | 20.69 | 0.4000 | 0.2882 |
| both independent GPTQ-ECSQ c=0.35, rest lossless | 170.5 | 20.93 | 0.5049 | 0.3567 |
| FAMILY: base I-frame + fine-tune P-frame (c=0.35), rest lossless (+XOR P) | 130.5 | 20.93 | 0.5378 | 0.3232 |
| FAMILY: base I-frame + P-frame (c=0.5) | 122.0 | 29.51 | 1.0116 | 0.6854 |
| lossless fp16 base + full-rank Δ c=0.06 | 315.1 | 15.43 | 0.0101 | 0.0069 |
| lossless fp16 base + full-rank Δ c=0.12 | 303.3 | 15.43 | 0.0299 | 0.0181 |
| lossless fp16 base + full-rank Δ c=0.25 | 292.2 | 15.43 | 0.1018 | 0.0427 |
| lossless fp16 base + LoRA-style Δ r=32 | 300.2 | 15.43 | 0.2204 | 0.0965 |
| lossless fp16 base + LoRA-style Δ r=64 | 319.7 | 15.43 | 0.1443 | 0.0706 |
| lossless fp16 base + LoRA-style Δ r=128 | 358.8 | 15.43 | 0.0707 | 0.0418 |


Rest tensors (embeddings + norms + biases, lossless): B_rest_raw16: 56.7 MB (16.00 bit/param), B_rest_ll16: 37.5 MB (10.60 bit/param), T_rest_raw16: 56.7 MB (16.00 bit/param), T_rest_ll16: 37.5 MB (10.59 bit/param), T_rest_xor_base: 30.7 MB (8.67 bit/param)
