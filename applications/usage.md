# Execution and Compilation Guide: AxPike JPEG Encoder

This guide covers the steps to compile and execute the RISC-V apllications using the **AxPike** simulator with **AxRAM**.

## 1. Important: Pre-Execution Steps

To ensure the output is not corrupted by system logs, you must handle the Proxy Kernel (PK) output.

### Option A: Modify the Proxy Kernel (Recommended)

You need to silence the bootloader messages in the `axpike-pk` source:

1. Open the file: `axpike-pk/machine/minit.c`
2. **Comment out** the following line:
```c
// printm("bbl loader");

```


3. Recompile the `axpike-pk`.

---

## 2. Compiling the Application

Use the RISC-V cross-compiler to generate a static binary with high-level optimization:

```bash

$ riscv64-unknown-elf-g++ -O3 -static -o ./toojpeg_encoder main.cpp toojpeg.cpp
OR
$ riscv64-unknown-elf-g++ -O3 -static -o ./dominant_freq dominant_freq.cpp kiss_fft.c -lm

```

* **`-O3`**: High optimization level for performance.
* **`-static`**: Required for bare-metal/simulator execution to include all libraries.

---

## 3. Executing the Application

Run the simulation using **AxPike** with the following parameters for cache and approximate memory (AXRAM) attention you can alter the parameters:

to you use the parameter de errors you need set for on the variavel initial in /axpike-isa-sim/adele/adf/AxRAM.adf


```bash
$ axpike --adele=mem_read_prob:1e-2,linesz:32 --adele-activate=0:AXRAM --dc=128:8:32 --ic=256:4:32 --l2=1024:4:32 pk /home/guilherme/AxCept-Bench/applications/jpeg/src/toojpeg_encoder 100 < path.csv > output.jpeg
OR
$ axpike  --adele=mem_read_prob:1e-3,linesz:32  --adele-activate=0:AXRAM  --dc=128:8:32  --ic=256:4:32  --l2=1024:4:32  pk /home/guilherme/AxCept-Bench/applications/fft/src/dominant_freq < input.csv > saida.bin

```

### Parameter Breakdown:

* **`--adele`**: Sets memory read error probability and line size.
* **`--adele-activate`**: Activates **AXRAM** for the specified memory region.
* **`--dc / --ic / --l2`**: Configures Data, Instruction, and L2 caches (Size:Ways:LineSize).
* **`pk`**: The Proxy Kernel used to load and run the binary.
* **`1 < lena.csv`**: Pass the quality factor (1) and redirect the input image data.
* **`> output.jpeg`**: Redirects the application output to the JPEG file.

---

## 4. Troubleshooting

If the `output.jpeg` is not opening, ensure that:

1. The `printm` line was successfully commented out.
2. You are using the correct `xxx.csv` format for the encoder.

---
