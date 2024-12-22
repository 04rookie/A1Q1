# Element-Wise 2D Filter with OpenMP - README

This project implements an efficient **OpenMP** program for applying 2D filters using element-wise operations. The implementation is designed for high performance in image and signal processing tasks, leveraging multithreading for parallelism.

## Key Features
- **Element-Wise Filtering**: For each pixel $$ A_{ij} $$, the filter computes the element-wise product between a 3×3 kernel $$ K $$ and the corresponding 3×3 submatrix of $$ A $$ centered at $$ (i, j) $$. The filtered value $$ A'_{ij} $$ is obtained by summing all elements of the resulting matrix.
- **Boundary Handling**: The filter is not applied to the first and last rows/columns of the input array $$ A $$; these boundary pixels remain unchanged.
- **Parallel Execution**: Utilizes OpenMP to parallelize the computation, achieving significant speedups on multi-core systems.

## Implementation Highlights
- Input image $$ A $$ is represented as a flattened 1D array for memory efficiency.
- The program achieves high performance by dividing work across threads using OpenMP, ensuring efficient load balancing.
- Designed for extensibility, allowing easy adaptation for other kernel sizes or filter types.

## Use Case
This implementation is ideal for scenarios requiring fast application of 2D filters in image or signal processing, particularly in high-performance computing environments.

## Requirements
- C++ compiler with OpenMP support (e.g., GCC or Clang)
- Input image data in a format compatible with the program

## How to Run
1. Compile the code with OpenMP support:
   ```bash
   g++ -fopenmp -o omp_filter omp_filter.cpp
   ```
2. Run the program with the desired number of threads:
   ```bash
   OMP_NUM_THREADS=<number_of_threads> ./omp_filter
   ```

## Output
The program outputs a filtered version of the input image, where each pixel (excluding boundaries) is processed using the element-wise product and summation method described above.

---

Feel free to explore and adapt this implementation for your filtering needs!
