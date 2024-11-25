/*  Atharva Vikas
 *  Jadhav
 *  ajadhav9
 */

#ifndef A1_HPP
#define A1_HPP

#include <vector>
#include <cmath>
#include <omp.h>

void filter2d(long long int n, long long int m, const std::vector<float>& K, std::vector<float>& A) {  
    // n / (5 * p) for block size
    long long int blockSize = ceil(double(n) / (double(5) * double(omp_get_max_threads())));
    // (n / blockSize) * 2 number of memoizations required
    long long int memoSize = (n / (blockSize));
    std::vector<float> bottomMemo(memoSize * m);
    std::vector<float> topMemo(memoSize * m);
    // Construct memoization for block overlaps
    #pragma omp parallel for default(none) shared(bottomMemo, topMemo, blockSize, A, n, m, memoSize)
    for (long long int row = 0; row < memoSize; row++) {
        for (long long int col = 0; col < m; col++) {
            if ((row + 1) * blockSize < n) {
                bottomMemo[(row * m) + col] = A[(((row + 1) * blockSize) * m) + col];
                topMemo[(row * m) + col] = A[((((row + 1) * blockSize) + 1) * m) + col];   
            }
        }
    }
    #pragma omp parallel default(none) shared(bottomMemo, topMemo, K, A, n, m, blockSize, memoSize)
    {
        // Temporary buffers and variables
        std::vector<float> buf(m);
        std::vector <float> temp(m);
        float val;
        float prev;
        float k;
        #pragma omp for schedule(dynamic, blockSize)
        for (long long int row = 1; row < n-1; row++) {
            // Initialize buf if the current row is the first row of that block
            if(row-1%blockSize == 0){
                if(row == 1){
                    for (long long int col = 0; col < m; col++) 
                        buf[col] = A[col];
                } else {
                    for (long long int col = 0; col < m; col++)
                        buf[col] = bottomMemo[((((row - 1) / blockSize) - 1) * m) + col];
                }
            }
            temp[0] = A[row * m];
            temp[m-1] = A[(row * m) + (m-1)];
            prev = A[row * m];
            for (long long int col = 1; col < m-1; col++) {
                val = 0.0f;
                temp[col] = A[(row * m) + col];
                // Apply and compute kernel
                for(long long int innerRow = -1; innerRow <= 1; innerRow++) {
                    for(long long int innerCol = -1; innerCol <= 1; innerCol++) {
                        k = K[((innerRow+1) * 3) + (innerCol+1)];
                        if (innerRow == 0 && innerCol == -1) {
                            val += (prev * k);
                        } else if (row == 1){
                            val += (A[((row+innerRow) * m) + (col+innerCol)] * k);
                        } else if (innerRow == -1){
                            if((row-1)%blockSize == 0)
                                val += (bottomMemo[((((row-1)/(blockSize)) - 1) * m) + (col+innerCol)] * k);
                            else 
                                val += (buf[(col+innerCol)] * k);                            
                        } else if (innerRow == 1 && row%blockSize == 0){
                            val += (topMemo[(((row/(blockSize)) - 1) * m) + (col+innerCol)] * k);
                        } else {
                            val += (A[((row+innerRow) * m) + (col+innerCol)] * k);
                        }
                    }
                }
                // Taking the original value for next iteration
                prev = A[(row * m) + col];
                A[(row * m) + col] = val;
            }
            // storing original values for next iteration
            for (long long int col = 0; col < m; col++)
                buf[col] = temp[col];
        }
    }
    return;
} // filter2d


#endif // A1_HPP

