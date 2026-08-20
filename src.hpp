#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    auto current_query = rater.GetNextQuery();

    // Step 1: Concatenate K[0..i] in HBM
    Matrix* k_concat = nullptr;
    if (i == 0) {
      k_concat = matrix_memory_allocator.Allocate("k_concat_0");
      gpu_sim.Copy(keys[0], k_concat, kInGpuHbm);
    } else {
      Matrix* temp_k = matrix_memory_allocator.Allocate("temp_k");
      gpu_sim.Copy(keys[0], temp_k, kInGpuHbm);
      for (size_t j = 1; j <= i; ++j) {
        Matrix* new_k = matrix_memory_allocator.Allocate("k_concat_" + std::to_string(j));
        gpu_sim.Concat(temp_k, keys[j], new_k, 0, kInGpuHbm);
        if (j > 1) {
          gpu_sim.ReleaseMatrix(temp_k);
        }
        temp_k = new_k;
      }
      k_concat = temp_k;
    }

    // Step 2: Concatenate V[0..i] in HBM
    Matrix* v_concat = nullptr;
    if (i == 0) {
      v_concat = matrix_memory_allocator.Allocate("v_concat_0");
      gpu_sim.Copy(values[0], v_concat, kInGpuHbm);
    } else {
      Matrix* temp_v = matrix_memory_allocator.Allocate("temp_v");
      gpu_sim.Copy(values[0], temp_v, kInGpuHbm);
      for (size_t j = 1; j <= i; ++j) {
        Matrix* new_v = matrix_memory_allocator.Allocate("v_concat_" + std::to_string(j));
        gpu_sim.Concat(temp_v, values[j], new_v, 0, kInGpuHbm);
        if (j > 1) {
          gpu_sim.ReleaseMatrix(temp_v);
        }
        temp_v = new_v;
      }
      v_concat = temp_v;
    }

    // Step 3: Move Q and K_concat to SRAM
    Matrix* q_sram = matrix_memory_allocator.Allocate("q_sram");
    Matrix* k_sram = matrix_memory_allocator.Allocate("k_sram");
    gpu_sim.Copy(current_query, q_sram, kInGpuHbm);
    gpu_sim.Copy(k_concat, k_sram, kInGpuHbm);
    gpu_sim.MoveMatrixToSharedMem(q_sram);
    gpu_sim.MoveMatrixToSharedMem(k_sram);

    // Step 4: Transpose K in SRAM
    gpu_sim.Transpose(k_sram, kInSharedMemory);

    // Step 5: MatMul Q * K^T
    Matrix* qk_t = matrix_memory_allocator.Allocate("qk_t");
    gpu_sim.MatMul(q_sram, k_sram, qk_t);

    // Step 6: Release Q and K from SRAM to save space
    gpu_sim.ReleaseMatrix(q_sram);
    gpu_sim.ReleaseMatrix(k_sram);

    // Step 7: Compute Softmax for each row of QK^T
    size_t num_rows = i + 1;
    std::vector<Matrix*> softmax_rows;

    for (size_t r = 0; r < num_rows; ++r) {
      // Get row r
      Matrix* row = matrix_memory_allocator.Allocate("row_" + std::to_string(r));
      gpu_sim.GetRow(qk_t, r, row, kInSharedMemory);

      // Compute exp(row)
      Matrix* exp_row = matrix_memory_allocator.Allocate("exp_row_" + std::to_string(r));
      gpu_sim.MatExp(row, exp_row);

      // Compute sum of exp_row
      Matrix* sum_exp = matrix_memory_allocator.Allocate("sum_exp_" + std::to_string(r));
      gpu_sim.Sum(exp_row, sum_exp);

      // Divide exp_row by sum_exp
      Matrix* softmax_row = matrix_memory_allocator.Allocate("softmax_row_" + std::to_string(r));
      gpu_sim.MatDiv(exp_row, sum_exp, softmax_row);

      softmax_rows.push_back(softmax_row);

      // Release intermediate matrices
      gpu_sim.ReleaseMatrix(row);
      gpu_sim.ReleaseMatrix(exp_row);
      gpu_sim.ReleaseMatrix(sum_exp);
    }

    // Step 8: Concatenate softmax rows to form softmax_matrix
    Matrix* softmax_matrix = nullptr;
    if (num_rows == 1) {
      softmax_matrix = softmax_rows[0];
    } else {
      Matrix* temp_softmax = softmax_rows[0];
      for (size_t r = 1; r < num_rows; ++r) {
        Matrix* new_softmax = matrix_memory_allocator.Allocate("softmax_matrix_" + std::to_string(r));
        gpu_sim.Concat(temp_softmax, softmax_rows[r], new_softmax, 0, kInSharedMemory);
        if (r > 1 || temp_softmax != softmax_rows[0]) {
          gpu_sim.ReleaseMatrix(temp_softmax);
        }
        temp_softmax = new_softmax;
      }
      softmax_matrix = temp_softmax;
    }

    // Release QK^T
    gpu_sim.ReleaseMatrix(qk_t);

    // Step 9: Move V_concat to SRAM
    Matrix* v_sram = matrix_memory_allocator.Allocate("v_sram");
    gpu_sim.Copy(v_concat, v_sram, kInGpuHbm);
    gpu_sim.MoveMatrixToSharedMem(v_sram);

    // Step 10: MatMul softmax_matrix * V_concat
    Matrix* result_sram = matrix_memory_allocator.Allocate("result_sram");
    gpu_sim.MatMul(softmax_matrix, v_sram, result_sram);

    // Step 11: Move result to HBM
    Matrix* result_hbm = matrix_memory_allocator.Allocate("result_hbm");
    gpu_sim.Copy(result_sram, result_hbm, kInSharedMemory);
    gpu_sim.MoveMatrixToGpuHbm(result_hbm);

    // Run the simulator
    gpu_sim.Run(false, &matrix_memory_allocator);

    // Commit the answer
    rater.CommitAnswer(*result_hbm);
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu
