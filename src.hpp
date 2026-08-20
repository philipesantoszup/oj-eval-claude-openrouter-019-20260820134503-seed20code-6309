#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());

  // Running concatenations in SRAM for maximum efficiency
  // - k_t_concat_sram: K^T concatenated horizontally (512 x m)
  // - v_concat_sram: V concatenated vertically (m x 512)
  Matrix* k_t_concat_sram = nullptr;
  Matrix* v_concat_sram = nullptr;

  // Pre-fetch and pre-transpose K[0] and V[0] before loop starts
  Matrix* next_k_t_sram = nullptr;
  Matrix* next_v_sram = nullptr;
  if (keys.size() > 0) {
    next_k_t_sram = matrix_memory_allocator.Allocate("next_k_t_sram");
    next_v_sram = matrix_memory_allocator.Allocate("next_v_sram");
    gpu_sim.Copy(keys[0], next_k_t_sram, kInGpuHbm);
    gpu_sim.Copy(values[0], next_v_sram, kInGpuHbm);
    gpu_sim.MoveMatrixToSharedMem(next_k_t_sram);
    gpu_sim.MoveMatrixToSharedMem(next_v_sram);
    // Transpose K[0] from 1x512 to 512x1
    gpu_sim.Transpose(next_k_t_sram, kInSharedMemory);
  }

  for (size_t i = 0; i < keys.size(); ++i) {
    auto current_query = rater.GetNextQuery();
    size_t num_rows = i + 1;

    // Use prefetched and pre-transposed matrices
    Matrix* k_i_t_sram = next_k_t_sram;
    Matrix* v_i_sram = next_v_sram;

    // Update running concatenations in SRAM (much faster than HBM)
    if (i == 0) {
      k_t_concat_sram = k_i_t_sram;
      v_concat_sram = v_i_sram;
    } else {
      // Concatenate K^T horizontally (axis=1) and V vertically (axis=0)
      Matrix* new_k_t_sram = matrix_memory_allocator.Allocate("new_k_t_sram");
      Matrix* new_v_sram = matrix_memory_allocator.Allocate("new_v_sram");
      gpu_sim.Concat(k_t_concat_sram, k_i_t_sram, new_k_t_sram, 1, kInSharedMemory);
      gpu_sim.Concat(v_concat_sram, v_i_sram, new_v_sram, 0, kInSharedMemory);
      gpu_sim.ReleaseMatrix(k_t_concat_sram);
      gpu_sim.ReleaseMatrix(v_concat_sram);
      gpu_sim.ReleaseMatrix(k_i_t_sram);
      gpu_sim.ReleaseMatrix(v_i_sram);
      k_t_concat_sram = new_k_t_sram;
      v_concat_sram = new_v_sram;
    }

    // Pre-fetch, pre-transpose next K and pre-fetch next V (if not last round)
    // This overlaps IO operations with subsequent calculation operations!
    if (i + 1 < keys.size()) {
      next_k_t_sram = matrix_memory_allocator.Allocate("next_k_t_sram_" + std::to_string(i+1));
      next_v_sram = matrix_memory_allocator.Allocate("next_v_sram_" + std::to_string(i+1));
      gpu_sim.Copy(keys[i+1], next_k_t_sram, kInGpuHbm);
      gpu_sim.Copy(values[i+1], next_v_sram, kInGpuHbm);
      gpu_sim.MoveMatrixToSharedMem(next_k_t_sram);
      gpu_sim.MoveMatrixToSharedMem(next_v_sram);
      // Transpose K[i+1] from 1x512 to 512x1
      gpu_sim.Transpose(next_k_t_sram, kInSharedMemory);
    } else {
      next_k_t_sram = nullptr;
      next_v_sram = nullptr;
    }

    // Step 1: Move Q to SRAM
    Matrix* q_sram = matrix_memory_allocator.Allocate("q_sram");
    gpu_sim.Copy(current_query, q_sram, kInGpuHbm);
    gpu_sim.MoveMatrixToSharedMem(q_sram);

    // Step 2: MatMul Q * K^T (no need to transpose K anymore - we already have K^T!)
    Matrix* qk_t = matrix_memory_allocator.Allocate("qk_t");
    gpu_sim.MatMul(q_sram, k_t_concat_sram, qk_t);

    // Release Q early to save SRAM
    gpu_sim.ReleaseMatrix(q_sram);

    // Step 3: Compute Softmax for each row
    std::vector<Matrix*> softmax_rows;
    for (size_t r = 0; r < num_rows; ++r) {
      Matrix* row = matrix_memory_allocator.Allocate("row_" + std::to_string(r));
      gpu_sim.GetRow(qk_t, r, row, kInSharedMemory);

      Matrix* exp_row = matrix_memory_allocator.Allocate("exp_row_" + std::to_string(r));
      gpu_sim.MatExp(row, exp_row);

      Matrix* sum_exp = matrix_memory_allocator.Allocate("sum_exp_" + std::to_string(r));
      gpu_sim.Sum(exp_row, sum_exp);

      Matrix* sm_row = matrix_memory_allocator.Allocate("sm_row_" + std::to_string(r));
      gpu_sim.MatDiv(exp_row, sum_exp, sm_row);

      softmax_rows.push_back(sm_row);

      gpu_sim.ReleaseMatrix(row);
      gpu_sim.ReleaseMatrix(exp_row);
      gpu_sim.ReleaseMatrix(sum_exp);
    }

    // Step 4: Concatenate softmax rows into a single matrix
    Matrix* sm_matrix = nullptr;
    if (num_rows == 1) {
      sm_matrix = softmax_rows[0];
    } else {
      Matrix* temp_sm = softmax_rows[0];
      for (size_t r = 1; r < num_rows; ++r) {
        Matrix* new_sm = matrix_memory_allocator.Allocate("new_sm_" + std::to_string(r));
        gpu_sim.Concat(temp_sm, softmax_rows[r], new_sm, 0, kInSharedMemory);
        if (r > 1 || temp_sm != softmax_rows[0]) {
          gpu_sim.ReleaseMatrix(temp_sm);
        }
        temp_sm = new_sm;
      }
      sm_matrix = temp_sm;
    }

    // Release QK^T to save SRAM
    gpu_sim.ReleaseMatrix(qk_t);

    // Step 5: MatMul softmax matrix * V (V is already in SRAM!)
    Matrix* result_sram = matrix_memory_allocator.Allocate("result_sram");
    gpu_sim.MatMul(sm_matrix, v_concat_sram, result_sram);

    // Release softmax matrix to save SRAM
    gpu_sim.ReleaseMatrix(sm_matrix);

    // Step 6: Move result to HBM
    Matrix* result_hbm = matrix_memory_allocator.Allocate("result_hbm");
    gpu_sim.Copy(result_sram, result_hbm, kInSharedMemory);
    gpu_sim.MoveMatrixToGpuHbm(result_hbm);

    // Release result_sram
    gpu_sim.ReleaseMatrix(result_sram);

    // Run the simulator to execute all queued instructions
    gpu_sim.Run(false, &matrix_memory_allocator);

    // Commit the answer (must be in HBM)
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
