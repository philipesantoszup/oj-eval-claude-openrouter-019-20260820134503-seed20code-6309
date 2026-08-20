#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());

  // Running concatenations of K and V in HBM (HBM usage is not penalized!)
  Matrix* k_concat_hbm = nullptr;
  Matrix* v_concat_hbm = nullptr;

  for (size_t i = 0; i < keys.size(); ++i) {
    auto current_query = rater.GetNextQuery();
    size_t num_rows = i + 1;

    // Update running concatenations in HBM (cheap and doesn't use SRAM)
    if (i == 0) {
      k_concat_hbm = matrix_memory_allocator.Allocate("k_concat_hbm");
      v_concat_hbm = matrix_memory_allocator.Allocate("v_concat_hbm");
      gpu_sim.Copy(keys[0], k_concat_hbm, kInGpuHbm);
      gpu_sim.Copy(values[0], v_concat_hbm, kInGpuHbm);
    } else {
      Matrix* new_k = matrix_memory_allocator.Allocate("new_k");
      Matrix* new_v = matrix_memory_allocator.Allocate("new_v");
      gpu_sim.Concat(k_concat_hbm, keys[i], new_k, 0, kInGpuHbm);
      gpu_sim.Concat(v_concat_hbm, values[i], new_v, 0, kInGpuHbm);
      gpu_sim.ReleaseMatrix(k_concat_hbm);
      gpu_sim.ReleaseMatrix(v_concat_hbm);
      k_concat_hbm = new_k;
      v_concat_hbm = new_v;
    }

    // Step 1: Move Q and K to SRAM (only when needed)
    Matrix* q_sram = matrix_memory_allocator.Allocate("q_sram");
    Matrix* k_sram = matrix_memory_allocator.Allocate("k_sram");
    gpu_sim.Copy(current_query, q_sram, kInGpuHbm);
    gpu_sim.Copy(k_concat_hbm, k_sram, kInGpuHbm);
    gpu_sim.MoveMatrixToSharedMem(q_sram);
    gpu_sim.MoveMatrixToSharedMem(k_sram);

    // Step 2: Transpose K in SRAM
    gpu_sim.Transpose(k_sram, kInSharedMemory);

    // Step 3: MatMul Q * K^T
    Matrix* qk_t = matrix_memory_allocator.Allocate("qk_t");
    gpu_sim.MatMul(q_sram, k_sram, qk_t);

    // Release Q and K immediately after use to minimize SRAM peak usage!
    gpu_sim.ReleaseMatrix(q_sram);
    gpu_sim.ReleaseMatrix(k_sram);

    // Step 4: Compute Softmax for each row
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

      // Release intermediate matrices immediately
      gpu_sim.ReleaseMatrix(row);
      gpu_sim.ReleaseMatrix(exp_row);
      gpu_sim.ReleaseMatrix(sum_exp);
    }

    // Step 5: Concatenate softmax rows
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

    // Release QK^T immediately after use
    gpu_sim.ReleaseMatrix(qk_t);

    // Step 6: Move V to SRAM
    Matrix* v_sram = matrix_memory_allocator.Allocate("v_sram");
    gpu_sim.Copy(v_concat_hbm, v_sram, kInGpuHbm);
    gpu_sim.MoveMatrixToSharedMem(v_sram);

    // Step 7: MatMul softmax matrix * V
    Matrix* result_sram = matrix_memory_allocator.Allocate("result_sram");
    gpu_sim.MatMul(sm_matrix, v_sram, result_sram);

    // Release softmax matrix and V immediately
    gpu_sim.ReleaseMatrix(sm_matrix);
    gpu_sim.ReleaseMatrix(v_sram);

    // Step 8: Move result to HBM
    Matrix* result_hbm = matrix_memory_allocator.Allocate("result_hbm");
    gpu_sim.Copy(result_sram, result_hbm, kInSharedMemory);
    gpu_sim.MoveMatrixToGpuHbm(result_hbm);

    // Release result_sram immediately
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
