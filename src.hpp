#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());

  // Keep individual K and V matrices in SRAM (each is only 512 elements!)
  // This saves time on data movement and allows fast concatenation in SRAM
  std::vector<Matrix*> k_sram_list;
  std::vector<Matrix*> v_sram_list;

  for (size_t i = 0; i < keys.size(); ++i) {
    auto current_query = rater.GetNextQuery();
    size_t num_rows = i + 1;

    // Move current K[i] and V[i] to SRAM and store in our lists
    Matrix* k_i_sram = matrix_memory_allocator.Allocate("k_i_sram_" + std::to_string(i));
    Matrix* v_i_sram = matrix_memory_allocator.Allocate("v_i_sram_" + std::to_string(i));
    gpu_sim.Copy(keys[i], k_i_sram, kInGpuHbm);
    gpu_sim.Copy(values[i], v_i_sram, kInGpuHbm);
    gpu_sim.MoveMatrixToSharedMem(k_i_sram);
    gpu_sim.MoveMatrixToSharedMem(v_i_sram);
    k_sram_list.push_back(k_i_sram);
    v_sram_list.push_back(v_i_sram);

    // Step 1: Move Q to SRAM
    Matrix* q_sram = matrix_memory_allocator.Allocate("q_sram");
    gpu_sim.Copy(current_query, q_sram, kInGpuHbm);
    gpu_sim.MoveMatrixToSharedMem(q_sram);

    // Step 2: Concatenate K[0..i] in SRAM (fast!)
    Matrix* k_concat_sram = nullptr;
    if (num_rows == 1) {
      k_concat_sram = matrix_memory_allocator.Allocate("k_concat_sram");
      gpu_sim.Copy(k_sram_list[0], k_concat_sram, kInSharedMemory);
    } else {
      Matrix* temp_k = matrix_memory_allocator.Allocate("temp_k");
      gpu_sim.Copy(k_sram_list[0], temp_k, kInSharedMemory);
      for (size_t j = 1; j < num_rows; ++j) {
        Matrix* new_k = matrix_memory_allocator.Allocate("new_k_" + std::to_string(j));
        gpu_sim.Concat(temp_k, k_sram_list[j], new_k, 0, kInSharedMemory);
        if (j > 1) {
          gpu_sim.ReleaseMatrix(temp_k);
        }
        temp_k = new_k;
      }
      k_concat_sram = temp_k;
    }

    // Step 3: Transpose K in SRAM
    gpu_sim.Transpose(k_concat_sram, kInSharedMemory);

    // Step 4: MatMul Q * K^T
    Matrix* qk_t = matrix_memory_allocator.Allocate("qk_t");
    gpu_sim.MatMul(q_sram, k_concat_sram, qk_t);

    // Release Q and K early to save SRAM
    gpu_sim.ReleaseMatrix(q_sram);
    gpu_sim.ReleaseMatrix(k_concat_sram);

    // Step 5: Compute Softmax for each row
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

    // Step 6: Concatenate softmax rows
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

    // Release QK^T
    gpu_sim.ReleaseMatrix(qk_t);

    // Step 7: Concatenate V[0..i] in SRAM (fast!)
    Matrix* v_concat_sram = nullptr;
    if (num_rows == 1) {
      v_concat_sram = matrix_memory_allocator.Allocate("v_concat_sram");
      gpu_sim.Copy(v_sram_list[0], v_concat_sram, kInSharedMemory);
    } else {
      Matrix* temp_v = matrix_memory_allocator.Allocate("temp_v");
      gpu_sim.Copy(v_sram_list[0], temp_v, kInSharedMemory);
      for (size_t j = 1; j < num_rows; ++j) {
        Matrix* new_v = matrix_memory_allocator.Allocate("new_v_" + std::to_string(j));
        gpu_sim.Concat(temp_v, v_sram_list[j], new_v, 0, kInSharedMemory);
        if (j > 1) {
          gpu_sim.ReleaseMatrix(temp_v);
        }
        temp_v = new_v;
      }
      v_concat_sram = temp_v;
    }

    // Step 8: MatMul softmax matrix * V
    Matrix* result_sram = matrix_memory_allocator.Allocate("result_sram");
    gpu_sim.MatMul(sm_matrix, v_concat_sram, result_sram);

    // Release softmax matrix and V
    gpu_sim.ReleaseMatrix(sm_matrix);
    gpu_sim.ReleaseMatrix(v_concat_sram);

    // Step 9: Move result to HBM
    Matrix* result_hbm = matrix_memory_allocator.Allocate("result_hbm");
    gpu_sim.Copy(result_sram, result_hbm, kInSharedMemory);
    gpu_sim.MoveMatrixToGpuHbm(result_hbm);

    // Release result_sram
    gpu_sim.ReleaseMatrix(result_sram);

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
