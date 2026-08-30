// C bridge between Flutter (dart:ffi) and the ExecuTorch runtime.
//
// Loads mobile/assets/model.pte and exposes:
//   int  ttt_engine_init(const char* pte_path)
//   void ttt_engine_free()
//   int  ttt_forward(const int32_t* board, int n, float* value_out,
//                    float* policy_out)  // policy_out length n^3
//
// On iOS/Android this is compiled by the Flutter CMake/Gradle build using
// ExecuTorch's prebuilt libs. On Linux/macOS for local tests you can point
// CMake at an ExecuTorch source checkout (see README).

#include <cstring>
#include <memory>
#include <vector>

// --- ExecuTorch runtime includes (path depends on SDK layout) ---
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/runtime.h>

namespace et = ::executorch::runtime;
namespace aten = ::executorch::aten;

namespace {
et::Result<et::Method> g_method;
std::unique_ptr<et::Program> g_program;
}  // namespace

extern "C" {

int ttt_engine_init(const char* pte_path) {
  et::runtime_init();
  auto file = fopen(pte_path, "rb");
  if (!file) return -1;
  fseek(file, 0, SEEK_END);
  long size = ftell(file);
  fseek(file, 0, SEEK_SET);
  std::vector<uint8_t> data(size);
  if (fread(data.data(), 1, size, file) != (size_t)size) {
    fclose(file);
    return -2;
  }
  fclose(file);

  auto prog = et::Program::load(data.data(), data.size());
  if (!prog.ok()) return -3;
  g_program = std::make_unique<et::Program>(std::move(*prog));

  auto method = g_program->load_method("forward");
  if (!method.ok()) return -4;
  g_method = std::move(*method);
  return 0;
}

void ttt_engine_free() { g_method.free(); g_program.reset(); }

int ttt_forward(const int32_t* board, int n, float* value_out,
                float* policy_out) {
  if (!g_method.ok()) return -1;
  const int n_tokens = n * n * n;
  auto& method = *g_method;

  // inputs: board (int32 [1, n^3])
  auto board_t = method.get_inputs_data()[0];
  std::memcpy(board_t.mutable_data_ptr(), board, n_tokens * sizeof(int32_t));

  auto err = method.execute();
  if (err != et::Error::Ok) return -2;

  // outputs: [value, policy_logits]
  const auto outputs = method.get_outputs_data();
  if (outputs.size() < 2) return -3;

  *value_out = static_cast<const float*>(outputs[0].const_data_ptr())[0];
  const float* pol = static_cast<const float*>(outputs[1].const_data_ptr());
  std::memcpy(policy_out, pol, n_tokens * sizeof(float));
  return 0;
}

}  // extern "C"