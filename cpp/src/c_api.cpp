#include <cstdint>
#include <memory>

#include "tfm/model.hpp"
#include "tfm/weights.hpp"

// C ABI for use from ctypes (backend) and from the mobile runtime.
//   TfmModel* m;
//   int rc = tfm_load("model.bin", &m);
//   rc = tfm_forward(m, board, mask, n, &value, policy);
//   tfm_free(m);
extern "C" {

typedef struct TfmModel TfmModel;
struct TfmModel {
  std::unique_ptr<tfm::Model> model;
};

int tfm_load(const char* path, TfmModel** out) {
  if (!path || !out) return -1;
  std::unique_ptr<TfmModel> m(new TfmModel);
  m->model.reset(new tfm::Model);
  std::string err;
  if (!tfm::loadWeights(path, *m->model, &err)) return -2;
  *out = m.release();
  return 0;
}

void tfm_free(TfmModel* m) {
  delete m;
}

// board: int[n^3] tokens 0/1/2; mask: uint8[n^3], 1 = legal.
// value_out: 1 float; policy_out: n^3 floats (-inf where masked).
// Thread-safe: forward allocates only locals.
int tfm_forward(const TfmModel* m, const int* board, const uint8_t* mask,
                int n, float* value_out, float* policy_out) {
  if (!m || !m->model || !board || !mask || !value_out || !policy_out || n < 1)
    return -1;
  m->model->forward(board, mask, n, *value_out, policy_out);
  return 0;
}

// Convenience: number of params (sanity / telemetry).
int tfm_numel(const TfmModel* m) {
  return m && m->model ? m->model->numel() : -1;
}

}  // extern "C"