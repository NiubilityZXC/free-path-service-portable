#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define MAXZ 118
#define MAXN 7
#define MAX_ORBITALS 19
#define PI 3.141592653589793238462643383279502884

struct Params {
  int zan;
  int n;
  int s_ini;
  double tep, rod, aw, ni, ne, tgama;
  double lbfac, fffac, bffac, bbfac, esfac;
  double emiff_pre2;
  double sig[MAXN + 1][MAXN + 1];
  double fnm[MAXN + 1][MAXN + 1];
  double nsp[MAXZ + 1];
  double ns[MAXZ + 1];
  double qnout[MAXZ + 1];
  double enout[MAXZ + 1];
  int pn[MAXZ + 1][MAXN + 1];
  int nout[MAXZ + 1];
  int nouts1[MAXZ + 1];
  int pn_nss1[MAXZ + 1];
  double qnn[MAXZ + 1][MAXN + 1];
  double enn[MAXZ + 1][MAXN + 1];
};

static Params h_params;
static int h_configured = 0;
static int d_ready = 0;
static int d_capacity = 0;
static int d_device = -1;
static double *d_x = nullptr;
static double *d_f = nullptr;

__constant__ Params c_params;

static int s_ini_for_zan(int zan) {
  switch (zan) {
    case 4: return 0;
    case 6: return 0;
    case 13: return 0;
    case 22: return 2;
    case 26: return 2;
    case 29: return 0;
    case 42: return 1;
    case 50: return 4;
    case 56: return 10;
    case 63: return 10;
    case 74: return 2;
    case 79: return 1;
    case 82: return 4;
    case 92: return 11;
    default: return 0;
  }
}

static void remove_electrons(int elec[MAX_ORBITALS + 1], int s,
                             const int layer_count[MAXN + 1],
                             const int layer_list[MAXN + 1][11]) {
  int remain = s;
  for (int ns = MAXN; ns >= 1; --ns) {
    if (remain <= 0) return;
    for (int i = 1; i <= layer_count[ns]; ++i) {
      const int idx = layer_list[ns][i];
      if (elec[idx] > 0) {
        const int rem = remain < elec[idx] ? remain : elec[idx];
        elec[idx] -= rem;
        remain -= rem;
        if (remain == 0) return;
      }
    }
  }
}

static void pn_ns_z2(int zan, int n, int chrem, int pn[MAXN + 1],
                     int *nout, int *nouts1, int *pn_nss1) {
  const int orb_n[MAX_ORBITALS + 1] =
      {0, 1, 2, 2, 3, 3, 4, 3, 4, 5, 4, 5, 6, 4, 5, 6, 7, 5, 6, 7};
  const int orb_cap[MAX_ORBITALS + 1] =
      {0, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 14, 10, 6};

  int layer_count[MAXN + 1] = {0};
  int layer_list[MAXN + 1][11] = {{0}};
  layer_count[1] = 1; layer_list[1][1] = 1;
  layer_count[2] = 2; layer_list[2][1] = 3; layer_list[2][2] = 2;
  layer_count[3] = 3; layer_list[3][1] = 7; layer_list[3][2] = 5; layer_list[3][3] = 4;
  layer_count[4] = 4; layer_list[4][1] = 13; layer_list[4][2] = 10; layer_list[4][3] = 8; layer_list[4][4] = 6;
  layer_count[5] = 4; layer_list[5][1] = 17; layer_list[5][2] = 14; layer_list[5][3] = 11; layer_list[5][4] = 9;
  layer_count[6] = 3; layer_list[6][1] = 18; layer_list[6][2] = 15; layer_list[6][3] = 12;
  layer_count[7] = 2; layer_list[7][1] = 19; layer_list[7][2] = 16;

  int elec_neutral[MAX_ORBITALS + 1] = {0};
  int rem = zan;
  for (int i = 1; i <= MAX_ORBITALS; ++i) {
    if (rem <= 0) break;
    elec_neutral[i] = rem < orb_cap[i] ? rem : orb_cap[i];
    rem -= elec_neutral[i];
  }

  switch (zan) {
    case 24: elec_neutral[6] -= 1; elec_neutral[7] += 1; break;
    case 29: elec_neutral[6] -= 1; elec_neutral[7] += 1; break;
    case 41: elec_neutral[9] -= 1; elec_neutral[10] += 1; break;
    case 42: elec_neutral[9] -= 1; elec_neutral[10] += 1; break;
    case 44: elec_neutral[9] -= 1; elec_neutral[10] += 1; break;
    case 45: elec_neutral[9] -= 1; elec_neutral[10] += 1; break;
    case 46: elec_neutral[9] -= 2; elec_neutral[10] += 2; break;
    case 47: elec_neutral[9] -= 1; elec_neutral[10] += 1; break;
    case 57: elec_neutral[13] -= 1; elec_neutral[14] += 1; break;
    case 58: elec_neutral[13] -= 1; elec_neutral[14] += 1; break;
    case 64: elec_neutral[13] -= 1; elec_neutral[14] += 1; break;
    case 78: elec_neutral[12] -= 1; elec_neutral[14] += 1; break;
    case 79: elec_neutral[12] -= 1; elec_neutral[14] += 1; break;
    case 89: elec_neutral[17] -= 1; elec_neutral[18] += 1; break;
    case 90: elec_neutral[17] -= 2; elec_neutral[18] += 2; break;
    case 91: elec_neutral[17] -= 1; elec_neutral[18] += 1; break;
    case 92: elec_neutral[17] -= 1; elec_neutral[18] += 1; break;
    case 93: elec_neutral[17] -= 1; elec_neutral[18] += 1; break;
    case 96: elec_neutral[17] -= 1; elec_neutral[18] += 1; break;
  }

  const int s = zan - chrem;
  int elec_ion[MAX_ORBITALS + 1];
  std::memcpy(elec_ion, elec_neutral, sizeof(elec_ion));
  remove_electrons(elec_ion, s, layer_count, layer_list);

  for (int i = 0; i <= MAXN; ++i) pn[i] = 0;
  *nout = 0;
  for (int i = 1; i <= MAX_ORBITALS; ++i) {
    const int j = orb_n[i];
    if (j <= MAXN) pn[j] += elec_ion[i];
    if (elec_ion[i] > 0 && j > *nout) *nout = j;
  }
  if (s == zan) *nout = 1;
  *nouts1 = *nout;
  *pn_nss1 = pn[*nout] - 1;
  if (*pn_nss1 < 0) *pn_nss1 = 0;
}

static void precompute_enn(int idx) {
  Params &p = h_params;
  const double sbar = p.ne / p.ni;
  const double del_en = 36.0 * sbar * std::pow(p.rod / p.aw, 1.0 / 3.0);

  for (int i = 1; i <= p.n; ++i) {
    double pmsig = 0.0;
    for (int j = 1; j <= i; ++j) {
      if (j != i) pmsig += p.pn[idx][j] * p.sig[i][j];
    }
    pmsig += p.pn[idx][i] * p.sig[i][i] * 0.5;
    p.qnn[idx][i] = p.zan - pmsig;
    const double enn_pre = 13.6 * p.qnn[idx][i] * p.qnn[idx][i] / (double)(i * i);
    p.enn[idx][i] = std::fabs(enn_pre - del_en);
  }
}

extern "C" int ross_gpu_init(int zan, int n, double tep, double rod, double aw,
	                             double ni, const double *sig, const double *fnm,
	                             const double *nsp, const double *qnout,
	                             const double *enout, double ne, double tgama,
	                             double lbfac, double fffac, double bffac,
	                             double bbfac, double esfac) {
  const char *use_gpu = std::getenv("ROSS_USE_GPU");
  if (use_gpu && std::strcmp(use_gpu, "0") == 0) {
    h_configured = 0;
    return 0;
  }

  std::memset(&h_params, 0, sizeof(h_params));
  h_params.zan = zan;
  h_params.n = n;
  h_params.s_ini = s_ini_for_zan(zan);
  h_params.tep = tep;
  h_params.rod = rod;
  h_params.aw = aw;
  h_params.ni = ni;
	  h_params.ne = ne;
	  h_params.tgama = tgama;
	  h_params.lbfac = lbfac;
	  h_params.fffac = fffac;
	  h_params.bffac = bffac;
	  h_params.bbfac = bbfac;
	  h_params.esfac = esfac;

  for (int col = 1; col <= n; ++col) {
    for (int row = 1; row <= n; ++row) {
      h_params.sig[row][col] = sig[(row - 1) + (col - 1) * n];
    }
  }
  for (int col = 1; col <= n; ++col) {
    for (int row = 1; row <= n - 1; ++row) {
      h_params.fnm[row][col] = fnm[(row - 1) + (col - 1) * (n - 1)];
    }
  }
  for (int i = 0; i <= zan; ++i) {
    h_params.nsp[i] = nsp[i];
    h_params.ns[i] = nsp[i] * ni;
    h_params.qnout[i] = qnout[i];
    h_params.enout[i] = enout[i];
    h_params.emiff_pre2 += h_params.ns[i] * (double)(i * i);
    pn_ns_z2(zan, n, zan - i, h_params.pn[i], &h_params.nout[i],
             &h_params.nouts1[i], &h_params.pn_nss1[i]);
    precompute_enn(i);
  }

  h_configured = 1;
  d_ready = 0;
  d_capacity = 0;
  d_x = nullptr;
  d_f = nullptr;
  return 1;
}

static int ensure_cuda(int nvec, int core) {
  if (!h_configured) return 0;
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess || device_count < 1) {
    std::fprintf(stderr, "[gpu] cudaGetDeviceCount failed: %s\n",
                 cudaGetErrorString(err));
    return 0;
  }
  int device = 0;
  if (core < 0) device = (-core - 1) % device_count;
  if (!d_ready) {
    err = cudaSetDevice(device);
    if (err != cudaSuccess) {
      std::fprintf(stderr, "[gpu] cudaSetDevice failed: %s\n", cudaGetErrorString(err));
      return 0;
    }
    err = cudaMemcpyToSymbol(c_params, &h_params, sizeof(Params));
    if (err != cudaSuccess) {
      std::fprintf(stderr, "[gpu] cudaMemcpyToSymbol failed: %s\n", cudaGetErrorString(err));
      return 0;
    }
    d_ready = 1;
    d_device = device;
  } else if (d_device != device) {
    err = cudaSetDevice(d_device);
    if (err != cudaSuccess) {
      std::fprintf(stderr, "[gpu] cudaSetDevice reuse failed: %s\n",
                   cudaGetErrorString(err));
      return 0;
    }
  }
  if (nvec > d_capacity) {
    if (d_x) cudaFree(d_x);
    if (d_f) cudaFree(d_f);
    cudaError_t err = cudaMalloc((void **)&d_x, (size_t)nvec * sizeof(double));
    if (err != cudaSuccess) {
      std::fprintf(stderr, "[gpu] cudaMalloc x failed: %s\n", cudaGetErrorString(err));
      d_x = nullptr; d_f = nullptr; d_capacity = 0;
      return 0;
    }
    err = cudaMalloc((void **)&d_f, (size_t)nvec * sizeof(double));
    if (err != cudaSuccess) {
      std::fprintf(stderr, "[gpu] cudaMalloc f failed: %s\n", cudaGetErrorString(err));
      cudaFree(d_x); d_x = nullptr; d_f = nullptr; d_capacity = 0;
      return 0;
    }
    d_capacity = nvec;
  }
  return 1;
}

__device__ double pow_int(double x, int n) {
  double y = 1.0;
  for (int i = 0; i < n; ++i) y *= x;
  return y;
}

__device__ double eval_one(double x1) {
  const Params &p = c_params;
  const double ue = x1 / (1.0 - x1);
  const double slo_ue = 1.0 / ((1.0 - x1) * (1.0 - x1));
  const double hplnu = ue * p.tep;
  const double x = hplnu / p.tep;
  const double exp_neg_x = exp(-x);
  const double exp_x = exp(x);

  const double emiff_pre1 = 5.0e-41 * p.ne / sqrt(p.tep) * exp_neg_x;
  const double emiff = emiff_pre1 * p.emiff_pre2;
  const double inu_p = 2.07e-4 * hplnu * hplnu * hplnu / (exp_x - 1.0);
  const double absff = emiff / (p.rod * inu_p);

  const double absbf_pre1 = 1.2e10 / p.aw * (1.0 - exp_neg_x);
  double absbf_pre2 = 0.0;
  const double inv_hplnu3 = 1.0 / (hplnu * hplnu * hplnu);
  for (int i = p.s_ini; i <= p.zan; ++i) {
    double absbf_pre3 = 0.0;
    for (int j = 1; j <= p.n; ++j) {
      if (hplnu >= p.enn[i][j]) {
        absbf_pre3 += pow_int(p.qnn[i][j], 4) * p.pn[i][j] /
                      (pow((double)j, 5.0)) * inv_hplnu3;
      }
    }
    absbf_pre2 += p.nsp[i] * absbf_pre3;
  }
  const double absbf = absbf_pre1 * absbf_pre2;

  double lsum = 0.0;
  for (int i = p.s_ini; i <= p.zan; ++i) {
    double lsum_pre = 0.0;
    for (int j = p.nout[i] - 1; j <= p.nout[i]; ++j) {
      if (j < 1) continue;
      if (j == p.n) break;

      const double q = p.qnn[i][j];
      double wlb = 0.637e-22 * p.ne / sqrt(p.tep) *
                   (5.0 * j * j / (q * q) *
                    (((i + 1.0) * (i + 1.0) * j * j / (q * q)) - 1.0));
      wlb *= p.lbfac;

      for (int k = j + 1; k <= p.n; ++k) {
        const double ffnm = p.fnm[j][k] * p.pn[i][j] *
                            (1.0 - p.pn[i][k] / (2.0 * k * k));
        const double del_chi = p.enn[i][j] - p.enn[i][k];
        const double denom = (del_chi - hplnu) * (del_chi - hplnu) + wlb * wlb;
        const double llp = (1.0 / PI) * wlb / denom;
        lsum_pre += ffnm * llp;
      }
    }
    lsum += p.nsp[i] * lsum_pre;
  }

  const double absbb = 6.6e7 * (1.0 - exp_neg_x) / p.aw * lsum;

	  const double abses = 0.4 * p.ne / p.ni / p.aw;
	  const double abstot = p.fffac * absff + p.bffac * absbf +
	                        p.bbfac * absbb + p.esfac * abses;
	  const double y = (p.tep / p.tgama) * ue;
	  const double exp_y = exp(y);
	  const double gru2 = 15.0 / (4.0 * PI * PI * PI * PI) *
	                      pow(p.tep / p.tgama, 5.0) *
	                      ue * ue * ue * ue * exp_y /
	                      ((exp_y - 1.0) * (exp_y - 1.0));
	  return (1.0 / abstot) * gru2 * slo_ue;
	}

__global__ void eval_kernel(int n, const double *x, double *f) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  for (int i = tid; i < n; i += stride) {
    f[i] = eval_one(x[i]);
  }
}

extern "C" int ross_gpu_eval(int nvec, const double *x, double *f, int core) {
  if (nvec <= 0) return 1;
  if (!ensure_cuda(nvec, core)) return 0;

  cudaError_t err = cudaMemcpy(d_x, x, (size_t)nvec * sizeof(double),
                               cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    std::fprintf(stderr, "[gpu] cudaMemcpy H2D failed: %s\n", cudaGetErrorString(err));
    return 0;
  }

  const int block = 256;
  int grid = (nvec + block - 1) / block;
  if (grid > 4096) grid = 4096;
  eval_kernel<<<grid, block>>>(nvec, d_x, d_f);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::fprintf(stderr, "[gpu] kernel launch failed: %s\n", cudaGetErrorString(err));
    return 0;
  }

  err = cudaMemcpy(f, d_f, (size_t)nvec * sizeof(double), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    std::fprintf(stderr, "[gpu] cudaMemcpy D2H failed: %s\n", cudaGetErrorString(err));
    return 0;
  }
  return 1;
}
