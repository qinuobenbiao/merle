#include "moderngpu/context.cuh"
#include "wahGpu.cuh"
#include <array>
using merle = mgpu::mem_t<int>;
// merle, merleXfer, decode-and-query, dnqXfer
using casetime = std::array<double, 5>;

extern "C" {
#include <malloc.h>
// nvcc hates intrinsics. Don't include <roaring.h> so that nvcc won't see the
// SSE/AVX intrinsics Roaring uses.
struct roaring_bitmap_t { char opaque[40]; };
size_t bitset_extract_setbits(uint64_t *bitset, size_t length, void *vout,
                              uint32_t base);
roaring_bitmap_t *roaring_bitmap_of_ptr(size_t n_args, const uint32_t *vals);
roaring_bitmap_t *roaring_bitmap_and(const roaring_bitmap_t *x1,
                                     const roaring_bitmap_t *x2);
roaring_bitmap_t *roaring_bitmap_or(const roaring_bitmap_t *x1,
                                    const roaring_bitmap_t *x2);
void roaring_bitmap_and_inplace(roaring_bitmap_t *x1,
                                const roaring_bitmap_t *x2);
uint64_t roaring_bitmap_get_cardinality(const roaring_bitmap_t *ra);
bool roaring_bitmap_run_optimize(roaring_bitmap_t *r);
void roaring_bitmap_free(const roaring_bitmap_t *r);

// roaring_bitmap_t *roaring_bitmap_of_devbitset(const mgpu::mem_t<int>& mem) {
roaring_bitmap_t *rbod(const mgpu::mem_t<int>& mem) {
  std::vector<int> bitset_h = from_mem(mem);
  std::vector<uint32_t> arr(bitset_h.size() * 31);
  size_t nr = bitset_extract_setbits((uint64_t *)bitset_h.data(),
                                     bitset_h.size() / 2, arr.data(), 0);
  arr.resize(nr);
  roaring_bitmap_t* res = roaring_bitmap_of_ptr(nr, arr.data());
  (void)roaring_bitmap_run_optimize(res);
  return res;
}
} // extern "C"

static std::pair<merle, merle> // wah and excscan
loadCase(int caseNr, int listNr, bool dec, mgpu::context_t &ctx) {
  char path[256];
  sprintf(path, "wahData/st/%dl%d.wah", caseNr, listNr);
  wahHost_s *h = loadWahFile(path);
  merle d = merle(h->cmprsNrWord, ctx);
  mgpu::htod(d.data(), (int*)h->dat, d.size());
  merle sc = wahCntExcScan(d.data(), d.size(), ctx);
  if (dec)
    d = wahDecomp(d.data(), sc.data(), d.size(), ctx);
  free(h);
  return std::make_pair(std::move(d), std::move(sc));
}

casetime t6_17(mgpu::context_t &ctx, bool is17) {
  int caseNr = is17 ? 17 : 6;
  mgpu::mem_t<int> xferbuf(10 << 20, ctx, mgpu::memory_space_host);
  auto [l0d, l0sc] = loadCase(caseNr, 0, false, ctx);
  auto [l1d, l1sc] = loadCase(caseNr, 1, false, ctx);
  auto [l2d, l2sc] = loadCase(caseNr, 2, true, ctx);

  casetime ret{0, 0, 0, 0};
  for (size_t i = 0; i < 1000; ++i) {
    ctx.timer_begin();
    merle res = wahAndNo1(l1d.data(), l1sc.data(), l1d.size(), l0d.data(),
                          l0sc.data(), l0d.size(), ctx);
    wahEncNo1AndDec(res.data(), l0sc.data(), res.size(), l1d.data(),
                    l1d.size(), ctx);
    ret[0] += ctx.timer_end();
    ctx.timer_begin();
    res = wahCompact(res.data(), res.size(), ctx);
    mgpu::dtoh(xferbuf.data(), res.data(), res.size());
    ret[1] += ctx.timer_end();
  }

  ctx.timer_begin();
  l0d = wahDecomp(l0d.data(), l0sc.data(), l0d.size(), ctx);
  l1d = wahDecomp(l1d.data(), l1sc.data(), l1d.size(), ctx);
  ret[2] = ctx.timer_end() * 1000;
  for (size_t i = 0; i < 1000; ++i) {
    ctx.timer_begin();
    merle res = l2d.clone();
    wahAndExpl(res.data(), l0d.data(), res.size(), ctx);
    wahAndExpl(res.data(), l1d.data(), res.size(), ctx);
    ret[2] += ctx.timer_end();
    ctx.timer_begin();
    mgpu::dtoh(xferbuf.data(), res.data(), res.size());
    ret[3] += ctx.timer_end();
  }

  roaring_bitmap_t *l0r = rbod(l0d), *l1r = rbod(l1d), *l2r = rbod(l2d);
  ctx.timer_begin();
  for (size_t i = 0; i < 1000; i++) {
    roaring_bitmap_t *res = roaring_bitmap_and(l0r, l1r);
    roaring_bitmap_and_inplace(res, l2r);
    roaring_bitmap_free(res);
  }
  ret[4] = ctx.timer_end();
  roaring_bitmap_free(l0r); roaring_bitmap_free(l1r); roaring_bitmap_free(l2r);
  return ret;
}

casetime t3(mgpu::context_t &ctx) {
  mgpu::mem_t<int> xferbuf(10 << 20, ctx, mgpu::memory_space_host);
  auto [l0d, l0sc] = loadCase(3, 0, true, ctx);
  auto [l1d, l1sc] = loadCase(3, 1, false, ctx);
  auto [l2d, l2sc] = loadCase(3, 2, true, ctx);
  auto [l3d, l3sc] = loadCase(3, 3, false, ctx);
  auto [l4d, l4sc] = loadCase(3, 4, true, ctx);

  casetime ret{0, 0, 0, 0};
  for (size_t i = 0; i < 1000; ++i) {
    ctx.timer_begin();
    merle or01 = wahEncOpDec(l1d.data(), l1sc.data(), l1d.size(), l0d.data(),
                             l0d.size(), wahOp::OR, ctx);
    merle or23 = wahEncOpDec(l3d.data(), l3sc.data(), l3d.size(), l2d.data(),
                             l2d.size(), wahOp::OR, ctx);
    wahAndExpl(or01.data(), or23.data(), std::min(or01.size(), or23.size()), ctx);
    wahAndExpl(or01.data(), l4d.data(), std::min(or01.size(), l4d.size()), ctx);
    ret[0] += ctx.timer_end();
    ctx.timer_begin();
    mgpu::dtoh(xferbuf.data(), or01.data(), or01.size());
    ret[1] += ctx.timer_end();
  }

  l1d = wahDecomp(l1d.data(), l1sc.data(), l1d.size(), ctx);
  l3d = wahDecomp(l3d.data(), l3sc.data(), l3d.size(), ctx);
  for (size_t i = 0; i < 1000; ++i) {
    ctx.timer_begin();
    merle or01 = l1d.clone(), or23 = l3d.clone();
    wahOrExpl(or01.data(), l0d.data(), or01.size(), ctx);
    wahOrExpl(or23.data(), l2d.data(), or23.size(), ctx);
    wahAndExpl(or01.data(), or23.data(), or01.size(), ctx);
    wahAndExpl(or01.data(), l4d.data(), or01.size(), ctx);
    ret[2] += ctx.timer_end();
    ctx.timer_begin();
    mgpu::dtoh(xferbuf.data(), or01.data(), or01.size());
    ret[3] += ctx.timer_end();
  }

  roaring_bitmap_t *l0r = rbod(l0d), *l1r = rbod(l1d), *l2r = rbod(l2d),
                   *l3r = rbod(l3d), *l4r = rbod(l4d);
  ctx.timer_begin();
  for (size_t i = 0; i < 1000; i++) {
    roaring_bitmap_t *or01 = roaring_bitmap_or(l0r, l1r);
    roaring_bitmap_t *or23 = roaring_bitmap_or(l2r, l3r);
    roaring_bitmap_and_inplace(or01, l4r);
    roaring_bitmap_and_inplace(or01, or23);
    roaring_bitmap_free(or01); roaring_bitmap_free(or23);
  }
  ret[4] = ctx.timer_end();
  roaring_bitmap_free(l0r); roaring_bitmap_free(l1r); roaring_bitmap_free(l2r);
  roaring_bitmap_free(l3r); roaring_bitmap_free(l4r);
  return ret;
}

casetime t12(mgpu::context_t& ctx) {
  casetime ret{0, 0, 0, 0};
  mgpu::mem_t<int> xferbuf(10 << 20, ctx, mgpu::memory_space_host);
  auto [l0d, l0sc] = loadCase(12, 0, true, ctx);
  auto [l1d, l1sc] = loadCase(12, 1, true, ctx);
  auto [l2d, l2sc] = loadCase(12, 2, false, ctx);
  for (size_t i = 0; i < 1000; ++i) {
    ctx.timer_begin();
    merle or01 = l0d.clone(), res = l2d.clone();
    wahOrExpl(or01.data(), l1d.data(), std::min(l0d.size(), l1d.size()), ctx);
    wahEncNo1AndDec(res.data(), l2sc.data(), res.size(), or01.data(), or01.size(), ctx);
    ret[0] += ctx.timer_end();
    ctx.timer_begin();
    mgpu::dtoh(xferbuf.data(), res.data(), res.size());
    ret[1] += ctx.timer_end();
  }

  l2d = wahDecomp(l2d.data(), l2sc.data(), l2d.size(), ctx);
  for (size_t i = 0; i < 1000; ++i) {
    ctx.timer_begin();
    merle res = l0d.clone();
    wahOrExpl(res.data(), l1d.data(), res.size(), ctx);
    wahAndExpl(res.data(), l2d.data(), res.size(), ctx);
    ret[2] += ctx.timer_end();
    ctx.timer_begin();
    mgpu::dtoh(xferbuf.data(), res.data(), res.size());
    ret[3] += ctx.timer_end();
  }

  roaring_bitmap_t *l0r = rbod(l0d), *l1r = rbod(l1d), *l2r = rbod(l2d);
  ctx.timer_begin();
  for (size_t i = 0; i < 1000; i++) {
    roaring_bitmap_t *res = roaring_bitmap_or(l0r, l1r);
    roaring_bitmap_and_inplace(res, l2r);
    roaring_bitmap_free(res);
  }
  ret[4] = ctx.timer_end();
  roaring_bitmap_free(l0r); roaring_bitmap_free(l1r); roaring_bitmap_free(l2r);
  return ret;
}

casetime s12_13(mgpu::context_t& ctx, bool is3) {
  int caseNr = is3 ? -13 : -12;
  mgpu::mem_t<int> xferbuf(10 << 20, ctx, mgpu::memory_space_host);
  auto [l0d, l0sc] = loadCase(caseNr, 0, true, ctx);
  auto [l1d, l1sc] = loadCase(caseNr, 1, true, ctx);
  auto [l2d, l2sc] = loadCase(caseNr, 2, false, ctx);

  casetime ret{0, 0, 0, 0};
  for (size_t i = 0; i < 1000; ++i) {
    ctx.timer_begin();
    merle res = wahEncAndDec(l2d.data(), l2sc.data(), l2d.size(), l0d.data(),
                             l0d.size(), ctx);
    wahEncNo1AndDec(res.data(), l2sc.data(), res.size(), l1d.data(),
                    l1d.size(), ctx);
    ret[0] += ctx.timer_end();
    ctx.timer_begin();
    mgpu::dtoh(xferbuf.data(), res.data(), res.size());
    ret[1] += ctx.timer_end();
  }

  l2d = wahDecomp(l2d.data(), l2sc.data(), l2d.size(), ctx);
  for (size_t i = 0; i < 1000; ++i) {
    ctx.timer_begin();
    merle res = l2d.clone();
    wahAndExpl(res.data(), l0d.data(), res.size(), ctx);
    wahAndExpl(res.data(), l1d.data(), res.size(), ctx);
    ret[2] += ctx.timer_end();
    ctx.timer_begin();
    mgpu::dtoh(xferbuf.data(), res.data(), res.size());
    ret[3] += ctx.timer_end();
  }

  roaring_bitmap_t *l0r = rbod(l0d), *l1r = rbod(l1d), *l2r = rbod(l2d);
  ctx.timer_begin();
  for (size_t i = 0; i < 1000; i++) {
    roaring_bitmap_t *res = roaring_bitmap_and(l0r, l2r);
    roaring_bitmap_and_inplace(res, l1r);
    roaring_bitmap_free(res);
  }
  ret[4] = ctx.timer_end();
  roaring_bitmap_free(l0r); roaring_bitmap_free(l1r); roaring_bitmap_free(l2r);
  return ret;
}

casetime s23(mgpu::context_t& ctx) {
  casetime ret{0, 0, 0, 0};
  mgpu::mem_t<int> xferbuf(10 << 20, ctx, mgpu::memory_space_host);
  auto [l0d, l0sc] = loadCase(-23, 0, false, ctx);
  auto [l1d, l1sc] = loadCase(-23, 1, true, ctx);
  for (size_t i = 0; i < 1000; ++i) {
    ctx.timer_begin();
    merle res = wahEncAndDec(l0d.data(), l0sc.data(), l0d.size(), l1d.data(),
                             l1d.size(), ctx);
    ret[0] += ctx.timer_end();
    ctx.timer_begin();
    mgpu::dtoh(xferbuf.data(), res.data(), res.size());
    ret[1] += ctx.timer_end();
  }

  l0d = wahDecomp(l0d.data(), l0sc.data(), l0d.size(), ctx);
  for (size_t i = 0; i < 1000; ++i) {
    ctx.timer_begin();
    merle res = l1d.clone();
    wahAndExpl(res.data(), l0d.data(), res.size(), ctx);
    ret[2] += ctx.timer_end();
    ctx.timer_begin();
    mgpu::dtoh(xferbuf.data(), res.data(), res.size());
    ret[3] += ctx.timer_end();
  }

  roaring_bitmap_t *l0r = rbod(l0d), *l1r = rbod(l1d);
  ctx.timer_begin();
  for (size_t i = 0; i < 1000; i++) {
    roaring_bitmap_t *res = roaring_bitmap_and(l0r, l1r);
    roaring_bitmap_free(res);
  }
  ret[4] = ctx.timer_end();
  roaring_bitmap_free(l0r); roaring_bitmap_free(l1r);
  return ret;
}

casetime s34(mgpu::context_t& ctx) {
  casetime ret{0, 0, 0, 0};
  mgpu::mem_t<int> xferbuf(10 << 20, ctx, mgpu::memory_space_host);
  auto [l0d, l0sc] = loadCase(-34, 0, false, ctx);
  auto [l1d, l1sc] = loadCase(-34, 1, false, ctx);
  auto [l2d, l2sc] = loadCase(-34, 2, false, ctx);
  auto [l3d, l3sc] = loadCase(-34, 3, false, ctx);
  auto [l4d, l4sc] = loadCase(-34, 4, false, ctx);
  for (size_t i = 0; i < 1000; ++i) {
    ctx.timer_begin();
    merle or01 = wahOr(l1d.data(), l1sc.data(), l1d.size(), l0d.data(),
                       l0sc.data(), l0d.size(), ctx),
          or01s = wahCntExcScan(or01.data(), or01.size(), ctx);
    merle or23 = wahOr(l2d.data(), l2sc.data(), l2d.size(), l3d.data(),
                       l3sc.data(), l3d.size(), ctx),
          or23s = wahCntExcScan(or23.data(), or23.size(), ctx);
    merle res = wahAndNo1(or01.data(), or01s.data(), or01.size(), l4d.data(),
                          l4sc.data(), l4d.size(), ctx);
    assert(res.size() == l4d.size());
    res = wahAndNo1(or23.data(), or23s.data(), or23.size(), res.data(),
                    l4sc.data(), res.size(), ctx);
    ret[0] += ctx.timer_end();
    ctx.timer_begin();
    mgpu::dtoh(xferbuf.data(), res.data(), res.size());
    ret[1] += ctx.timer_end();
  }

  l0d = wahDecomp(l0d.data(), l0sc.data(), l0d.size(), ctx);
  l1d = wahDecomp(l1d.data(), l1sc.data(), l1d.size(), ctx);
  l2d = wahDecomp(l2d.data(), l2sc.data(), l2d.size(), ctx);
  l3d = wahDecomp(l3d.data(), l3sc.data(), l3d.size(), ctx);
  l4d = wahDecomp(l4d.data(), l4sc.data(), l4d.size(), ctx);
  for (size_t i = 0; i < 1000; ++i) {
    ctx.timer_begin();
    merle or01 = l1d.clone(), or23 = l3d.clone();
    wahOrExpl(or01.data(), l0d.data(), or01.size(), ctx);
    wahOrExpl(or23.data(), l2d.data(), or23.size(), ctx);
    wahAndExpl(or01.data(), or23.data(), or01.size(), ctx);
    wahAndExpl(or01.data(), l4d.data(), or01.size(), ctx);
    ret[2] += ctx.timer_end();
    ctx.timer_begin();
    mgpu::dtoh(xferbuf.data(), or01.data(), or01.size());
    ret[3] += ctx.timer_end();
  }

  roaring_bitmap_t *l0r = rbod(l0d), *l1r = rbod(l1d), *l2r = rbod(l2d),
                   *l3r = rbod(l3d), *l4r = rbod(l4d);
  ctx.timer_begin();
  for (size_t i = 0; i < 1000; i++) {
    roaring_bitmap_t *or01 = roaring_bitmap_or(l0r, l1r);
    roaring_bitmap_t *or23 = roaring_bitmap_or(l2r, l3r);
    roaring_bitmap_and_inplace(or01, l4r);
    roaring_bitmap_and_inplace(or01, or23);
    roaring_bitmap_free(or01); roaring_bitmap_free(or23);
  }
  ret[4] = ctx.timer_end();
  roaring_bitmap_free(l0r); roaring_bitmap_free(l1r); roaring_bitmap_free(l2r);
  roaring_bitmap_free(l3r); roaring_bitmap_free(l4r);
  return ret;
}

casetime s41(mgpu::context_t& ctx) {
  casetime ret{0, 0, 0, 0};
  mgpu::mem_t<int> xferbuf(10 << 20, ctx, mgpu::memory_space_host);
  auto [l0d, l0sc] = loadCase(-41, 0, false, ctx);
  auto [l1d, l1sc] = loadCase(-41, 1, false, ctx);
  auto [l2d, l2sc] = loadCase(-41, 2, false, ctx);
  auto [l3d, l3sc] = loadCase(-41, 3, false, ctx);
  ctx.timer_begin();
  l0d = wahDecomp(l0d.data(), l0sc.data(), l0d.size(), ctx);
  l1d = wahDecomp(l1d.data(), l1sc.data(), l1d.size(), ctx);
  l2d = wahDecomp(l2d.data(), l2sc.data(), l2d.size(), ctx);
  l3d = wahDecomp(l3d.data(), l3sc.data(), l3d.size(), ctx);
  ret[2] = ctx.timer_end() * 1000;

  for (size_t i = 0; i < 1000; ++i) {
    ctx.timer_begin();
    merle res(l0d.size(), ctx);
    mgpu::transform(
        [] MGPU_DEVICE(int i, const int *l0, const int *l1, const int *l2,
                       const int *l3, int *r) {
          int o = l2[i], a = l0[i];
          o |= l3[i];
          a &= l1[i];
          a &= o;
          r[i] = a;
        },
        l0d.size(), ctx, l0d.data(), l1d.data(), l2d.data(), l3d.data(),
        res.data());
    ret[0] += ctx.timer_end();
    ctx.timer_begin();
    mgpu::dtoh(xferbuf.data(), res.data(), res.size());
    ret[1] += ctx.timer_end();
  }
  ret[2] += ret[0];
  ret[3] = ret[1];

  roaring_bitmap_t *l0r = rbod(l0d), *l1r = rbod(l1d),
                   *l2r = rbod(l2d), *l3r = rbod(l3d);
  ctx.timer_begin();
  for (size_t i = 0; i < 1000; i++) {
    roaring_bitmap_t *res = roaring_bitmap_or(l2r, l3r);
    roaring_bitmap_and_inplace(res, l0r);
    roaring_bitmap_and_inplace(res, l1r);
    roaring_bitmap_free(res);
  }
  ret[4] = ctx.timer_end();
  roaring_bitmap_free(l0r); roaring_bitmap_free(l1r);
  roaring_bitmap_free(l2r); roaring_bitmap_free(l3r);
  return ret;
}

void runTestCase(mgpu::context_t &ctx) {
#define bah(tcase)                                                             \
  printf(#tcase ",%f,%f,%f,%f,%f\n", t[0], t[1], t[2], t[3], t[4]);       \
  malloc_trim(0);
  puts("case,merle,xfer,dnq,xfer,roaring");
  casetime t;

  t = s12_13(ctx, false); bah(S12);
  t = s12_13(ctx, true); bah(S13);
  t = s23(ctx); bah(S23);
  t = s34(ctx); bah(S34);
  t = s41(ctx); bah(S41);
  t = t3(ctx); bah(T 3);
  t = t6_17(ctx, false); bah(T 6);
  t = t12(ctx); bah(T12);
  t = t6_17(ctx, true); bah(T17);
}
