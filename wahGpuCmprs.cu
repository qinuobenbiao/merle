#include "wahGpu.cuh"
#include "moderngpu/kernel_load_balance.cuh"
#include "moderngpu/kernel_compact.cuh"

mgpu::mem_t<int> wahCntExcScan(const int *wah, size_t sz, mgpu::context_t &ctx) {
  mgpu::mem_t<int> res(sz + 1, ctx);
  mgpu::transform_scan<int>([=] MGPU_DEVICE (int index) {
    int val = wah[index];
    return val & 0x80000000 ? val & 0x3fffffff : 1;
  }, sz, res.data(), mgpu::plus_t<int>(),
    res.data() + sz, ctx);
  return res;
}

void wahAndExpl(int *upDecomp, const int *downDecomp, size_t minSz,
                mgpu::context_t &context) {
  mgpu::transform([] MGPU_DEVICE(int i, int *u, const int *d) { u[i] &= d[i]; },
                  minSz, context, upDecomp, downDecomp);
}

void wahOrExpl(int *upDecomp, const int *downDecomp, size_t minSz,
                mgpu::context_t &context) {
  mgpu::transform([] MGPU_DEVICE(int i, int *u, const int *d) { u[i] |= d[i]; },
                  minSz, context, upDecomp, downDecomp);
}

void wahXorExpl(int *upDecomp, const int *downDecomp, size_t minSz,
                mgpu::context_t &context) {
  mgpu::transform([] MGPU_DEVICE(int i, int *u, const int *d) { u[i] ^= d[i]; },
                  minSz, context, upDecomp, downDecomp);
}

mgpu::mem_t<int> wahDecomp(const int *wah, const int *excScan, size_t sz,
                           mgpu::context_t &ctx, int decompSz)
{
  using namespace mgpu;
  if (decompSz == 0)
    dtoh(&decompSz, excScan + sz, 1);
  mem_t<int> res(decompSz, ctx); int* resData = res.data();
  transform_lbs([=] MGPU_DEVICE (int idx, int , int, tuple<int> w) {
    int val = get<0>(w);
    val = val & 0x80000000 ? val << 1 >> 31 : val;
    val &= 0x7fffffff;
    resData[idx] = val;
  }, decompSz, excScan, sz, make_tuple(wah), ctx);
  return res;
}

// `dec` is decompressed, `enc` is compressed
mgpu::mem_t<int> wahEncOpDec(const int *enc, const int *excScan, size_t encSz,
                             const int *dec, size_t decSz, wahOp op,
                             mgpu::context_t &ctx, int encDecompSz) {
  using namespace mgpu;
  if (encDecompSz == 0)
    dtoh(&encDecompSz, excScan + encSz, 1);
  size_t resSz = max(decSz, size_t(encDecompSz));
  mem_t<int> res(resSz, ctx); int* resData = res.data();
  transform_lbs([=] MGPU_DEVICE (int decAt, int, int, tuple<int> w) {
    int val = get<0>(w);
    int decVal = decAt < decSz ? dec[decAt] : 0;
    val = val & 0x80000000 ? val << 1 >> 31 : val;
    if (op == wahOp::OR) val |= decVal;
    // else if (op == wahOp::AND) val &= decVal;
    else val ^= decVal;
    resData[decAt] = val & 0x7fffffff;
  }, encDecompSz, excScan, encSz,
    make_tuple(enc), ctx);
  if (decSz > encDecompSz)
    dtod(encDecompSz + resData, encDecompSz + dec, decSz - encDecompSz);
  return res;
}

mgpu::mem_t<int> wahEncAndDec(const int *enc, const int *excScan, size_t encSz,
                              const int *dec, size_t decSz, mgpu::context_t &ctx) {
  using namespace mgpu;
  mem_t<int> workCntScan(encSz + 1, ctx);
  transform_scan<int, scan_type_inc>([=] MGPU_DEVICE (int thId) {
    if (thId == 0) return 0;
    uint32_t encWah = enc[thId - 1];
    if ((encWah >> 30) != 3)
      return 1;
    return int(encWah & 0x3fffffff);
  }, encSz + 1, workCntScan.data(), plus_t<int>(), discard_iterator_t<int>(), ctx);
  int nRes; dtoh(&nRes, workCntScan.data() + encSz, 1);

  mem_t<int> res(nRes, ctx); int* resData = res.data();
  transform_lbs([=] MGPU_DEVICE(int idx, int, int rank, tuple<int, int> c) {
    int encWah = get<1>(c), out; // down[seg]
    uint32_t ty = (uint32_t)encWah >> 30;
    if (ty == 2) { // enc is 0-fill
      out = encWah;
    } else {
      int decAt = get<0>(c) + rank;
      if (decAt >= decSz) { resData[idx] = 0x80000001; return; }
      int decWah = dec[decAt];
      if (ty == 3) // enc is 1-fill
        out = decWah;
      else // enc is tail
        out = decWah & encWah;
    }
    resData[idx] = out;
  }, nRes, workCntScan.data(), encSz + 1,
    make_tuple(excScan, enc), ctx);
  return res;
}

// `enc` is in place
void wahEncNo1AndDec(int *enc, const int *excScan, size_t encSz, const int *dec,
                     size_t decSz, mgpu::context_t &ctx) {
  mgpu::transform([=] MGPU_DEVICE (int encAt) {
    int encWah = enc[encAt], decAt = excScan[encAt], out;
    if (decAt >= decSz)
      out = 0x80000001;
    else if (encWah & 0x80000000)
      out = encWah;
    else
      out = encWah & dec[decAt];
    enc[encAt] = out;
  }, encSz, ctx);
}

mgpu::mem_t<int> wahCompact(const int* wah, size_t wahSz, mgpu::context_t& context) {
  auto compact = mgpu::transform_compact(wahSz, context);
  int resCnt = compact.upsweep([s=(int)wahSz - 1, wah] MGPU_DEVICE (int i) {
    int val = wah[i];
    if (val == 0x7fffffff) val = 0xc0000001;
    if (val == 0) val = 0x80000001;
    if (!(val & 0x80000000))
      return true;
    if (i == s)
      return (val & 0x40000000) != 0;

    val &= 0xc0000000;
    int nxtVal = wah[i + 1];
    if (nxtVal == 0x7fffffff) nxtVal = 0xc0000001;
    if (nxtVal == 0) nxtVal = 0x80000001;
    nxtVal &= 0xc0000000;
    return nxtVal != val;
  });

  mgpu::mem_t<int> cntIncScan(wahSz, context);
  mgpu::transform_scan<int, mgpu::scan_type_inc>([=] MGPU_DEVICE (int index) {
    int v = wah[index];
    return v & 0x80000000 ? v & 0x3fffffff : 1;
  }, wahSz, cntIncScan.data(), mgpu::plus_t<int>(),
    mgpu::discard_iterator_t<int>(), context);

  mgpu::mem_t<int> res(resCnt, context);
  mgpu::mem_t<int> cntScanCompacted(resCnt, context);
  int *scanDat = cntIncScan.data(), *scanCpaDat = cntScanCompacted.data(),
      *resDat = res.data();
  compact.downsweep([=] MGPU_DEVICE (int destIdx, int srcIdx) {
    resDat[destIdx] = wah[srcIdx];
    scanCpaDat[destIdx] = scanDat[srcIdx];
  });

  mgpu::transform([=] MGPU_DEVICE (int i) {
    auto val = resDat[i];
    if (val == 0x7fffffff) val = 0xc0000001;
    if (val == 0) val = 0x80000001;
    if (!(val & 0x80000000))
      return;
    auto sum = scanCpaDat[i], prevSum = i == 0 ? 0 : scanCpaDat[i - 1];
    val &= 0xc0000000;
    val += (sum - prevSum);
    resDat[i] = val;
  }, resCnt, context);
  return res;
}

mgpu::mem_t<int> wahCompress(const int* dec, size_t wahSz, mgpu::context_t& context) {
  auto compact = mgpu::transform_compact(wahSz, context);
  int resCnt = compact.upsweep([s=(int)wahSz - 1, dec] MGPU_DEVICE (int i) {
    int val = dec[i];
    if (i == s) return val != 0;
    if (val != 0x7fffffff && val != 0) return true;
    return dec[i + 1] != val;
  });

  mgpu::mem_t<int> res(resCnt, context);
  mgpu::mem_t<int> cntScanCompacted(resCnt, context);
  int *scanCpaDat = cntScanCompacted.data(), *resDat = res.data();
  compact.downsweep([=] MGPU_DEVICE (int destIdx, int srcIdx) {
    resDat[destIdx] = dec[srcIdx];
    scanCpaDat[destIdx] = srcIdx + 1;
  });

  mgpu::transform([=] MGPU_DEVICE (int i) {
    auto val = resDat[i];
    if (val == 0x7fffffff) val = 0xc0000000;
    else if (val == 0) val = 0x80000000;
    else return;
    auto sum = scanCpaDat[i], prevSum = i == 0 ? 0 : scanCpaDat[i - 1];
    resDat[i] = val + sum - prevSum;
  }, resCnt, context);
  return res;
}
