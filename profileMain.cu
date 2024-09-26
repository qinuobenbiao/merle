#include "wahGpu.cuh"
using merle = mgpu::mem_t<int>;

int main(int argc, char** argv) {
  dumb_pool_t ctx(512 * 1024 * 1024);
  // mgpu::standard_context_t context;
  const char* inFmt = nullptr;
  const char* stashOutPath = "stash.csv";
  const char* opOutPath = "op.csv";
  size_t nFile = 200, nDup = 100;
  wahOp op = wahOp::AND;
  bool semi = false;

  for (int i = 1; i < argc; i += 2) {
    if (0 == strcasecmp(argv[i], "--inFmt"))
      inFmt = argv[1 + i];
    else if (0 == strcasecmp(argv[i], "--stashOut"))
      stashOutPath = argv[1 + i];
    else if (0 == strcasecmp(argv[i], "--nFile"))
      nFile = atoi(argv[1 + i]);
    else if (0 == strcasecmp(argv[i], "--nDup"))
      nDup = atoi(argv[1 + i]);
    else if (0 == strcasecmp(argv[i], "--opOut"))
      opOutPath = argv[1 + i];
    else if (0 == strcasecmp(argv[i], "--semi")) {
      semi = true; --i;
    } else if (0 == strcasecmp(argv[i], "--op")) {
      if (argv[1 + i][0] == 'o' || argv[1 + i][0] == 'O')
        op = wahOp::OR;
      else if (argv[1 + i][0] == 'x' || argv[1 + i][0] == 'X')
        op = wahOp::XOR;
      else if (argv[1 + i][0] == 'd' || argv[1 + i][0] == 'D')
        stashOutPath = nullptr;
    }
    else return fputs("Invalid Argument", stderr);
  }

  if (inFmt == nullptr) {
    // ssb and tpch (SF=20) if no dataset specified.
    // 1000 iterations each testcase and total time is reported in secs
    void runTestCase(mgpu::context_t &ctx);
    runTestCase(ctx);
    return 0;
  }
  if (strlen(inFmt) > 240)
    return fputs("Name too long!", stderr);

  std::vector<wahHost_s*> dupped;
  for (size_t i = 0; i < nFile; ++i) {
    char path[256];
    sprintf(path, inFmt, i);
    wahHost_s* orig = loadWahFile(path), *d = wahDup(orig, nDup);
    dupped.push_back(d);
    free(orig);
  }

  // Heat up & correctness check
  wahOpProfile profile;
  srand(time(0));
  for (size_t i = 0; i < 10; ++i) {
    size_t lId = rand() % nFile, rId = rand() % nFile;
    profile.setLhs(ctx, dupped[lId], wahStashDat_s());
    profile.setRhs(ctx, dupped[rId], wahStashDat_s());
    profile.profileAnd(true);
    profile.profileOr(true);
    profile.profileXor(true);
    profile.clear();
  }

  FILE *opOutf = fopen(opOutPath, "w");
  if (stashOutPath == nullptr) {
    // Profile decompression only
    merle andrzejewski(const int *C, size_t m, mgpu::context_t &ctx);
    merle ours(const int *C, size_t m, mgpu::context_t &ctx);
    fputs("id,encSz,decSz,andrzejewski,ours\n", opOutf);
    for (size_t i = 0; i < nFile; ++i) {
      wahHost_s *encHost = dupped[i];
      size_t sz = encHost->cmprsNrWord;
      merle encDev = merle(sz, ctx);
      mgpu::htod(encDev.data(), reinterpret_cast<const int *>(encHost->dat), sz);
      ctx.timer_begin();
      auto resD = ours(encDev.data(), sz, ctx);
      double goodTime = ctx.timer_end();
      auto goodResH = mgpu::from_mem(resD);
      resD = merle();
      ctx.timer_begin();
      resD = andrzejewski((const int *)encDev.data(), sz, ctx);
      double badTime = ctx.timer_end();
      auto badResH = mgpu::from_mem(resD);
      fprintf(opOutf, "%zu,%zu,%zu,%f,%f\n", i, sz, resD.size(),
              badTime, goodTime);
      if (badResH != goodResH)
        exit(1);
    }
    fclose(opOutf);
    return 0;
  }

  // stash data
  FILE *stashOutf = fopen(stashOutPath, "w");
  if (stashOutf == nullptr)
    return fputs("Cannot open stash output!", stderr);
  fputs("id,excScanTime,decTime,orig+scanSz,decSz\n", stashOutf);
  std::vector<wahStashDat_s> stashDats;
  for (size_t i = 0; i < nFile; ++i) {
    wahDev_s wah(ctx, dupped[i]);
    stashDats.emplace_back(i, wah, 4);
    stashDats.back().print(stashOutf);
  }
  fclose(stashOutf);

  fputs(semi ? "lhs,rhs,expl,explXfer,explCptXfer,direct,directXfer,"
               "directCptXfer,semi,semiXfer,semiCptXfer,cpu\n"
             : "lhs,rhs,expl,explXfer,explCptXfer,direct,directXfer,"
               "directCptXfer,cpu\n",
        opOutf);
  for (size_t i = 0; i < nFile; ++i) {
    profile.clear();
    profile.setLhs(ctx, dupped[i], stashDats[i]);
    for (size_t j = i+1; j < nFile; ++j) {
      profile.setRhs(ctx, dupped[j], stashDats[j]);
      profile.profile(op, semi);
      profile.print(opOutf, semi);
    }
  }

  fclose(opOutf);
  return 0;
}
