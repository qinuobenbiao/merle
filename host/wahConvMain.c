#include "wahCpu.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef enum {
  BitsetToWah = 1,
  ArrToBitset = 2,
  ArrToWah = 3,
  OpDemo = 4,
} conv_e;

int main(int argc, const char* const* argv) {
  // const char* inFmt = "dataset/census-income/census-income.csv%zu.txt";
  // const char* inFmt = "dataset/census-income_srt/census-income_srt.csv%zu.txt";
  // const char* inFmt = "dataset/census1881/census1881.csv%zu.txt";
  // const char* inFmt = "dataset/census1881_srt/census1881_srt.csv%zu.txt";
  // const char* inFmt = "dataset/weather_sept_85/weather_sept_85.csv%zu.txt";
  const char* inFmt = "dataset/weather_sept_85_srt/weather_sept_85_srt.csv%zu.txt";
  // const char* inFmt = "dataset/wikileaks-noquotes/wikileaks-noquotes.csv%zu.txt";
  // const char* inFmt = "dataset/wikileaks-noquotes_srt/wikileaks-noquotes_srt.csv%zu.txt";
  const char* outFmt = "wahData/weaSrt/%zu.wah";
  conv_e conv = ArrToWah;
  size_t unroll1Fill = 2, nFile = 200;

  if (0 == strcasecmp(argv[1], "arrToBitset"))
    conv = ArrToBitset;
  else if (0 == strcasecmp(argv[1], "BitsetToWah"))
    conv = BitsetToWah;
  else if (0 == strcasecmp(argv[1], "arrToWah"))
    conv = ArrToWah;
  else if (0 == strcasecmp(argv[1], "opDemo"))
    conv = OpDemo;
  else return fputs("Invalid conversion type", stderr);

  for (int i = 2; i < argc; i += 2) {
    if (0 == strcasecmp(argv[i], "--inFmt"))
      inFmt = argv[1 + i];
    else if (0 == strcasecmp(argv[i], "--outFmt"))
      outFmt = argv[1 + i];
    else if (0 == strcasecmp(argv[i], "--nFile"))
      nFile = atoi(argv[1 + i]);
    else if (0 == strcasecmp(argv[i], "--unroll"))
      unroll1Fill = atoi(argv[1 + i]);
    else return fputs("Invalid Argument", stderr);
  }
  if (strlen(inFmt) > 240)
    return fputs("Name too long!", stderr);
  if (strlen(outFmt) > 240)
    return fputs("Name too long!", stderr);

  if (conv == OpDemo) {
    struct wahHost_s *lhs = loadWahFile(inFmt), *rhs = loadWahFile(outFmt);
    uint32_t *out = malloc((lhs->cmprsNrWord + rhs->cmprsNrWord) * 4), *r;
    clock_t clk = clock();
    for (size_t i = 0; i < 100; ++i) {
      r = wahAndCPU(lhs->dat, lhs->dat + lhs->cmprsNrWord, rhs->dat,
                    rhs->dat + rhs->cmprsNrWord, out, (void *)-1);
    }
    printf("clock:%ld\tsize:%ld\n", clock() - clk, (r - out) * 4);
    return 0;
  }

  for (size_t i = 0; i < nFile; ++i) {
    char path[256];
    sprintf(path, inFmt, i);
    struct wahHost_s *res = conv & 2 ? loadArrayFile(path) : loadWahFile(path);
    if (conv & 1) {
      size_t len1Fill = wahLongest1Fill(res->dat, res->cmprsNrWord);
      if (len1Fill > unroll1Fill) {
        res->cmprsNrWord = wahCmprsCPU(res->dat, res->cmprsNrWord, res->dat);
        res->has1Fill = true;
        printf("%zu: %lu bits~\n", i, res->cmprsNrWord * 32);
      } else {
        res->cmprsNrWord = wahCmprsNo1CPU(res->dat, res->cmprsNrWord, res->dat);
        res->has1Fill = false;
        printf("%zu: %lu bits \n", i, res->cmprsNrWord * 32);
      }
    }

    // TODO: Function?
    sprintf(path, outFmt, i);
    FILE* outStream = fopen(path, "wb");
    fwrite(res, sizeof(*res) + res->cmprsNrWord * 4, 1, outStream);
    fclose(outStream);
    free(res);
  }
  return 0;
}
