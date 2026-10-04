#include "ppocr/ppocr.h"

#include <stdio.h>
#include <string.h>

static int g_failures = 0;

static void Expect(int condition, const char* name) {
  if (!condition) {
    fprintf(stderr, "FAIL %s\n", name);
    ++g_failures;
  } else {
    printf("ok %s\n", name);
  }
}

int main(void) {
  ppocr_cpu_info cpu;
  memset(&cpu, 0, sizeof(cpu));
  ppocr_query_cpu_info(&cpu);
  Expect(cpu.active_isa[0] != '\0', "cpu info isa");
  Expect(cpu.threads > 0, "cpu info threads");
  Expect(!(cpu.avx512 && !cpu.avx2), "avx512 implies avx2");

  ppocr_options options;
  ppocr_options_init(&options);
  Expect(options.struct_size == (int)sizeof(ppocr_options), "options size");
  Expect(options.rec_batch_size > 0, "default batch");
  Expect(options.backend == PPOCR_BACKEND_HYBRID, "default backend");

  Expect(ppocr_create(NULL, "a", "b", NULL) == NULL, "null det");
  Expect(ppocr_last_error()[0] != '\0', "null det error");
  Expect(ppocr_create("missing-det.onnx", "missing-rec.onnx", "missing-dict.txt", &options) == NULL,
         "missing files");

  ppocr_result result = {0};
  Expect(ppocr_recognize(NULL, NULL, &result) == PPOCR_ERR_INVALID_ARGUMENT, "null recognize");
  ppocr_image_buffer image = {0};
  Expect(ppocr_load_ppm("missing.ppm", &image) != PPOCR_OK, "missing ppm");

  ppocr_backend_info backend;
  ppocr_query_backend_info(&backend);
  Expect(backend.vulkan_loader_available == 0 || backend.vulkan_loader_available == 1,
         "backend info");

  if (g_failures) {
    fprintf(stderr, "%d failure(s)\n", g_failures);
    return 1;
  }
  printf("c api smoke passed isa=%s\n", cpu.active_isa);
  return 0;
}
