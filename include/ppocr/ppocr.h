#ifndef PPOCR_PPOCR_H_
#define PPOCR_PPOCR_H_

#include "ppocr/export.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Stable C ABI for the PP-OCRv6 engine. Handles are opaque. Strings and
   result arrays returned to the caller are owned by the result object and
   released with ppocr_result_free / ppocr_image_free. The library does not
   throw across this boundary; failures return a status and a thread-local
   message from ppocr_last_error(). */

#define PPOCR_VERSION_MAJOR 5
#define PPOCR_VERSION_MINOR 0
#define PPOCR_VERSION_PATCH 0

#define PPOCR_OK 0
#define PPOCR_ERR_INVALID_ARGUMENT 1
#define PPOCR_ERR_MODEL 2
#define PPOCR_ERR_IO 3
#define PPOCR_ERR_RUNTIME 4
#define PPOCR_ERR_NOMEM 5
#define PPOCR_ERR_UNSUPPORTED 6

#define PPOCR_BACKEND_CPU 0
#define PPOCR_BACKEND_GPU 1
#define PPOCR_BACKEND_HYBRID 2

typedef struct ppocr_ocr ppocr_ocr;

typedef struct ppocr_options {
  int struct_size;
  int backend; /* PPOCR_BACKEND_* */
  float det_threshold;
  float det_box_threshold;
  float det_unclip_ratio;
  int det_limit_side_len;
  int rec_height;
  int rec_max_width;
  int rec_batch_size;
  int rec_batch_width_bucket;
  int rec_parallelism;
  int hybrid_graph_parallelism;
  int det_batch_size;
  int batch_preprocess_parallelism;
  int image_batch_parallelism;
  /* 0: shrink when the long side exceeds det_limit_side_len (C++ default).
     1: grow when the short side is below det_limit_side_len (rapidocr 3.x "min"). */
  int det_limit_type;
  /* Plane order is B, G, R, matching an OpenCV image and rapidocr 3.x. */
  float det_mean[3];
  float det_std[3];
  float rec_mean[3];
  float rec_std[3];
  int det_use_dilation; /* 1: 2x2 dilation before DB components */
  int det_max_candidates;
  /* -1: automatic adapter. >=0: that physical device, including lavapipe.
     Read on the first Vulkan initialization in the process. */
  int vulkan_device_index;
} ppocr_options;

typedef struct ppocr_image {
  int width;
  int height;
  const uint8_t* rgb; /* packed RGB, row-major, width * height * 3 bytes */
} ppocr_image;

typedef struct ppocr_image_buffer {
  int width;
  int height;
  uint8_t* rgb;
} ppocr_image_buffer;

typedef struct ppocr_point {
  float x;
  float y;
} ppocr_point;

typedef struct ppocr_text_box {
  char* text;
  float confidence;
  int bbox_x;
  int bbox_y;
  int bbox_w;
  int bbox_h;
  ppocr_point box[4]; /* TL, TR, BR, BL */
} ppocr_text_box;

typedef struct ppocr_result {
  ppocr_text_box* items;
  size_t count;
} ppocr_result;

typedef struct ppocr_backend_info {
  int vulkan_loader_available;
  int vulkan_compute_available;
  int full_graph_gpu_available;
  char device_name[256];
  uint64_t vulkan_runtime_generation;
} ppocr_backend_info;

typedef struct ppocr_cpu_info {
  int avx2_compiled;
  int avx512_compiled;
  int neon_compiled;
  int avx2;    /* runtime path enabled (hardware AND not forced off) */
  int avx512;
  int neon;
  int threads;
  char active_isa[16]; /* "avx512", "avx2", "neon", or "scalar" */
} ppocr_cpu_info;

/* Fills options with the same defaults as ppocr::Options and sets struct_size. */
PPOCR_API void ppocr_options_init(ppocr_options* options);

/* Returns NULL on failure. ppocr_last_error() is set. options may be NULL. */
PPOCR_API ppocr_ocr* ppocr_create(const char* det_model, const char* rec_model,
                                  const char* dictionary,
                                  const ppocr_options* options);
PPOCR_API void ppocr_destroy(ppocr_ocr* ocr);

/* Updates detector post-process thresholds on an existing handle. */
PPOCR_API int ppocr_set_det_thresholds(ppocr_ocr* ocr, float det_threshold,
                                       float det_box_threshold,
                                       float det_unclip_ratio);

PPOCR_API int ppocr_recognize(ppocr_ocr* ocr, const ppocr_image* image,
                              ppocr_result* result);
/* outs must point at count result objects. Input order is preserved. */
PPOCR_API int ppocr_recognize_batch(ppocr_ocr* ocr, const ppocr_image* images,
                                    size_t count, ppocr_result* outs);
PPOCR_API void ppocr_result_free(ppocr_result* result);

PPOCR_API int ppocr_load_ppm(const char* path, ppocr_image_buffer* image);
PPOCR_API void ppocr_image_free(ppocr_image_buffer* image);

PPOCR_API void ppocr_query_backend_info(ppocr_backend_info* info);
/* Pins the Vulkan physical device before the first initialization.
   index < 0 leaves the current choice unchanged. */
PPOCR_API void ppocr_request_vulkan_device(int device_index);
/* ERROR-severity VK_LAYER_KHRONOS_validation messages since process start.
   The layer is enabled only when PPOCR_VULKAN_VALIDATION is set and not "0". */
PPOCR_API int ppocr_vulkan_validation_error_count(void);
PPOCR_API void ppocr_query_cpu_info(ppocr_cpu_info* info);

/* Thread-local. Pointer is valid until the next API call on this thread. */
PPOCR_API const char* ppocr_last_error(void);

#ifdef __cplusplus
}
#endif

#endif
