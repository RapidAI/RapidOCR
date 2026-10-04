#include "ppocr/ppocr.h"

#include "ppocr/ppocr.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <new>
#include <string>
#include <utility>
#include <vector>

namespace {

thread_local std::string g_error;

void SetError(const std::string& message) { g_error = message; }

void ClearError() { g_error.clear(); }

int Fail(int status, const std::string& message) {
  SetError(message);
  return status;
}

ppocr::Options ToOptions(const ppocr_options* options) {
  ppocr::Options out;
  if (!options) return out;
  ppocr_options local{};
  ppocr_options_init(&local);
  const int bytes = options->struct_size > 0 ? options->struct_size : static_cast<int>(sizeof(ppocr_options));
  const int copy = bytes < static_cast<int>(sizeof(local)) ? bytes : static_cast<int>(sizeof(local));
  if (copy > 0) std::memcpy(&local, options, static_cast<std::size_t>(copy));
  switch (local.backend) {
    case PPOCR_BACKEND_CPU: out.backend = ppocr::Backend::cpu_only; break;
    case PPOCR_BACKEND_GPU: out.backend = ppocr::Backend::gpu_only; break;
    default: out.backend = ppocr::Backend::hybrid; break;
  }
  out.det_threshold = local.det_threshold;
  out.det_box_threshold = local.det_box_threshold;
  out.det_unclip_ratio = local.det_unclip_ratio;
  out.det_limit_side_len = local.det_limit_side_len;
  out.rec_height = local.rec_height;
  out.rec_max_width = local.rec_max_width;
  out.rec_batch_size = local.rec_batch_size;
  out.rec_batch_width_bucket = local.rec_batch_width_bucket;
  out.rec_parallelism = local.rec_parallelism;
  out.hybrid_graph_parallelism = local.hybrid_graph_parallelism;
  out.det_batch_size = local.det_batch_size;
  out.batch_preprocess_parallelism = local.batch_preprocess_parallelism;
  out.image_batch_parallelism = local.image_batch_parallelism;
  return out;
}

char* Dup(const std::string& text) {
  char* owned = static_cast<char*>(std::malloc(text.size() + 1));
  if (!owned) return nullptr;
  std::memcpy(owned, text.data(), text.size() + 1);
  return owned;
}

int FillResult(const std::vector<ppocr::Result>& source, ppocr_result* result) {
  result->items = nullptr;
  result->count = 0;
  if (source.empty()) return PPOCR_OK;
  auto* items = static_cast<ppocr_text_box*>(std::calloc(source.size(), sizeof(ppocr_text_box)));
  if (!items) return Fail(PPOCR_ERR_NOMEM, "out of memory");
  for (std::size_t i = 0; i < source.size(); ++i) {
    items[i].text = Dup(source[i].text);
    if (!items[i].text) {
      for (std::size_t j = 0; j < i; ++j) std::free(items[j].text);
      std::free(items);
      return Fail(PPOCR_ERR_NOMEM, "out of memory");
    }
    items[i].confidence = source[i].confidence;
    items[i].bbox_x = source[i].bbox[0];
    items[i].bbox_y = source[i].bbox[1];
    items[i].bbox_w = source[i].bbox[2];
    items[i].bbox_h = source[i].bbox[3];
    for (int k = 0; k < 4; ++k) {
      items[i].box[k].x = static_cast<float>(source[i].box[k][0]);
      items[i].box[k].y = static_cast<float>(source[i].box[k][1]);
    }
  }
  result->items = items;
  result->count = source.size();
  return PPOCR_OK;
}

ppocr::Image Borrow(const ppocr_image& image) {
  ppocr::Image out;
  out.width = image.width;
  out.height = image.height;
  if (image.rgb && image.width > 0 && image.height > 0) {
    const std::size_t bytes = static_cast<std::size_t>(image.width) * image.height * 3;
    out.rgb.assign(image.rgb, image.rgb + bytes);
  }
  return out;
}

}  // namespace

struct ppocr_ocr {
  ppocr::OCR engine;
  explicit ppocr_ocr(ppocr::OCR engine_in) : engine(std::move(engine_in)) {}
};

extern "C" {

void ppocr_options_init(ppocr_options* options) {
  if (!options) return;
  std::memset(options, 0, sizeof(*options));
  options->struct_size = static_cast<int>(sizeof(ppocr_options));
  options->backend = PPOCR_BACKEND_HYBRID;
  const ppocr::Options defaults;
  options->det_threshold = defaults.det_threshold;
  options->det_box_threshold = defaults.det_box_threshold;
  options->det_unclip_ratio = defaults.det_unclip_ratio;
  options->det_limit_side_len = defaults.det_limit_side_len;
  options->rec_height = defaults.rec_height;
  options->rec_max_width = defaults.rec_max_width;
  options->rec_batch_size = defaults.rec_batch_size;
  options->rec_batch_width_bucket = defaults.rec_batch_width_bucket;
  options->rec_parallelism = defaults.rec_parallelism;
  options->hybrid_graph_parallelism = defaults.hybrid_graph_parallelism;
  options->det_batch_size = defaults.det_batch_size;
  options->batch_preprocess_parallelism = defaults.batch_preprocess_parallelism;
  options->image_batch_parallelism = defaults.image_batch_parallelism;
}

const char* ppocr_last_error(void) { return g_error.c_str(); }

ppocr_ocr* ppocr_create(const char* det_model, const char* rec_model,
                        const char* dictionary, const ppocr_options* options) {
  ClearError();
  if (!det_model || !rec_model || !dictionary) {
    Fail(PPOCR_ERR_INVALID_ARGUMENT, "model and dictionary paths are required");
    return nullptr;
  }
  try {
    auto* handle = new ppocr_ocr(ppocr::OCR(det_model, rec_model, dictionary, ToOptions(options)));
    return handle;
  } catch (const std::bad_alloc&) {
    Fail(PPOCR_ERR_NOMEM, "out of memory");
  } catch (const std::exception& ex) {
    const std::string message = ex.what();
    const int status = message.find("cannot open") != std::string::npos ? PPOCR_ERR_IO : PPOCR_ERR_MODEL;
    Fail(status, message);
  }
  return nullptr;
}

void ppocr_destroy(ppocr_ocr* ocr) { delete ocr; }

int ppocr_set_det_thresholds(ppocr_ocr* ocr, float det_threshold, float det_box_threshold,
                             float det_unclip_ratio) {
  ClearError();
  if (!ocr) return Fail(PPOCR_ERR_INVALID_ARGUMENT, "null OCR handle");
  try {
    ocr->engine.SetDetectionThresholds(det_threshold, det_box_threshold, det_unclip_ratio);
    return PPOCR_OK;
  } catch (const std::exception& ex) {
    return Fail(PPOCR_ERR_INVALID_ARGUMENT, ex.what());
  }
}

int ppocr_recognize(ppocr_ocr* ocr, const ppocr_image* image, ppocr_result* result) {
  ClearError();
  if (!result) return Fail(PPOCR_ERR_INVALID_ARGUMENT, "null result");
  result->items = nullptr;
  result->count = 0;
  if (!ocr || !image) return Fail(PPOCR_ERR_INVALID_ARGUMENT, "null OCR handle or image");
  try {
    return FillResult(ocr->engine.Recognize(Borrow(*image)), result);
  } catch (const std::bad_alloc&) {
    return Fail(PPOCR_ERR_NOMEM, "out of memory");
  } catch (const std::exception& ex) {
    return Fail(PPOCR_ERR_RUNTIME, ex.what());
  }
}

int ppocr_recognize_batch(ppocr_ocr* ocr, const ppocr_image* images, size_t count,
                          ppocr_result* outs) {
  ClearError();
  if (!ocr || (!images && count > 0) || !outs) {
    return Fail(PPOCR_ERR_INVALID_ARGUMENT, "null batch argument");
  }
  for (size_t i = 0; i < count; ++i) {
    outs[i].items = nullptr;
    outs[i].count = 0;
  }
  try {
    std::vector<ppocr::Image> owned;
    owned.reserve(count);
    for (size_t i = 0; i < count; ++i) owned.push_back(Borrow(images[i]));
    const auto batch = ocr->engine.RecognizeBatch(owned);
    if (batch.size() != count) return Fail(PPOCR_ERR_RUNTIME, "batch size mismatch");
    for (size_t i = 0; i < count; ++i) {
      const int status = FillResult(batch[i], &outs[i]);
      if (status != PPOCR_OK) {
        for (size_t j = 0; j < i; ++j) ppocr_result_free(&outs[j]);
        return status;
      }
    }
    return PPOCR_OK;
  } catch (const std::bad_alloc&) {
    return Fail(PPOCR_ERR_NOMEM, "out of memory");
  } catch (const std::exception& ex) {
    return Fail(PPOCR_ERR_RUNTIME, ex.what());
  }
}

void ppocr_result_free(ppocr_result* result) {
  if (!result) return;
  if (result->items) {
    for (size_t i = 0; i < result->count; ++i) std::free(result->items[i].text);
    std::free(result->items);
  }
  result->items = nullptr;
  result->count = 0;
}

int ppocr_load_ppm(const char* path, ppocr_image_buffer* image) {
  ClearError();
  if (!image) return Fail(PPOCR_ERR_INVALID_ARGUMENT, "null image");
  image->width = 0;
  image->height = 0;
  image->rgb = nullptr;
  if (!path) return Fail(PPOCR_ERR_INVALID_ARGUMENT, "null path");
  try {
    ppocr::Image loaded = ppocr::LoadPPM(path);
    const std::size_t bytes = loaded.rgb.size();
    image->rgb = static_cast<uint8_t*>(std::malloc(bytes));
    if (!image->rgb) return Fail(PPOCR_ERR_NOMEM, "out of memory");
    std::memcpy(image->rgb, loaded.rgb.data(), bytes);
    image->width = loaded.width;
    image->height = loaded.height;
    return PPOCR_OK;
  } catch (const std::exception& ex) {
    return Fail(PPOCR_ERR_IO, ex.what());
  }
}

void ppocr_image_free(ppocr_image_buffer* image) {
  if (!image) return;
  std::free(image->rgb);
  image->rgb = nullptr;
  image->width = 0;
  image->height = 0;
}

void ppocr_query_backend_info(ppocr_backend_info* info) {
  if (!info) return;
  std::memset(info, 0, sizeof(*info));
  try {
    const ppocr::BackendInfo source = ppocr::QueryBackendInfo();
    info->vulkan_loader_available = source.vulkan_loader_available ? 1 : 0;
    info->vulkan_compute_available = source.vulkan_compute_available ? 1 : 0;
    info->full_graph_gpu_available = source.full_graph_gpu_available ? 1 : 0;
    info->vulkan_runtime_generation = source.vulkan_runtime_generation;
    std::snprintf(info->device_name, sizeof(info->device_name), "%s", source.device_name.c_str());
  } catch (const std::exception& ex) {
    SetError(ex.what());
  }
}

void ppocr_query_cpu_info(ppocr_cpu_info* info) {
  if (!info) return;
  std::memset(info, 0, sizeof(*info));
  const ppocr::CpuInfo source = ppocr::QueryCpuInfo();
  info->avx2_compiled = source.avx2_compiled ? 1 : 0;
  info->avx512_compiled = source.avx512_compiled ? 1 : 0;
  info->neon_compiled = source.neon_compiled ? 1 : 0;
  info->avx2 = source.avx2 ? 1 : 0;
  info->avx512 = source.avx512 ? 1 : 0;
  info->neon = source.neon ? 1 : 0;
  info->threads = source.threads;
  std::snprintf(info->active_isa, sizeof(info->active_isa), "%s", source.active_isa.c_str());
}

}  // extern "C"
