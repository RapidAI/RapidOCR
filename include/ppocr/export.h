#ifndef PPOCR_EXPORT_H_
#define PPOCR_EXPORT_H_

/* Shared-library export macro. Static consumers leave PPOCR_API empty.
   The shared build compiles with PPOCR_BUILDING_SHARED. Windows consumers
   of the DLL define PPOCR_SHARED. */
#if defined(_WIN32)
#  if defined(PPOCR_BUILDING_SHARED)
#    define PPOCR_API __declspec(dllexport)
#  elif defined(PPOCR_SHARED)
#    define PPOCR_API __declspec(dllimport)
#  else
#    define PPOCR_API
#  endif
#else
#  if defined(PPOCR_BUILDING_SHARED)
#    define PPOCR_API __attribute__((visibility("default")))
#  else
#    define PPOCR_API
#  endif
#endif

#endif
