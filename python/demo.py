"""Same shape as ``python/demo.py`` on ``main``.

Pass model files or let the package download the PP-OCRv6 tiny bundle:

    python demo.py --image page.png --model-type tiny
"""

from rapidocr import RapidOCR

engine = RapidOCR(
    params={
        "Det.model_type": "tiny",
        "Rec.model_type": "tiny",
        "Det.ocr_version": "PP-OCRv6",
        "Rec.ocr_version": "PP-OCRv6",
        "Global.use_cls": False,
    }
)

if __name__ == "__main__":
    import sys

    image = sys.argv[1] if len(sys.argv) > 1 else None
    if image is None:
        raise SystemExit("usage: python demo.py image.jpg")
    result = engine(image)
    print(result.txts)
    print(result.scores)
    print(result.boxes)
    print("elapse", result.elapse)
