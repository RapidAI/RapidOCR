import fitz
import numpy as np

from rapidocr_onnxruntime import RapidOCR

if __name__ == "__main__":
    ocr = RapidOCR()
    for pdf_path in ["docs/page2.pdf"]:
        doc = fitz.open(pdf_path)
        for page in doc:
            img_list = page.get_images()
            for img in img_list:
                pix = fitz.Pixmap(doc, img[0])
                img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, -1
                )
                result, _ = ocr(img_array)
                if result:
                    ocr_result = [line[1] for line in result]
                    print(ocr_result)
