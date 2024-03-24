from mmocr.apis import MMOCRInferencer, TextDetInferencer
# Load models into memory
# ocr = MMOCRInferencer(det='DBNet', rec='SAR')
# # Perform inference
# ocr('demo/demo_text_ocr.jpg', out_dir='demo/mmocr_out', save_vis=True)

det = TextDetInferencer(model="DBNet")
det('demo/demo_text_ocr.jpg', out_dir='demo/mmocr_out', save_vis=True)