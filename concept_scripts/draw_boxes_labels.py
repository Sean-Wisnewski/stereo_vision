from PIL import  Image, ImageDraw, ImageFont
import cv2
import numpy as np
test_img_fname = "../imgs_for_code/sample_rect_imgs/image0000.jpg"
img = cv2.imread(test_img_fname)
img = Image.fromarray(img)
#img = Image.open(test_img_fname)
draw = ImageDraw.Draw(img)
font = ImageFont.load_default()
draw.text((0,0), "Sample Text", (255,255,255),font=font)
cv2.imshow('test', np.array(img, dtype=np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()