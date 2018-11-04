import os
import sys
from utils import is_img

contentPath = "images/UHD_content/center_img"
stylePath   = "images/UHD_style"
content_imgs = [x for x in os.listdir(contentPath) if is_img(x)] 
style_imgs =   [x for x in os.listdir(stylePath)   if is_img(x)]
pairs = [(i,j) for i in content_imgs for j in style_imgs]
for i,j in pairs:
  script = "python WCT_my.py --cuda  --UHD --UHD_contentPath images/UHD_content/center_img/ --fineSize=3000 --log_mark=20181031-1808  --picked_content_mark=%s  --picked_style_mark=%s" % (i, j)
  os.system(script)
