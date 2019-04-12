#Author: Jordan Bennett

import numpy as np
from PIL import Image

#take katie's original black hole image as input
img = Image.open('einstein_katie-bouman_black-hole_photograph [original-version].jpg')
lr_img = np.array(img)

#establish super resolution model wrt saved weights
from ISR.models import RDN

rdn = RDN(arch_params={'C':6, 'D':20, 'G':64, 'G0':64, 'x':2})
rdn.model.load_weights('C:/Users/bennettjm/Desktop/black-hole-source/ISR-test/weights/sample_weights/rdn-C6-D20-G64-G064-x2/ArtefactCancelling/rdn-C6-D20-G64-G064-x2_ArtefactCancelling_epoch219.hdf5')

#supply image to super resolution model prediction
sr_img = rdn.predict(lr_img)

#Line added to save super resolved version of katie's image
superResolutionImage = Image.fromarray(sr_img)
superResolutionImage.save('einstein_katie-bouman_black-hole_photograph [super-resolution-version].jpg')
