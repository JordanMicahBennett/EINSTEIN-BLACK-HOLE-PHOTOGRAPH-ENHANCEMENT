About
===
I quickly used a [generative adversarial artificial neural network model](https://github.com/idealo/image-super-resolution) to enhance or to generate super resolution version of the first ever photograph of Einstein's black-hole, lead by PhD Katie Bouman.

Images
===
[Original-image (970x545)](https://github.com/JordanMicahBennett/EINSTEIN-BLACK-HOLE-PHOTOGRAPH-ENHANCEMENT/blob/master/source-code/einstein_katie-bouman_black-hole_photograph%20%5Boriginal-version%5D.jpg) taken [from space.com](https://www.space.com/first-black-hole-photo-by-event-horizon-telescope.html).

[Super-resolution version (1940x1090)](https://github.com/JordanMicahBennett/EINSTEIN-BLACK-HOLE-PHOTOGRAPH-ENHANCEMENT/blob/master/source-code/einstein_katie-bouman_black-hole_photograph%20%5Bsuper-resolution-version%5D.jpg) produced using the code in [source code folder](https://github.com/JordanMicahBennett/EINSTEIN-BLACK-HOLE-PHOTOGRAPH-ENHANCEMENT/tree/master/source-code), and adjust the directory according to your location.

Background
===
**Einstein's General-Relativity black hole** viewed for the first time, lead by PhD scientist **Katie Bouman. [Using machine learning]**

https://www.nationalgeographic.com/science/2019/04/first-picture-black-hole-revealed-m87-event-horizon-telescope-astrophysics/

Intriguingly Katie used machine learning to help reveal said black hole. Notably, Katie is not a trained astrophysicist, however her creativity has contributed, and the power of machine learning, has yet again produced extraordinary results. I ponder, is her work Nobel Prize worthy?

https://en.wikipedia.org/wiki/Katie_Bouman#Research_and_career


Installation
==
1. Install [ISR 2.0.2](https://pypi.org/project/ISR/2.0.2).

2. README Update: The newest version of ISR after 2.0.2 differs in its weight download strategy. I found the old weights I had used with ISR 2.0.2, by first installing and triggering the latest ISR 2.2 RDN model creation, finding out the weight location being auto-downloaded, then uninstalling latest ISR, and reinstalling ISR 2.0.2 that I had initially used, while utilizing the old "ArtefactCancelling" name to try a url with that prefix, along with the old file name, which successfully pointed to an aws location. 
  * Download [the weights from the aws resource related to the original repository](https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/ISR/rdn-C6-D20-G64-G064-x2/ArtefactCancelling/rdn-C6-D20-G64-G064-x2_ArtefactCancelling_epoch219.hdf5), (or download it [from my google drive](https://drive.google.com/file/d/1QweYEIDCtel-m1G5GMIxBZeGxy5yAb0i/view?usp=sharing)) and place it in the [/weights/sample_weights/rdn-C6-D20-G64-G064-x2/](https://github.com/JordanMicahBennett/EINSTEIN-BLACK-HOLE-PHOTOGRAPH-ENHANCEMENT/tree/master/source-code/weights/sample_weights/rdn-C6-D20-G64-G064-x2/ArtefactCancelling) of this source code. Adjust the "rdn.model.load_weights" location parameter to yours. 
  * If you get a file not found error and everything seems correct, maybe your Python [MAX_PATH limits directory length to 260 characters](https://stackoverflow.com/questions/1880321/why-does-the-260-character-path-length-limit-exist-in-windows). In that case, simply rename the long EINSTEIN...folder name to something maybe one word short, then readjust the location above, and re-run application.py.

3. Ensure tensorflow/gpu backend is enabled. On your windows machine, go to _root_/Users/_username_/.keras/keras.json, and change backend and device to "tensorflow" and "gpu" respectively, for fast gpu parralelization at inference. Inference is ~2 minutes on gtx 1060 8gb, i7 6700 machine.


The power of machine learning 
==
This is yet another scenario where machine learning has produced excellent results in a field, by a person, where said person isn't trained in said field.

Katie is not trained in Astrophysics, yet through machine learning, and her creativity, excellent results have been produced in the field of Astrophysics :]


The power of Science overall
==
Katie's work also marks yet another way to further validate Einstein's General Relativity theory. This is another chance to see how magnificent Einstein really was, and also a chance to appreciate how tremendous and effective Science is as a tool.

Youtube/Vox: [Why this black hole photo is such a big deal](https://www.youtube.com/watch?v=pAoEHR4aW8I)


Something to remember about this repository
==
An increase in resolution does not neccesitate that there's an increase in the accuracy of the information produced via original model!!
