ImageClassification_based_on_BOF
================================
This is a comparision experiment of some feature coding method,including VQ + SPM, ScSPM and LLC.
The whole flow mainly includs feature extracted, dictionary learning, feature coding and classfication.
In this code, feature extracted and feature coding are on single image , and then save them to disk in a corresponding address.
when used in other stage, loading it. Seen details in the following.

1. demo.m demonstrates the whole flow, modificating the value of conf.codemethod to be 'psix_coding', 'sc_coding', 'LLC_coding'
makes 'VQ + SPM', 'ScSPM', 'LLC' running respectively. conf.numWords and conf.feature('dsift or phow') can also be modified.

2. CalculatePhowDescriptor.m implements feature extracted of all images in the assigned dataset(Caltech101). 
In this function, vl_phow() is used(Need to add vl_feat library, see http://www.vlfeat.org/).
using retr_database_dir.m to directly retrive feature database when it already exists.

3. Dictionary is Trained on a set of sift feature (randomly selected same amount of sift features in each image, seen in normalize_feature.m)
by kmeans, which is implemented by vl_kmeans().

4. In the feature coding stage:
(1)The VQ + SPM method directly uses vl_feat codes, seen in CalculateHistDescriptor.m;
(2)The ScSPM and LLC method directly uses their released code in their homepage respectively, seen in folder sc_coding and LLC_coding.
(http://www.ifp.illinois.edu/~jyang29/ScSPM.htm)
(http://www.ifp.illinois.edu/~jyang29/LLC.htm)

5. vl_svmtrain() is used to classify the final_code of each image after feature coding. Note that in the VQ + SPM method,
vl_homkermap() is needed before classify, which implements a higher dimension nonlinear mapping.

6. In each class, 30 or 15 images (which can be assigned by conf.numTrain)to be selected for training, and others for testing.
