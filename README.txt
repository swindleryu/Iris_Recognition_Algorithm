0. Instruction:

1• Read and run the 'Main.ipynb' directly to see all the results.

2. Experiment design: 

• images from the first session will be used for training(3*108 classes) and images from the second session will be used for testing(4*108 classes).

• Create 7 templates for each training images with different rotation degrees in order to solve the rotation limitations of the algorithm, and pick the one that has the shortest distance among 7 for each class image. 

• Applied LDA and PCA to do feature selections, and compared performance on the 'Main.ipynb'. Same as stated in the paper, LDA gives better accuracy performance.




3. Experimental results:

• The Correct Recognition Rate (CRR) for the identification mode (refer to Tables 3 & 10 of Ma’s paper) (Check the Main function for details). Below are a sample of output for CRR.

          ########################################################################
	  ##                           Original Feature Set	Reduced Feature Set ##
          ## L1 distance measure	         0.925926	           0.902778 ##
          ## L2 distance measure	         0.891204	           0.909722 ##
          ## Cosine similarity measure	         0.895833	           0.930556 ##
          ########################################################################




4. Limitations:

• For now, the localization part is perfect for most of the pupil, only about 2% of the images might have a larger or smaller circles. However, the localization of outer boundary for iris needs to be improved better, since for now we have 93% accuracy for cosine similarity, even though which is not that bad. We can improve the localization part if we can come up with a more robust method to adjust radius according to each image. In this way, the circle might get a more perfect fit, and accuracy should be higher than what we got now.

