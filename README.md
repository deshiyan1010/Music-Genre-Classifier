# Music-Genre-Classifier
This program can be trained and used to classify music. Currently it is can be trained to a validation accuracy of nearly 75% with marsyas dataset which is a benchmarker for music genre classifier.

# Download link for Marsyas Dataset
http://opihi.cs.uvic.ca/sound/genres.tar.gz

# How to use?

Step 1: Clone or download the code on to you machine.

        https://github.com/deshiyan1010/Music-Genre-Classifier/
Step 2: The XYZ is the root directory of the training dataset file. Run the data_preprocessing.py file as 

        python data_preprocessing.py -path XYZ
        Example: python data_preprocessing.py -path "/content/Music-Genre-Classifier/genres/"
        
Step 3: Run building_model_training.py.

        python building_model_training.py
        
Step 4: Run testing.py with the path to your music you want to classify.

        python testing.py -path XYZ
        Example:
        python testing.py -path "/content/Music-Genre-Classifier/genres/classical/classical.00002.wav"
        
        
 # Setting up the training dataset
 From the above examples:
 
 ![Image of Instruction](https://github.com/deshiyan1010/Music-Genre-Classifier/blob/master/Dataset%20folder%20instruction.png)
                        
                        
