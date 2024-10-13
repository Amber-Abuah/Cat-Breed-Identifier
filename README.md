# Cat Breed Identifier

A Python app capable of predicting cat breeds from image inputs. The model is able to identify the following 12 breeds:
- Abyssinian, 
- Bengal, 
- Birman, 
- Bombay,
- British Shorthair, 
- Egyptian Mau, 
- Maine Coon, 
- Persian, 
- Ragdoll, 
- Russian Blue, 
- Siamese, 
- Sphynx.

Training Accuracy: 91.97%, Validation Accuracy: 71.68%.  
Trained on 50 epochs with a train-test-spilt seed of 1.

#### Image Preprocessing Techniques:
Before input into the convolutional layers the images undergo:
- Random horizontal and vertical flips,
- Random rotations in the range [-0.2 * 2pi, 0.2 * 2pi]
- Random zooms from 0 to 20%.

This allows the network to train on variations of the same image and identify cat breeds more accurately.

The following datasets were combined and used to train the model:  
**Oxford IIIT Cats**: https://www.kaggle.com/datasets/imbikramsaha/cat-breeds  
**CatBreedsRefined-7k**: https://www.kaggle.com/datasets/doctrinek/catbreedsrefined-7k

This allowed each class to have ~550 images for each breed.

To reduce overfitting Batch Normalisation layers were applied after every Convolutional layer and a Dropout layer was utilised within the first fully connected layer to allow the network to better generalise.