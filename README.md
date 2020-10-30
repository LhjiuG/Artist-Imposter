# Neural-Style-Transfer
Simple neural style transfer model based on VGG19

The NST is an algorithm that try to manipulate a "*Content*" image to look more
like a "*Style*" image (e.g., make a portrait of someone look like it was painted).

## What was my goal?
My main goal when doing this project was to make me more familiar with Tensorflow. The math behind the algorithm being quite simple allowed me to focus on understanding the code.

## What did i learn?
* Simple image manipulation using Tensorflow methods.
* Importing and using pretrained model from Tensorflow.
* Creating custom training loop.
* Using outputs from different layers.
* Using PIL and Matplotlib

## What problems did I encounter?
Most of the problems I encountered were not really problems, but more dissatisfaction of the result I got. In order to obtain something that I felt was good enough, I had to make multiple adjustments in the hyperparameter of the model (learning rate and decay, momentum, the number of iterations, weight...).

## Results
Here you can see a mix between a paint and an image containing the Canadian interceptor jet Avro Arrow.
![alt text](https://github.com/LhjiuG/Artist-Imposter/blob/main/ade.png?raw=true)



<!-- 
## TODO: 
1. Refractor the code
2. Add more image preprocessing, mainly with color modification.

## What I would like to do:
1. I would like to find a way to differ the intensity of the style on the generated image on different part of the image.
			
## What I have to fix:
1. Decrease the amount of artifact in the generated image.
 -->
