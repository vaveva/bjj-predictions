# BJJ submission predictor

This repository contains the code for an LSTM model that can predict what the most likely next submission is. The model takes a user given submission as input and outputs the next most likely one.

## Data collection

The data was crawled from the Finnish Brazilian Jiu-Jitsu Federation [website](https://bjjliitto.fi/?lang=en) and stored in an SQL database. Results of all of the federation's competitions from about 2011 onwards can be found online. The crawled data only includes results from the Finnish BJJ Federation's competitions, not for example ADCC or smaller competitions.

I will not publish the database at this point of time, but I am planning on providing an anonymized version of the database here at a later date.

## Training the model

The model here is an [LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory) model. During training, the model is given one submission as input. The model's output is then compared to the next submission in the sequence and the model weights are then adjusted based on whether the prediction is correct or not.

To be able to train the model, the submissions are mapped to numbers, which are then turned into tensors that can be fed to the model.

## What is BJJ

If you're confused by what exaclty BJJ is and what the submissions predicted here are, check out the [Wikipedia article on BJJ](https://en.wikipedia.org/wiki/Brazilian_jiu-jitsu).

## Future plans

The collected dataset the initial model was trained on is not quite large enough to train a model that can predict based on a larger window than one previous submission. Most people only get a couple of matches in a row in a competition, so it is only possible to look at one previous match when training the model to make predicitons. I am planning on using the initial model to augment the real data so that I can create data that has longer series of submissions, so predictions could be made based on two or three previous submissions instead of just one.