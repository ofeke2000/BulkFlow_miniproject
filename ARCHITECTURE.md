# Arcitecture

Here i will describe the purpose of the different parts of the code

## miniproj

### config (yaml)

It will be a file where I insert into the code everything that needs changing, from overdensity radius to file paths

### main (py)

That is the part where i combine my functions into an actual working code with purpose

## data

Where i keep the source data for rockstar and CF4

## output

A place to put all files at

## src

Where I'll put all my functions

### __init__ (py)

I have no idea what these is for, Chat GPT said it needs to be there for python to recognize it like a library

### dataloader (py)

Functions that loads data frames from the files in the wanted format, usually edites the files thus seperated from utils

### overdensity (py)

Calculates and addes the overdensity to the data frame for a given radius around a galaxy

### masks (py)

creates a new data frame which is masked, either with the CF4 data or uniformly. can be expended in the future

### bulkflow (py)

Functions that calculates the bulk flow of galaxies around a given dot in a given radii from a given data frame (could be masked if needed)

### visualize

Functions to create plots

### utils (py)

Functions which can be used for all around helping, contains:

1. logger setup
2. timing functions
3. ensur directory exists
4. save data frame

### specific_utils (py)

Like utils but specificly for calculations like:

1. periodic distance calculator
