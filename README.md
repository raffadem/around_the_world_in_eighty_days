# Around the world in eighty days

## Table of Contents

1. [General Info](#general-info)
2. [Usage](#usage)
2. [Technologies](#technologies)
3. [Authors](#authors)
4. [License](#license)

### General Info

Like a new Phileas Fogg you have the desire to travel around the world always moving east, could you do it in 80 days?
The aim of the project is to help you and show you the best way to fulfill your dream.
The starting point is the city of London (GB), but it could be the one you want.

#### Dataset definition

The traveler can go through cities in the [worldcities.xlsx](worldcities.xlsx) dataset, from which we extract information about the city name, the country, the iso 2 (unique identifier code of the country), longitude and latitude, and the population. 
Each city is assigned a weight based on the following criteria:
1. **Distance**: at each step, according to the increasing Euclidean distance, the 3 closest cities are assigned values 2, 4, 8 respectively. To perform this calculation, the function [euclidean_distances](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.euclidean_distances.html), of sklearn library, is used.
2. **Population**: a weight of value 2 is added if the city has a population greater than or equal to 200 thousand inhabitants.
3. **Country**: a weight of value 2 is added if the city in the next step is located in a different country than the previous one.

#### Grid variability

At each step, the algorithm calculates the distance of the cities present within a **rectangle**.
It has as its **base** the distance between the longitude of the starting city, and a subsequent point at a variable distance given as input (x_size). 
The **height** changes depending on if the current visited city is northern/southern (the latitude is greater/lesser) than the starting city:
* if the current city is *northern* that the starting city, it has as its height  the distance generated between the latitude of the current city, adding a quantity (y_size/2) and the latitude of the starting city, subtracting a quantity (y_size/2) 
* if the current city is *southern* that the starting city, it has as its height  the distance generated between the latitude of the starting city, adding a quantity (y_size/2) and the latitude of the current city, subtracting a quantity (y_size/2) 

This last operation is differentiated between north and south to ensure that the traveler does not move too much towards these two directions but always moves to the east (to the right). 
The size of the polygon taken into account at each step is variable  to ensure the functionality of the algorithm with a minimum number of at least 3 cities. 
In particular, in addition to the variability of the size of the input values (x_size, y_size) which guarantees the starting size of the polygon, the absence in the rectangle of at least 3 cities makes these dimensions vary by a multiplicative rise_factor (by default equal to 2).

#### Path
The algorithm that is obtained taking as values of x_size, y_size, and rise_factor those of default (x_size=0.3 and y_size=0.15 and rise_factor=2) guarantees that the traveler returns to London before the 80 days, in fact, it ends after 666 steps and 62.42 days (1498 hours).

## Usage

See the notebook document [main.ipynb](main.ipynb) to see how to run the algorithm.

## Technologies

The algorithm is written in [python 3.8+](https://docs.python.org/3/) and mainly uses [pandas](https://pandas.pydata.org/docs/), [matplotlib](https://matplotlib.org/contents.html) and [sklearn](https://scikit-learn.org/stable/user_guide.html) libraries.
It consists of 2 files, the python file [around_the_world.py](around_the_world.py) containing the class, and the notebook document [main.ipynb](main.ipynb) with the functions to run the algorithm.

## Authors

* **Raffaella Michaela DeMarco**
* **Pietro Russo**

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for more details.