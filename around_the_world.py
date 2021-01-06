import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from IPython.display import clear_output
import matplotlib.patches as patches
import time
from typing import List, Tuple, Union
import warnings

warnings.filterwarnings("ignore")


class AroundTheWorld(object):
    """
    Given a starting city, it travels around the world returning to the city of departure.

    Attributes
    ----------
    dataframe : pd.DataFrame
        Dataset of all cities
    city_start : str
        Name of the starting city
    country_start : str
        Name of the starting country
    n_min : int
        Number of the closest cities to which it is possible to travel
    x_size : float
        Size of the longitudinal side of the grid used to search for the nearest cities
    y_size : float
        Size of the latitudinal side of the grid used to search for the nearest cities
    rise_factor : float
        Multiplication factor to increase the grid used to search for the nearest cities
    map_city : pd.DataFrame
        Dataset of all cities of the journey route
    hours : int
        Total number of journey hours
    n_steps : int
        Total number of steps of the journey
    """

    def __init__(self,
                 dataframe: pd.DataFrame,
                 city_start: str,
                 country_start: str,
                 n_min: int = 3,
                 x_size: float = 0.3,
                 y_size: float = 0.15,
                 rise_factor: float = 2):
        # If the dataframe is not a pandas.Dataframe it rise a TypeError
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(f"The given dataframe must to be a \"pandas.Dataframe\" type, not \"{type(dataframe)}\"")
        self.dataframe: pd.DataFrame = self._initialise_dataframe(dataframe)
        # If the city start is not a string it rise a TypeError
        if not isinstance(city_start, str):
            raise TypeError(f"The given city_start must to be a \"str\" type, not \"{type(city_start)}\"")
        self.city_start: str = city_start
        # If the country start is not a string it rise a TypeError
        if not isinstance(country_start, str):
            raise TypeError(f"The given country_start must to be a \"str\" type, not \"{type(country_start)}\"")
        self.country_start: str = country_start
        # If the n_min is not a integer it rise a TypeError
        if not isinstance(n_min, int):
            raise TypeError(f"The given n_min must to be a \"int\" type, not \"{type(n_min)}\"")
        self.n_min: int = n_min
        # If the x_size is not a integer or a float it rise a TypeError
        if not isinstance(x_size, (int, float)):
            raise TypeError(f"The given x_size must to be a \"int\" or \"float\" type, not \"{type(x_size)}\"")
        self.x_size: float = float(x_size)
        # If the y_size is not a integer or a float it rise a TypeError
        if not isinstance(y_size, (int, float)):
            raise TypeError(f"The given y_size must to be a \"int\" or \"float\" type, not \"{type(y_size)}\"")
        self.y_size: float = float(y_size)
        # If the rise_factor is not a integer or a float it rise a TypeError
        if not isinstance(rise_factor, (int, float)):
            raise TypeError(f"The given y_size must to be a \"int\" or \"float\" type, not \"{type(rise_factor)}\"")
        self.rise_factor: float = float(rise_factor)

        self.map_city: pd.DataFrame = pd.DataFrame([], columns=["city", "lat", "lng", "country", "iso2"])
        self.hours: int = 0
        self.n_steps: int = 0
        # _x_size_default : float
        #    Default size of the longitudinal side of the grid used to search for the nearest cities
        # _y_size_default : float
        #    Default size of the latitudinal side of the grid used to search for the nearest cities
        self._x_size_default: float = self.x_size
        self._y_size_default: float = self.y_size
        # _coord_start : Tuple[float, float]
        #    Coordinates of the starting city declared as (longitude, latitude)
        self._coord_start: Tuple[float, float] = None  # (lng, lat)
        # _city_step : str
        #    Name of the current city of the journey route
        # _country_step : str
        #    Name of the current country of the journey route
        # _coord_step : Tuple[float, float]
        #    Coordinates of the current city of the journey route declared as (longitude, latitude)
        self._city_step: str = self.city_start
        self._country_step: str = self.country_start
        self._coord_step: Tuple[float, float] = None  # (lng, lat)
        # _lat_max : float
        #    Maximum latitude used as a reference
        # _lat_min : float
        #    Minimum latitude used as a reference
        # _lng_max : float
        #    Maximum longitude used as a reference
        # _lng_min : float
        #    Minimum longitude used as a reference
        self._lat_max: float = None
        self._lat_min: float = None
        self._lng_min: float = None
        self._lng_max: float = None
        # _is_near_destination : bool
        #    Minimum longitude used as a reference
        self._is_near_destination: bool = False
        # Set coordinates of the starting city, and take it as current coordinates
        self._set_coordinates()

    @staticmethod
    def _initialise_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
        """Given a dataframe with all cities, it selects interested columns,
        it adds the flag population and visited city columns.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Dataset of all cities

        Returns
        -------
        pd.DataFrame
            The initialised dataframe
        """
        # Prepare a list with the interested column names
        columns = ["city", "lat", "lng", "country", "iso2", "population"]
        # Deep copy of the dataframe
        df = dataframe[columns].copy()
        # Add the flag for the population, in the dataframe.
        # It is 1 if a city has a population is greater than 200000, 0 otherwise
        df["flg_pop"] = df.population.apply(lambda x: 1 if not (pd.isna(x)) and x > 200000 else 0)
        # Initialise the visited city column to 0
        df["visited_city"] = 0
        return df

    def _set_coordinates(self):
        """Set the longitude and the latitude of the starting city, and take it as current coordinates
        """
        # Get the row that contains the starting city and country
        temp_dataframe = self.dataframe.loc[
            (self.dataframe["city"] == self.city_start) &
            (self.dataframe["iso2"] == self.country_start)]
        # Select the longitude and the latitude of the previous dataframe,
        # get the first row, add values in a tuple and store it as coordinates start
        self._coord_start = tuple(temp_dataframe[["lng", "lat"]].iloc[0])
        # Store the previous coordinates as coordinates step too
        self._coord_step = self._coord_start

    def generate_grid(self):
        """Generate the grid used to search for the nearest cities
        as range of maximum and minimum latitudes and longitudes.
        """
        # If the latitude of the current city is greater than the latitude of the starting city
        if self._coord_step[1] >= self._coord_start[1]:
            # As latitude max is used the latitude of the current city plus half of the size of y_size
            self._lat_max = self._coord_step[1] + self.y_size / 2
            # As latitude min is used the latitude of the starting city minus half of the size of y_size
            self._lat_min = self._coord_start[1] - self.y_size / 2
        else:
            # As latitude max is used the latitude of the starting city plus half of the size of y_size
            self._lat_max = self._coord_start[1] + self.y_size / 2
            # As latitude min is used the latitude of the current city minus half of the size of y_size
            self._lat_min = self._coord_step[1] - self.y_size / 2

        # The longitude min is the longitude of the current city
        self._lng_min = self._coord_step[0]

        # If it is near the destination
        if self._is_near_destination:
            # The longitude max is slightly higher than the starting longitude
            self._lng_max = self._coord_start[0] + 0.005
        else:
            # The longitude max is the longitude of the current city plus x_size
            self._lng_max = self._coord_step[0] + self.x_size

        # Fix the latitude values of the grid
        if self._lat_max >= 90:
            self._lat_max -= 180
        if self._lat_min <= -90:
            self._lat_min += 180
        # Fix the longitude values of the grid
        if self._lng_max >= 180:
            self._lng_max -= 360
        if self._lng_min <= -180:
            self._lng_min += 360
        # print(self.lat_max, self.lat_min, self.lng_min, self.lng_max)

    def query(self) -> pd.DataFrame:
        """Select the cities that are located within the grid,
        shaped by the range of maximum and minimum latitudes and longitudes.

        Returns
        -------
        pd.DataFrame
            Dataframe that contains the cities that are inside the grid
        """
        # Initialise the conditions for the latitude and the longitude
        lat_condition = False
        lng_condition = False

        # If both latitude values of the grid are positive or negative
        if (self._lat_min >= 0 and self._lat_max >= 0) or (self._lat_min <= 0 and self._lat_max <= 0):
            lat_condition = (self.dataframe["lat"] >= self._lat_min) & (self.dataframe["lat"] <= self._lat_max)
        # If latitude min is positive and latitude max is negative
        elif self._lat_min >= 0 and self._lat_max <= 0:
            # The condition considers the highest and the lowest value of the latitude values
            lat_condition = ((self.dataframe["lat"] >= self._lat_min) & (self.dataframe["lat"] <= 90)) | (
                    (self.dataframe["lat"] >= -90) & (self.dataframe["lat"] <= self._lat_max))
        # If latitude min is negative and latitude max is positive
        elif self._lat_min <= 0 and self._lat_max >= 0:
            # The condition considers the 0 value between the latitude values
            lat_condition = ((self.dataframe["lat"] >= self._lat_min) & (self.dataframe["lat"] <= 0)) | (
                    (self.dataframe["lat"] >= 0) & (self.dataframe["lat"] <= self._lat_max))

        # If both longitude values of the grid are positive or negative
        if (self._lng_min >= 0 and self._lng_max >= 0) or (self._lng_min <= 0 and self._lng_max <= 0):
            lng_condition = (self.dataframe["lng"] >= self._lng_min) & (self.dataframe["lng"] <= self._lng_max)
        # If longitude min is positive and longitude max is negative
        elif self._lng_min >= 0 and self._lng_max <= 0:
            # The condition considers the highest and the lowest value of the longitude values
            lng_condition = ((self.dataframe["lng"] >= self._lng_min) & (self.dataframe["lng"] <= 180)) | (
                    (self.dataframe["lng"] >= -180) & (self.dataframe["lng"] <= self._lng_max))
        # If longitude min is negative and longitude max is positive
        elif self._lng_min <= 0 and self._lng_max >= 0:
            # The condition considers the 0 value between the longitude values
            lng_condition = ((self.dataframe["lng"] >= self._lng_min) & (self.dataframe["lng"] <= 0)) | (
                    (self.dataframe["lng"] >= 0) & (self.dataframe["lng"] <= self._lng_max))

        # Select the rows that verifies the latitude and longitude condition
        grid_city = self.dataframe.loc[lat_condition & lng_condition &
                                       (self.dataframe["city"] != self._city_step) &
                                       (self.dataframe["visited_city"] == 0)]
        # print(grid_city)
        # It returns the dataframe that contains the cities that are inside the grid
        return grid_city

    def check_grid_city(self) -> pd.DataFrame:
        """It generates a grid, it selects the cities in the grid and:
        if it contains at least n_min cities, it returns the dataframe
        if it doesn't contain at least n_min cities, it rises the grid size and recalculate the grid
        and reselect cities inside the new grid

        Returns
        -------
        pd.DataFrame
            Dataframe that contains the cities that are inside the grid
        """
        # Initialise variables
        n_row = 0
        grid_city = pd.DataFrame()

        while n_row < self.n_min:
            # Generate the grid used to search for the nearest cities
            self.generate_grid()
            # Select the cities that are inside the grid
            grid_city = self.query()
            # Get the number of the rows of the dataframe
            n_row = grid_city.shape[0]
            # It rises the grid size
            self.x_size *= self.rise_factor
            self.y_size *= self.rise_factor
            # Check if the current city is close to the starting city
            # and the latter is in the grid
            self._is_near_destination = self._check_is_near_destination(grid_city)

        # It assigns the default grid size
        self.x_size = self._x_size_default
        self.y_size = self._y_size_default
        # It returns the dataframe that contains the cities that are inside the grid
        return grid_city

    def _check_is_near_destination(self, grid_city: pd.DataFrame) -> bool:
        """It checks:
         if in the grid there is the starting city
         if the longitude of the current city is close to the longitude of the starting city

        Returns
        -------
        bool
            It is true if the grid contains the starting city and
            the longitude of the current city and the starting city are close.
            It is false otherwise.
        """
        # Check if the coordinates of the starting city are inside the grid
        is_destination_in_grid = len(
            grid_city.loc[(grid_city["lng"] == self._coord_start[0]) & (grid_city["lat"] == self._coord_start[1])]) == 1
        # Check if the longitude of the current city is close to the longitude of the starting city
        is_longitude_close = abs(self._coord_start[0] - self._coord_step[0]) <= (self.x_size * self.rise_factor)
        # Check if both of the previous conditions are true
        return is_destination_in_grid and is_longitude_close

    def weight(self, grid_city: pd.DataFrame) -> pd.DataFrame:
        """It adds the distance column to the dataframe grid_city, 
        extracts the 3 nearest cities and assigns them a weight, respectively the values 2,4,8.
        It sums the weight obtained previously with the one of population and country.

        Parameters
        ----------
        grid_city : pd.DataFrame
            Dataframe of cities that are inside the grid

        Returns
        -------
        pd.DataFrame
            Dataframe that contains the nearest cities with assigned weights
        """
        # Calculate the distance between the current coordinates and the coordinates of the cities in the grid
        grid_city["distance"] = grid_city.apply(self.calculate_distance, axis=1, point_city_step=self._coord_step)
        # Sort the cities by distance
        grid_city.sort_values(by=["distance"], inplace=True)
        # Get the first n_min rows (the nearest cities)
        grid_city = grid_city[:self.n_min]
        # Add a weight given by the distance,
        # the first nearest city has weight 2, the second 4 and the third 8
        grid_city["weight"] = [2 ** x for x in range(1, self.n_min + 1)]
        # Add a weight given by the population of the city and the different country
        grid_city["weight"] += grid_city.apply(self.calculate_weight, axis=1, country_step=self._country_step)
        # Returns the dataframe with the nearest cities and the assigned weight
        return grid_city

    @staticmethod
    def calculate_weight(row: pd.Series, country_step: str) -> int:
        """It sums the weight of the population and of the country

        Parameters
        ----------
        row: pd.Series
            Row of the dataframe of cities that are inside the grid
        country_step: str
            Country of the current city

        Returns
        -------
        int
            Weight that sums the weights if the population is big and if the country is different
        """
        # It assigns 2 if the population of the city is greater than 200000, 0 otherwise
        pop_weight = 2 if row["flg_pop"] == 1 else 0
        # It assigns 2 if the country is different from the current country, 0 otherwise
        country_weight = 2 if row["iso2"] != country_step else 0
        # It returns the sum of the previous weights
        return pop_weight + country_weight

    @staticmethod
    def calculate_distance(row: pd.Series, point_city_step: Tuple[float, float]) -> float:
        """It calculates the euclidean distance between the coordinates of a city and the current city

        Parameters
        ----------
        row: pd.Series
            Row of the dataframe of cities that are inside the grid
        point_city_step: Tuple[float, float]
            Coordinates of the current city

        Returns
        -------
        float
            Distance between the coordinates of a city and the current city
        """
        return euclidean_distances([list(point_city_step)], [[row["lng"], row["lat"]]])[0][0]

    def stop_condition(self, grid_city: pd.DataFrame) -> pd.DataFrame:
        """It checks if the coordinates of the start city are inside the grid:
        if it is true, it returns a dataframe with only the row of the starting city
        if it is false, it does nothing and it returns the given dataframe

        Parameters
        ----------
        grid_city : pd.DataFrame
            Dataframe of cities that are inside the grid

        Returns
        -------
        pd.DataFrame
            Dataframe that contains the nearest cities or only the start city
        """
        # Select the row that contains the starting coordinates from the grid dataframe
        temp_grid_city = grid_city.loc[
            (grid_city["lng"] == self._coord_start[0]) &
            (grid_city["lat"] == self._coord_start[1])]
        # If the starting coordinates are inside the grid dataframe
        if len(temp_grid_city) == 1:
            # It assigns the row of the starting city to the grid city
            grid_city = temp_grid_city
        # It returns the grid city
        return grid_city

    def step(self, grid_city: pd.DataFrame):
        """It gets the smallest weight from the nearest cities and it updates:
        the current city values, the hours, the number of steps, the visited city column of the dataframe,
        it adds the selected city to the list of all visited cities

        Parameters
        ----------
        grid_city : pd.DataFrame
            Dataframe of cities that are inside the grid
        """
        # Sort nearest cities by the weight and distance
        # For the same weight, the city with the shortest distance is taken
        grid_city.sort_values(by=["weight", "distance"], ascending=[True, False], inplace=True)
        # Select the first row
        step = grid_city.iloc[0]
        # Update the current city values: coordinates, city and country
        self._coord_step = tuple(step[["lng", "lat"]])
        self._city_step = step["city"]
        self._country_step = step["iso2"]
        # Increase the hours of the journey
        self.hours += step["weight"]
        # Increase the number of steps
        self.n_steps += 1
        # Mark the visited city
        self.dataframe["visited_city"].iloc[step.name] = 1
        # Add the nearest city to the list of all visited cities
        self.map_city = self.map_city.append(step[["city", "lat", "lng", "country", "iso2"]])
        # print(self.coord_step, self.city_step, self.country_step, self.hours)

    def travel(self, is_intermediate_plot: bool = True, n_intermediate_step: int = 100, is_clear_output: bool = False):
        """It calculates the cities to visit until it returns to the starting city.
        It plots the journey every n_intermediate_step, if is_intermediate_plot is true

        Parameters
        ----------
        is_intermediate_plot: bool
            If it is true, it plots the intermediate journey
        n_intermediate_step: int
            Number of the step the plot is printed
        is_clear_output: bool
            If it is true, the previous plot is cleared
        """
        # Get the start time
        start = time.time()
        # Plot the world, if the variable is true
        if is_intermediate_plot:
            self.plot_world(is_clear_output)
        # While the current city is different from the starting city or it is the first step
        while (self.city_start != self._city_step) or (self.country_start != self._country_step) or (self.n_steps == 0):
            # Generate a grid that contains at least n_min nearest cities
            grid_city = self.check_grid_city()
            # Calculate the weight of the n_min nearest cities
            grid_city_weight = self.weight(grid_city)
            # Check if the starting city is in the grid
            grid_city_weight = self.stop_condition(grid_city_weight)
            # Updates the current city, the number of hours of the journey and so on
            self.step(grid_city_weight)
            # Plot the intermediate journey, if the variable is true, every n_intermediate_step
            if is_intermediate_plot and self.n_steps % n_intermediate_step == 0:
                self.plot_world(is_clear_output)
        # Plot the world with the completed journey, if the variable is true
        self.plot_world(is_clear_output)
        print(f"Completed the journey starting from {self.city_start} ({self.country_start})",
              f"in {self.hours / 24:.2f} days ({self.hours} hours) after visited {self.n_steps} cities.")
        # Get and print the elapsed time
        elapsed = time.time() - start
        print(f"Done in {elapsed:.2f} seconds.")

    def plot_world(self, is_clear_output: bool = False):
        """It plots the world with the journey path

        Parameters
        ----------
        is_clear_output: bool
            If it is true, the previous plot is cleared
        """
        # Clear the previous plot, if it is true
        if is_clear_output:
            clear_output(wait=True)
        # Select the visited cities
        visited_df = self.dataframe.loc[(self.dataframe["visited_city"] == 1)]
        # Select the remaining cities
        city_df = self.dataframe.loc[(self.dataframe["visited_city"] == 0)]
        # Prepare the plot figure and a subplot
        fig = plt.figure(figsize=(100, 50))
        ax = fig.add_subplot(111)
        # Add a scatter plot of the not visited cities
        ax.scatter(city_df.lng, city_df.lat, color="green", marker="o", alpha=0.5, s=30)
        # Add a scatter plot of the visited cities
        ax.scatter(visited_df.lng, visited_df.lat, color="magenta", marker="o", alpha=0.5, s=200)
        # Add the point of starting coordinates
        ax.scatter(self._coord_start[0], self._coord_start[1], color="red", marker="+", s=100)
        # Assign labels
        plt.xlabel("Longitude", fontsize=75)
        plt.ylabel("Latitude", fontsize=75)
        # Set axes limits
        plt.xlim([-180, 180])
        plt.ylim([-90, 90])
        # Add the grid
        ax.grid()

        # If the latitude and longitude values exists
        if (self._lng_min is not None and self._lng_max is not None and
                self._lat_min is not None and self._lat_max is not None):
            # Prepare the rectangles according to the coordinates max and min of the grid
            rectangle_values = self._prepare_rectangle_values()
            for origin_point, width, height in rectangle_values:
                # Add the rectangle of the grid
                ax.add_patch(
                    patches.Rectangle(
                        xy=origin_point,  # Point of origin
                        width=width,
                        height=height,
                        linewidth=3,
                        color="blue",
                        fill=False
                    )
                )
        # Plot the figure
        plt.show()

    def _prepare_rectangle_values(self) -> List[Tuple[Tuple[float, float], float, float]]:
        """Calculate the point of the origin, the width and the height of the rectangle,
        according to the coordinates max and min of the grid

        Returns
        -------
        List[Tuple[Tuple[float, float], float, float]]
            A list of tuples of point of the origin, the width and the height of the rectangle
        """
        rectangle_values: List[Tuple[Tuple[float, float], float, float]] = []  # point of origin, width, height
        # If longitude min and max are both positive or negative,
        # or the longitude min is negative and the longitude max is positive
        if (self._lng_min >= 0 and self._lng_max >= 0) or \
                (self._lng_min <= 0 and self._lng_max <= 0) or \
                (self._lng_min <= 0 and self._lng_max >= 0):
            # Prepare only one rectangle
            rectangle_values = [
                ((self._lng_min, self._lat_min),
                 self._lng_max - self._lng_min,
                 self._lat_max - self._lat_min)]
        else:
            # Otherwise, there are 2 rectangles, one on the far right.
            # the other on the far left
            rectangle_values = [
                ((self._lng_min, self._lat_min),
                 180 - self._lng_min,
                 self._lat_max - self._lat_min),
                ((-180, self._lat_min),
                 self._lng_max + 180,
                 self._lat_max - self._lat_min)
            ]
        # Return rectangle values
        return rectangle_values
