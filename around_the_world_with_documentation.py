import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from IPython.display import clear_output
import matplotlib.patches as patches
import warnings

warnings.filterwarnings('ignore')


class AroundTheWorld(object):
    """
    Given a starting city, it travels around the world returning to the city of departure.

    ...

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
        Size of the longitudinal side of the rectangle used to search for the nearest cities
    y_size : float
        Size of the latitudinal side of the rectangle used to search for the nearest cities
    rise_factor : float
        Multiplication factor to increase the rectangle used to search for the nearest cities

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
                 y_size: float = 0.1,
                 rise_factor: float = 2):
        self.dataframe = self._initialise_dataframe(dataframe)
        self.city_start = city_start
        self.country_start = country_start
        self.n_min = n_min
        self.x_size = x_size
        self.y_size = y_size
        self.rise_factor = rise_factor

        self.map_city = pd.DataFrame([], columns=["city", "lat", "lng", "country", "iso2"])
        self.hours = 0
        self.n_steps = 0
        """
        _x_size_default : float
            Default size of the longitudinal side of the rectangle used to search for the nearest cities
        _y_size_default : float
            Default size of the latitudinal side of the rectangle used to search for the nearest cities
        """
        self._x_size_default = x_size
        self._y_size_default = y_size
        """
        _coord_start : (float, float)
            Coordinates of the starting city declared as (longitude, latitude)
        """
        self._coord_start = None  # (lng, lat)
        """
        _city_step : str
            Name of the current city of the journey route
        country_start : str
            Name of the current country of the journey route
        _coord_step : (float, float)
            Coordinates of the current city of the journey route declared as (longitude, latitude)
        """
        self._city_step = city_start
        self._country_step = country_start
        self._coord_step = None
        """
        _lat_max : float
            Maximum latitude used as a reference
        _lat_min : float
            Minimum latitude used as a reference
        _lng_max : float
            Maximum longitude used as a reference
        _lng_min : float
            Minimum longitude used as a reference
        """
        self._lat_max, self._lat_min, self._lng_min, self._lng_max = None, None, None, None

        self._calculate_lat()

    @staticmethod
    def _initialise_dataframe(dataframe):
        # get a data frame with selected columns
        columns = ["city", "lat", "lng", "country", "iso2", "population"]
        df = dataframe[columns].copy()
        # insert in the dataframe the flag for the population
        df["flg_pop"] = df.population.apply(lambda x: 1 if not (pd.isna(x)) and x > 200000 else 0)
        df["visited_city"] = 0
        return df

    def _calculate_lat(self):
        temp_dataframe = self.dataframe.loc[
            (self.dataframe['city'] == self.city_start) &
            (self.dataframe['iso2'] == self.country_start)]
        self._coord_start = tuple(temp_dataframe[['lng', 'lat']].iloc[0])
        self._coord_step = self._coord_start

    def generate_grid(self):
        if self._coord_step[1] >= self._coord_start[1]:
            self._lat_max = self._coord_step[1] + self.y_size / 2
            self._lat_min = self._coord_start[1] - self.y_size / 2
        else:
            self._lat_max = self._coord_start[1] + self.y_size / 2
            self._lat_min = self._coord_step[1] - self.y_size / 2

        self._lng_min = self._coord_step[0]
        self._lng_max = self._coord_step[0] + self.x_size

        if self._lat_max >= 90:
            self._lat_max -= 180
        if self._lat_min <= -90:
            self._lat_min += 180
        if self._lng_max >= 180:
            self._lng_max -= 360
        if self._lng_min <= -180:
            self._lng_min += 360
        # print(self.lat_max, self.lat_min, self.lng_min, self.lng_max)

    def query(self):
        lat_condition = False
        lng_condition = False

        if (self._lat_min >= 0 and self._lat_max >= 0) or (self._lat_min <= 0 and self._lat_max <= 0):
            lat_condition = (self.dataframe["lat"] >= self._lat_min) & (self.dataframe["lat"] <= self._lat_max)
        elif self._lat_min >= 0 and self._lat_max <= 0:
            lat_condition = ((self.dataframe["lat"] >= self._lat_min) & (self.dataframe["lat"] <= 90)) | (
                    (self.dataframe["lat"] >= -90) & (self.dataframe["lat"] <= self._lat_max))
        elif self._lat_min <= 0 and self._lat_max >= 0:
            lat_condition = ((self.dataframe["lat"] >= self._lat_min) & (self.dataframe["lat"] <= 0)) | (
                    (self.dataframe["lat"] >= 0) & (self.dataframe["lat"] < self._lat_max))

        if (self._lng_min >= 0 and self._lng_max >= 0) or (self._lng_min <= 0 and self._lng_max <= 0):
            lng_condition = (self.dataframe["lng"] >= self._lng_min) & (self.dataframe["lng"] <= self._lng_max)
        elif self._lng_min >= 0 and self._lng_max <= 0:
            lng_condition = ((self.dataframe["lng"] >= self._lng_min) & (self.dataframe["lng"] <= 180)) | (
                    (self.dataframe["lng"] >= -180) & (self.dataframe["lng"] <= self._lng_max))
        elif self._lng_min <= 0 and self._lng_max >= 0:
            lng_condition = ((self.dataframe["lng"] >= self._lng_min) & (self.dataframe["lng"] <= 0)) | (
                    (self.dataframe["lng"] >= 0) & (self.dataframe["lng"] < self._lng_max))

        grid_city = self.dataframe.loc[lat_condition & lng_condition &
                                       (self.dataframe["city"] != self._city_step) &
                                       (self.dataframe["visited_city"] == 0)]
        # print(grid_city)
        return grid_city

    def check_if_stop_city_is_in_grid(self, grid_city):
        temp_grid_city = grid_city.loc[(grid_city["lng"] == self._coord_start[0]) &
                                       (grid_city["lat"] == self._coord_start[1])]
        return len(temp_grid_city) == 1

    def check_grid_city(self):
        self.generate_grid()
        grid_city = self.query()
        n_row = grid_city.shape[0]
        if n_row < self.n_min:
            self.x_size *= self.rise_factor
            self.y_size *= self.rise_factor
            return self.check_grid_city()
        else:
            self.x_size = self._x_size_default
            self.y_size = self._y_size_default
            return grid_city

    def weight(self, grid_city):
        grid_city["distance"] = grid_city.apply(self.calculate_distance, axis=1, point_city_step=self._coord_step)
        grid_city.sort_values(by=['distance'], inplace=True)

        grid_city = grid_city[:self.n_min]
        grid_city["weight"] = [2 ** x for x in range(1, self.n_min + 1)]
        grid_city["weight"] += grid_city.apply(self.calculate_weight, axis=1, country_step=self._country_step)

        return grid_city

    def stop_condition(self, grid_city):
        temp_grid_city = grid_city.loc[
            (grid_city["lng"] == self._coord_start[0]) & (grid_city["lat"] == self._coord_start[1])]
        if len(temp_grid_city) == 1:
            grid_city = temp_grid_city
        return grid_city

    @staticmethod  # decorator
    def calculate_weight(row, country_step):
        pop_weight = 2 if row['flg_pop'] == 1 else 0
        country_weight = 2 if row['iso2'] != country_step else 0
        return pop_weight + country_weight

    @staticmethod
    def calculate_distance(row, point_city_step):
        return euclidean_distances([list(point_city_step)], [[row['lng'], row['lat']]])[0][0]

    def step(self, grid_city):
        grid_city.sort_values(by=["weight", "distance"], ascending=[True, False], inplace=True)
        step = grid_city.iloc[0]
        self._coord_step = tuple(step[["lng", "lat"]])
        self._city_step = step["city"]
        self._country_step = step["iso2"]
        self.hours += step["weight"]
        self.dataframe["visited_city"].iloc[step.name] = 1
        self.map_city = self.map_city.append(step[["city", "lat", "lng", "country", "iso2"]])
        # print(self.coord_step, self.city_step, self.country_step, self.hours)

    def travel(self):
        while (self.city_start != self._city_step) or (self.country_start != self._country_step) or (self.n_steps == 0):
            grid_city = self.check_grid_city()
            grid_city_weight = self.weight(grid_city)
            grid_city_weight = self.stop_condition(grid_city_weight)
            self.step(grid_city_weight)
            self.n_steps += 1
            # print(self.dataframe)
            if self.n_steps != 0 and self.n_steps % 100 == 0:
                self.plot_world()
        self.plot_world()
        print(self.n_steps, self.hours)
        pd.set_option("display.max_rows", None, "display.max_columns", None)
        print(self.map_city.to_string(index=False))

    def plot_world(self):
        # clear_output(wait=True)
        visited_df = self.dataframe.loc[(self.dataframe["visited_city"] == 1)]
        city_df = self.dataframe.loc[(self.dataframe["visited_city"] == 0)]

        fig = plt.figure(figsize=(100, 50))
        ax = fig.add_subplot(111)
        ax.scatter(city_df.lng, city_df.lat, color="green", marker="o", alpha=0.5, s=30)
        ax.scatter(visited_df.lng, visited_df.lat, color="magenta", marker="o", alpha=0.5, s=200)
        ax.scatter(self._coord_start[0], self._coord_start[1], color="red", marker="o", s=100)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        ax.grid()

        ax.add_patch(
            patches.Rectangle(
                xy=(self._lng_min, self._lat_min),  # point of origin.
                width=self._lng_max - self._lng_min,
                height=self._lat_max - self._lat_min,
                linewidth=1,
                color='blue',
                fill=False
            )
        )

        plt.show()
