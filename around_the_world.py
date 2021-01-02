import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
import warnings

warnings.filterwarnings('ignore')
from IPython.display import clear_output
import matplotlib.patches as patches


class AroundTheWorld(object):

    def __init__(self, dataframe, city_start, country_start, n_min=3, x_size=0.3, y_size=0.1, rise_factor=2):
        self.dataframe = self._initialise_dataframe(dataframe)
        self.coord_step = None  # (lng, lat) city_step
        self.lat_max = None
        self.lat_min = None
        self.lng_min = None
        self.lng_max = None
        self.city_start = city_start
        self.country_start = country_start
        self.city_step = city_start
        self.country_step = country_start
        self.n_min = n_min
        self.rise_factor = rise_factor
        self.x_size = x_size
        self.y_size = y_size
        self.hours = 0
        self.n_step = 0
        self.x_size_default = x_size
        self.y_size_default = y_size
        self.lat_inf = None
        self.lat_sup = None
        self._calculate_lat()
        self.map_city = pd.DataFrame([], columns=["city", "lat", "lng", "country", "iso2"])
        # self.is_near_start_city = False

    def _initialise_dataframe(self, dataframe):
        # get a data frame with selected columns
        columns = ["city", "lat", "lng", "country", "iso2", "population"]
        df = dataframe[columns].copy()
        # insert in the dataframe the flag for the population
        df["flg_pop"] = df.population.apply(lambda x: 1 if not (pd.isna(x)) and x > 200000 else 0)
        df["visited_city"] = 0
        return df

    def _calculate_lat(self):
        self.coord_start = tuple(self.dataframe.loc[
                                     (self.dataframe['city'] == self.city_start) &
                                     (self.dataframe['iso2'] == self.country_start)][['lng', 'lat']].iloc[0])
        self.coord_step = self.coord_start

    def generate_grid(self):
        if self.coord_step[1] >= self.coord_start[1]:
            self.lat_max = self.coord_step[1] + self.y_size / 2
            self.lat_min = self.coord_start[1] - self.y_size / 2
        else:
            self.lat_max = self.coord_start[1] + self.y_size / 2
            self.lat_min = self.coord_step[1] - self.y_size / 2

        self.lng_min = self.coord_step[0]
        self.lng_max = self.coord_step[0] + self.x_size

        if self.lat_max >= 90:
            self.lat_max -= 180
        if self.lat_min <= -90:
            self.lat_min += 180
        if self.lng_max >= 180:
            self.lng_max -= 360
        if self.lng_min <= -180:
            self.lng_min += 360
        # print(self.lat_max, self.lat_min, self.lng_min, self.lng_max)

    def query(self):
        if (self.lat_min >= 0 and self.lat_max >= 0) or (self.lat_min <= 0 and self.lat_max <= 0):
            lat_condition = (self.dataframe["lat"] >= self.lat_min) & (self.dataframe["lat"] <= self.lat_max)
        elif self.lat_min >= 0 and self.lat_max <= 0:
            lat_condition = ((self.dataframe["lat"] >= self.lat_min) & (self.dataframe["lat"] <= 90)) | (
                    (self.dataframe["lat"] >= -90) & (self.dataframe["lat"] <= self.lat_max))
        elif self.lat_min <= 0 and self.lat_max >= 0:
            lat_condition = ((self.dataframe["lat"] >= self.lat_min) & (self.dataframe["lat"] <= 0)) | (
                    (self.dataframe["lat"] >= 0) & (self.dataframe["lat"] < self.lat_max))
        else:
            lat_condition = False

        if (self.lng_min >= 0 and self.lng_max >= 0) or (self.lng_min <= 0 and self.lng_max <= 0):
            lng_condition = (self.dataframe["lng"] >= self.lng_min) & (self.dataframe["lng"] <= self.lng_max)
        elif self.lng_min >= 0 and self.lng_max <= 0:
            lng_condition = ((self.dataframe["lng"] >= self.lng_min) & (self.dataframe["lng"] <= 180)) | (
                    (self.dataframe["lng"] >= -180) & (self.dataframe["lng"] <= self.lng_max))
        elif self.lng_min <= 0 and self.lng_max >= 0:
            lng_condition = ((self.dataframe["lng"] >= self.lng_min) & (self.dataframe["lng"] <= 0)) | (
                    (self.dataframe["lng"] >= 0) & (self.dataframe["lng"] < self.lng_max))
        else:
            lng_condition = False

        grid_city = self.dataframe.loc[(lat_condition) & (lng_condition) &
                                       (self.dataframe["city"] != self.city_step) &
                                       (self.dataframe["visited_city"] == 0)]
        # print(grid_city)
        return grid_city

    def check_if_stop_city_is_in_grid(self, grid_city):
        # print(grid_city)
        temp_grid_city = grid_city.loc[(grid_city["lng"] == self.coord_start[0]) &
                                       (grid_city["lat"] == self.coord_start[1])]
        # print(temp_df)
        check_condition = len(temp_grid_city) == 1
        return check_condition

    def check_grid_city(self):
        self.generate_grid()
        grid_city = self.query()
        n_row = grid_city.shape[0]
        if n_row < self.n_min:
            self.x_size *= self.rise_factor
            self.y_size *= self.rise_factor
            return self.check_grid_city()
        else:
            self.x_size = self.x_size_default
            self.y_size = self.y_size_default
            return grid_city
        # print(n_row)

    def weight(self, grid_city):

        grid_city["distance"] = grid_city.apply(self.calculate_distance, axis=1, point_city_step=self.coord_step)
        grid_city.sort_values(by=['distance'], inplace=True)

        grid_city = grid_city[:self.n_min]
        grid_city["weight"] = [2 ** x for x in range(1, self.n_min + 1)]
        grid_city["weight"] += grid_city.apply(self.calculate_weigth, axis=1, country_step=self.country_step)

        return grid_city

    def stop_condition(self, grid_city):
        temp_grid_city = grid_city.loc[
            (grid_city["lng"] == self.coord_start[0]) & (grid_city["lat"] == self.coord_start[1])]
        if len(temp_grid_city) == 1:
            grid_city = temp_grid_city
        return grid_city

    @staticmethod  # decorator
    def calculate_weigth(row, country_step):
        pop_weigth = 2 if row['flg_pop'] == 1 else 0
        country_weigth = 2 if row['iso2'] != country_step else 0
        return pop_weigth + country_weigth

    @staticmethod
    def calculate_distance(row, point_city_step):
        return euclidean_distances([list(point_city_step)], [[row['lng'], row['lat']]])[0][0]

    def step(self, grid_city):
        grid_city.sort_values(by=["weight", "distance"], ascending=[True, False], inplace=True)
        step = grid_city.iloc[0]
        self.coord_step = tuple(step[["lng", "lat"]])
        self.city_step = step["city"]
        self.country_step = step["iso2"]
        self.hours += step["weight"]
        self.dataframe["visited_city"].iloc[step.name] = 1
        self.map_city = self.map_city.append(step[["city", "lat", "lng", "country", "iso2"]])
        # print(self.coord_step, self.city_step, self.country_step, self.hours)

    def travel(self):
        while ((self.city_start != self.city_step) or (self.country_start != self.country_step) or (self.n_step == 0)):
            grid_city = self.check_grid_city()
            grid_city_weight = self.weight(grid_city)
            grid_city_weight = self.stop_condition(grid_city_weight)
            self.step(grid_city_weight)
            self.n_step += 1
            # print(self.dataframe)
            if self.n_step != 0 and self.n_step % 100 == 0:
                self.plot_world()
        self.plot_world()
        print(self.n_step, self.hours)
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
        ax.scatter(self.coord_start[0], self.coord_start[1], color="red", marker="o", s=100)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        ax.grid()

        ax.add_patch(
            patches.Rectangle(
                xy=(self.lng_min, self.lat_min),  # point of origin.
                width=self.lng_max - self.lng_min,
                height=self.lat_max - self.lat_min,
                linewidth=1,
                color='blue',
                fill=False
            )
        )

        plt.show()
