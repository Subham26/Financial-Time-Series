import numpy as np
import matplotlib.pyplot as plt


class StockPriceGenerator():
    """    Time-Series Data Generator    """

    def __init__(self, slope, period=365, amplitude=1, phase=0,
                 noise_level=1, seed=None):
        self.time = 0  # count days
        self.slope = slope
        self.period = period
        self.amplitude = amplitude
        self.phase =phase
        self.noise_level = noise_level
        self.seed = seed

        self.max = 4 * 365 + 1
        self.baseline = 10

    def trend(self):
        return self.slope * self.time

    def seasonal_pattern(self, season_time):
        """Just an arbitrary pattern, you can change it if you wish"""
        return np.where(season_time < 0.4,
                        np.cos(season_time * 2 * np.pi),
                        1 / np.exp(3 * season_time))

    def seasonality(self):
        """Repeats the same pattern at each period"""
        season_time = ((self.time + self.phase) % self.period) / self.period
        return self.amplitude * self.seasonal_pattern(season_time)

    def noise(self):
        rnd = np.random.RandomState(self.seed)
        return rnd.randn() * self.noise_level

    def __iter__(self):
        """returns the self object to be accessed by the for loop"""
        return self

    def __next__(self):
        
        """returns stock price up to 4 years"""
        
        if self.time >= self.max:
            raise StopIteration
        
        # Generate Time-Series Data
        element = self.baseline + self.trend() + self.seasonality()

        # Update with noise
        element += self.noise()

        self.time += 1

        return element


def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)
    

if __name__ == '__main__':
    time = np.arange(4 * 365 + 1, dtype="float32")
    amplitude = 40
    slope = 0.05
    noise_level = 5

    stock_price_generator = StockPriceGenerator(slope, amplitude, noise_level)
    stock_prices = [price for price in stock_price_generator]
    plot_series(time, stock_prices)
    plt.show()
