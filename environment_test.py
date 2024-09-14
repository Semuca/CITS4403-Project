import matplotlib.pyplot as plt
from scipy.signal import correlate2d
import numpy as np
from map import map

## Parameters
ROOM_TEMPERATURE_CELSIUS = 25
HEAT_FIRE_STARTS_CELSIUS = 100

# Fire parameters
FIRE_EMISSION_CELSIUS = 200
FIRE_MAX_HEAT_CELSIUS = 1000
FIRE_EMISSION_KERNEL = np.array([[0, 0.6, 0], [0.6, 0.6, 0.6], [0, 0.6, 0]])

# Heat parameters
HEAT_DIFFUSION = 0.1

# Display parameters
MAX_HEAT_DISPLAY_CELSIUS = 100
MIN_HEAT_DISPLAY_CELSIUS = 25
MAX_HEAT_TRANSPARENCY = 0.5

class Cell:
    def __init__(self, isBurnable):
        self.heatCelsius = ROOM_TEMPERATURE_CELSIUS
        self.isBurnable = isBurnable

        self.isOnFire = False

class BurnSimulation():
    def __init__(self, map):
        self.environment = np.array([[Cell(isBurnable) for isBurnable in row] for row in map])

        self.environment[4][5].heatCelsius = 150
        self.environment[4][5].isOnFire = True

    def step(self):
        # Calculate heat dispersion
        heatMap = np.array([[cell.heatCelsius for cell in row] for row in self.environment])
        heatDispersion = correlate2d(heatMap, [[0, HEAT_DIFFUSION, 0], [HEAT_DIFFUSION, 1 - 4 * HEAT_DIFFUSION, HEAT_DIFFUSION], [0, HEAT_DIFFUSION, 0]], mode='same', boundary='fill', fillvalue=ROOM_TEMPERATURE_CELSIUS)

        # Calculate heat emissions from fire using convolution
        fires = np.array([[cell.isOnFire for cell in row] for row in self.environment])
        heatEmissions = correlate2d(fires, FIRE_EMISSION_KERNEL, mode='same', boundary='fill', fillvalue=0)

        # Add heat emissions to environment
        for i in range(len(self.environment)):
            for j in range(len(self.environment[i])):
                cell = self.environment[i][j]
                cell.heatCelsius = heatDispersion[i][j] + heatEmissions[i][j] * FIRE_EMISSION_CELSIUS
                cell.heatCelsius = min(cell.heatCelsius, FIRE_MAX_HEAT_CELSIUS)

                if cell.heatCelsius >= HEAT_FIRE_STARTS_CELSIUS and cell.isBurnable:
                    cell.isOnFire = True

        self.draw()

    def draw(self):
        height, width = map.shape

        plt.axis([0, height, 0, width])
        plt.xticks([])
        plt.yticks([])

        options={
            "cmap": 'Greens',
            "alpha": 0.7,
            "vmin": 0, "vmax": 1, 
            "interpolation": 'none', 
            "origin": 'upper',
            "extent": [0, height, 0, width]
        }

        plt.imshow(map, **options)

        # Draw heat on the environment
        heatMap = np.array([[min(cell.heatCelsius - MIN_HEAT_DISPLAY_CELSIUS, MAX_HEAT_DISPLAY_CELSIUS) / MAX_HEAT_TRANSPARENCY for cell in row] for row in self.environment])
        options={
            "cmap": 'Reds',
            "alpha": 0.5,
            "vmin": 0, "vmax": 1, 
            "interpolation": 'none', 
            "origin": 'upper',
            "extent": [0, height, 0, width]
        }

        plt.imshow(heatMap, **options)

        fireXs = []
        fireYs = []
        for y in range(height):
            for x in range(width):
                cell = self.environment[y][x]
                if cell.isOnFire:
                    fireXs.append(x + 0.5)
                    fireYs.append(height - y - 0.5)
        plt.plot(fireXs, fireYs, '.', color='red')
        plt.show()
    
sim = BurnSimulation(map)

while True:
    sim.draw()
    sim.step()