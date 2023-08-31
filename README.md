# RDB Labs Project

Our project is focused on the drone-navigation aspect of the course and room modeling in particular.

## Installation

1. Clone the repository by using ``git clone``, or alternatively download the ``.zip`` file.


2. Use the [pip](https://pip.pypa.io/en/stable/) python package installer to install the following packages:

```bash
pip install numpy
```
```bash
pip install matplotlib
```
```bash
pip install math
```
```bash
pip install pandas
```
```bash
pip install random
```

## Usage

If you haven't done so, please install the packages mentioned above, after which, please ensure that a `map.csv` file of the following format is present in the same directory:


| X |
| ---|
| $$x_1 \ y_1 \ z_1$$ |
| $$x_2 \ y_2 \ z_2$$ |
| $$\vdots$$ |
| $$x_n \ y_n \ z_n$$ |

> A table of size $n\times 1$, such that the single column is labeled X, the points are seperated by rows, and the coordinataes are seperated by spaces.

Once the `map.csv` file is present in the same directory, you are good to go! 
