# Gauss Stylization

This is a C++ implementation of "Gauss Stylization". It contains the code from "[Cubic Stylization](https://www.dgp.toronto.edu/projects/cubic-stylization/)" by [Hsueh-Ti Derek Liu](https://www.dgp.toronto.edu/~hsuehtil/) and [Alec Jacobson](https://www.cs.toronto.edu/~jacobson/).

### ImGui version
We offer an [ImGui](https://github.com/ocornut/imgui) version in folder `gaussStylization_ImGui` for one to play with the stylization interactively in the GUI. To compile the application, please type these commands in the terminal
```
cd gaussStylization_ImGui
mkdir build
cd build
cmake ..
make
```
This will create the executable of the gauss stylization. To start the application, please run
```
./gaussStylization_ImGui
```
where the example meshes are provided in `/meshes`. Instructions of how to control the gauss stylization is listed on the side of the GUI.


### Python version

We also offer a [pybind11](https://github.com/pybind/pybind11) interface in folder `gaussStylization_python`. To compile the python module, please type these commands in the terminal

```
cd gaussStylization_python
mkdir build
cd build
cmake ..
make
```
This will create a python module. To you can take a look in `gaussStylization_python/main.py` to see how to use it. Make sure, that your python script can import the module e.g. by putting a `__init__.py` in the `gaussStylization_python/build` folder.
