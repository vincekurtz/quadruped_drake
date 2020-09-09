This directory contains a basic URDF description of the ANYmal robot. 

This was adopted from [this repository](https://github.com/ANYbotics/anymal_b_simple_description) and modified to work with Drake. 
Specifically, `.dae` files were converted to `.obj` format using [this converter](https://products.aspose.app/3d/conversion/dae-to-obj)
and `<transmission>` elements were added for each of the actuated joints so that Drake parses the model correctly. 
