# Guide to Implementing a New Model Interface

This guide explains how to add support for a new biomechanical model format (e.g., from a different software) to this project. By following these steps, you can ensure your new model interface is compatible with the existing application structure.

The core of the system relies on a set of Abstract Base Classes (ABCs) that define a standardized "contract" for what a model interface must provide. These are found in `abstract_model_interface.py` and are crucial for ensuring interoperability.

---

## 🏗️ Core Architecture: The ABCs

Before you begin, familiarize yourself with these key abstract classes:

* **`AbstractSegment`**: This defines the required properties for a single body or segment within a model. Any new segment class you create must provide implementations for properties like `.name`, `.id`, and `.mass`.
* **`AbstractModelNoMesh`**: This is the main ABC for any model. It defines all the essential methods and properties for accessing model kinematics and kinetics, such as degrees of freedom, marker positions, and centers of mass.
* **`AbstractModel`**: This class inherits from `AbstractModelNoMesh` and adds methods specifically for handling 3D mesh data for visualization.

---

## 📝 Step-by-Step Implementation Guide

To add a new interface, you will create a new Python file (e.g., `newformat_model_interface.py`) and implement classes that inherit from the ABCs.

### **Step 1: Implement the Segment Class**

First, create a class for the model's segments that inherits from `AbstractSegment`. You must implement all of its abstract properties.

* **Required Properties**: `name`, `id`, `mass`, `has_mesh`, `has_meshlines`, `mesh_path`, and `mesh_scale_factor`.

```python
from .abstract_model_interface import AbstractSegment

class NewFormatSegment(AbstractSegment):
    # ... implementation for all abstract properties ...
```
### Step 2: Implement the Core Model Class
Next, create the main model class that inherits from AbstractModelNoMesh. 
This class will contain the logic for loading your model file and calculating biomechanical values.

Key Responsibilities:
- Load the model from a file path.
- Implement methods to access kinematics like `markers(q)` and `segment_homogeneous_matrices_in_global(q, segment_index)`.
- Implement properties to describe the model, such as `nb_q`, `dof_names`, `nb_markers`, and `marker_names`.

### Step 3: Implement the Mesh-Enabled Model Class
If your model has visual meshes, create a second class that inherits from your class 
from Step 2 and from AbstractModel.

Required Methods:
- `meshlines(self)`: Returns the vertices for mesh line drawings.
- `mesh_homogenous_matrices_in_global(...)`: Returns the transformation matrix for a specific mesh in the global frame.
You may also override the .segments property to filter for segments that have a mesh, as seen in BiorbdModel and OsimModel.

### Step 4: Handling Unsupported Features
If your model format does not support a specific feature required by the ABC (e.g., ligaments or soft contacts), you must still implement the method. The standard practice is to return a sensible default value or raise an error.

For numbers (`nb_ligaments`): Return 0.
For booleans (`has_soft_contacts`): Return False.
For lists or tuples: Return an empty collection (e.g., [] or ()).
For data that cannot have a default: Raise a `NotImplementedError`, as seen in the `OsimModelNoMesh` for ligament names.