import numpy as np


def extract_number_from_line(line: str, pattern: str) -> int:
    """
    Extracts the number from a given pattern in a line.

    Parameters
    ----------
    line: str
        The line from which to extract the number.
    pattern: str
        The pattern to look for in the line.

    Returns
    -------
    int: The extracted number.
    """
    start_index = line.find(pattern) + len(pattern)
    end_index = line[start_index:].find('"')
    return int(line[start_index : start_index + end_index])


def handle_polygons_shape(mesh_dictionary: dict, polygon_apex_idx: np.ndarray) -> np.ndarray:
    """
    Handles the shape of the polygons array.

    Parameters
    ----------
    mesh_dictionary: dict
        The dictionary containing the mesh data.
    polygon_apex_idx: np.ndarray
        The polygon apex indices.

    Returns
    -------
        np.ndarray: The updated polygon apex indices.
    """
    if polygon_apex_idx.size > mesh_dictionary["polygons"].shape[1]:
        mat = np.zeros((mesh_dictionary["polygons"].shape[0], polygon_apex_idx.size))
        mat[:, : mesh_dictionary["polygons"].shape[1]] = mesh_dictionary["polygons"]
        diff = polygon_apex_idx.size - mesh_dictionary["polygons"].shape[1]
        mat[:, mesh_dictionary["polygons"].shape[1] :] = np.repeat(
            mesh_dictionary["polygons"][:, -1].reshape(-1, 1), diff, axis=1
        )
        mesh_dictionary["polygons"] = mat
    elif polygon_apex_idx.size < mesh_dictionary["polygons"].shape[1]:
        diff = mesh_dictionary["polygons"].shape[1] - polygon_apex_idx.size
        polygon_apex_idx = np.hstack([polygon_apex_idx, np.repeat(None, diff)])
    return polygon_apex_idx


def read_vtp_file(filename: str) -> dict:
    """
    Reads a VTP file and extracts the mesh data.

    Parameters
    ----------
    filename: str
        The name of the VTP file to read.

    Returns
    -------
    dict: A dictionary containing the mesh data.
        - "N_Obj": 1 (Only 1 object per file)
        - "normals": np.ndarray (The normals)
        - "nodes": np.ndarray (The nodes)
        - "polygons": np.ndarray (The polygons)

    """

    mesh_dictionary = {"N_Obj": 1}  # Only 1 object per file

    with open(filename, "r") as file:
        content = file.readlines()

    type_ = None
    i = 0

    for ligne in content:
        if "<Piece" in ligne:
            num_points = extract_number_from_line(ligne, 'NumberOfPoints="')
            num_polys = extract_number_from_line(ligne, 'NumberOfPolys="')

            mesh_dictionary["normals"] = np.zeros((num_points, 3))
            mesh_dictionary["nodes"] = np.zeros((num_points, 3))
            mesh_dictionary["polygons"] = np.zeros((num_polys, 3))

        elif '<PointData Normals="Normals">' in ligne:
            type_ = "normals"
            i = 0
        elif "<Points>" in ligne:
            type_ = "nodes"
            i = 0
        elif "<Polys>" in ligne:
            type_ = "polygons"
            i = 0
        elif 'Name="offsets"' in ligne:
            type_ = None
        elif "<" not in ligne and type_ is not None:
            i += 1
            tmp = np.fromstring(ligne, sep=" ")

            if type_ == "polygons":
                tmp = handle_polygons_shape(mesh_dictionary=mesh_dictionary, polygon_apex_idx=tmp)

            if mesh_dictionary[type_][i - 1, :].shape[0] == 3 and tmp.shape[0] == 6:
                raise NotImplementedError("This vtp file cannot be cleaned yet to get triangles.")

            mesh_dictionary[type_][i - 1, :] = tmp

    return mesh_dictionary
