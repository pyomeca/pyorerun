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
        - "polygons": np.ndarray (The polygons, always triangulated)

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

    # Triangulate polygons if necessary
    if mesh_dictionary["polygons"].shape[1] > 3:
        mesh_dictionary["polygons"], mesh_dictionary["nodes"], mesh_dictionary["normals"] = (
            transform_polygon_to_triangles(
                mesh_dictionary["polygons"],
                mesh_dictionary["nodes"],
                mesh_dictionary["normals"],
            )
        )

    return mesh_dictionary


def transform_polygon_to_triangles(polygons, nodes, normals) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform any polygons with more than 3 edges into polygons with 3 edges (triangles)."""

    if polygons.shape[1] == 3:
        return polygons, nodes, normals

    elif polygons.shape[1] == 4:
        return convert_quadrangles_to_triangles(polygons, nodes, normals)

    elif polygons.shape[1] > 4:
        return convert_polygon_to_triangles(polygons, nodes, normals)

    else:
        raise RuntimeError("The polygons array must have at least 3 columns.")


def norm2(v):
    """Compute the squared norm of each row of the matrix v."""
    return np.sum(v**2, axis=1)


def convert_quadrangles_to_triangles(polygons, nodes, normals) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform polygons with 4 edges (quadrangles) into polygons with 3 edges (triangles)."""
    # 1. Search for quadrangles
    quadrangles_idx = np.where((polygons[:, 3] != 0) & (~np.isnan(polygons[:, 3])))[0]
    triangles_idx = np.where(np.isnan(polygons[:, 3]))[0]

    # transform polygons[quadrangles, X] as a list of int
    polygons_0 = polygons[quadrangles_idx, 0].astype(int)
    polygons_1 = polygons[quadrangles_idx, 1].astype(int)
    polygons_2 = polygons[quadrangles_idx, 2].astype(int)
    polygons_3 = polygons[quadrangles_idx, 3].astype(int)

    # 2. Determine triangles to be made
    mH = 0.5 * (nodes[polygons_0] + nodes[polygons_2])  # Barycentres AC
    mK = 0.5 * (nodes[polygons_1] + nodes[polygons_3])  # Barycentres BD
    KH = mH - mK
    AC = -nodes[polygons_0] + nodes[polygons_2]  # Vector AC
    BD = -nodes[polygons_1] + nodes[polygons_3]  # Vector BD
    # Search for the optimal segment for the quadrangle cut
    type_ = np.sign((np.sum(KH * BD, axis=1) / norm2(BD)) ** 2 - (np.sum(KH * AC, axis=1) / norm2(AC)) ** 2)

    # 3. Creation of new triangles
    tBD = np.where(type_ >= 0)[0]
    tAC = np.where(type_ < 0)[0]
    # For BD
    PBD_1 = np.column_stack(
        [polygons[quadrangles_idx[tBD], 0], polygons[quadrangles_idx[tBD], 1], polygons[quadrangles_idx[tBD], 3]]
    )
    PBD_2 = np.column_stack(
        [polygons[quadrangles_idx[tBD], 1], polygons[quadrangles_idx[tBD], 2], polygons[quadrangles_idx[tBD], 3]]
    )
    # For AC
    PAC_1 = np.column_stack(
        [polygons[quadrangles_idx[tAC], 0], polygons[quadrangles_idx[tAC], 1], polygons[quadrangles_idx[tAC], 2]]
    )
    PAC_2 = np.column_stack(
        [polygons[quadrangles_idx[tAC], 2], polygons[quadrangles_idx[tAC], 3], polygons[quadrangles_idx[tAC], 0]]
    )

    # 4. Matrix of final polygons
    new_polygons = np.vstack([polygons[triangles_idx, :3], PBD_1, PBD_2, PAC_1, PAC_2])

    return new_polygons, nodes, normals


def convert_polygon_to_triangles(polygons, nodes, normals) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Transform any polygons with more than 3 edges into polygons with 3 edges (triangles).
    """

    # Search for polygons with more than 3 edges
    polygons_with_more_than_3_edges = np.where((polygons[:, 3] != 0) & (~np.isnan(polygons[:, 3])))[0]
    polygons_with_3_edges = np.where(np.isnan(polygons[:, 3]))[0]

    triangles = []
    for j, poly_idx in enumerate(polygons_with_more_than_3_edges):
        # get only the non-nan values
        current_polygon = polygons[poly_idx, np.isnan(polygons[poly_idx]) == False]
        # Split the polygons into triangles
        # For simplicity, we'll use vertex 0 as the common vertex and form triangles:
        # (0, 1, 2), (0, 2, 3), (0, 3, 4), ..., (0, n-2, n-1)

        for i in range(1, current_polygon.shape[0] - 1):
            triangles.append(np.column_stack([polygons[poly_idx, 0], polygons[poly_idx, i], polygons[poly_idx, i + 1]]))

    return (
        np.vstack([polygons[polygons_with_3_edges, :3], *triangles]),
        nodes,
        normals,
    )
