from graphviz import Digraph

def add_subtree(dot, parent, structure):
    """
    Recursively add nodes/edges to a graphviz Digraph
    based on nested dict schema.
    """
    for key, val in structure.items():
        node_id = f"{parent}_{key}"
        dot.node(node_id, key, shape="folder" if isinstance(val, dict) else "note")
        dot.edge(parent, node_id)
        if isinstance(val, dict):
            add_subtree(dot, node_id, val)

def make_schema_diagram(output_file="hdf5_schema"):
    # Define the schema as nested dict
    schema = {
        "CL": {
            "raw": {"image_001 [256x256x1024]": None,
                    "image_002 [256x256x1024]": None},
            "processed": {"image_001_processed [256x256x3]": None},
            "cutouts": {
                "from_image_001": {
                    "raw": {"cutout_00 [15x15x1024]": None,
                            "cutout_01 [15x15x1024]": None},
                    "processed": {"cutout_00 [15x15x3]": None,
                            "cutout_01 [15x15x3]": None}                            
                }
            },
            "metadata": {
                "acquisition_params": {"image_001_params": None},
                "cutout_coords": {"image_001_cutouts [N x 2]": None}
            }
        },
        "AFM": {
            "raw": {"afm_001 [512x512]": None},
            "processed": {"afm_001_filtered [512x512]": None},
            "cutouts": {
                "from_afm_001": {"cutout_00 [15x15]": None}
            },
            "manual labels": {
                "afm_001_manual_labels.csv": None,
                "afm_002_manual_labels.csv": None
            },
            "metadata": {
                "acquisition_params": {"afm_001_params": None},
                "cutout_coords": {"afm_001_cutouts [N x 2]": None}
            }
        },
        "alignment": {
            "CSVs": {
                "image_001_alignment_points.csv": None,
                "image_002_alignment_points.csv": None
            },
            "transformers": {
                "image_001_transformers.pkl": None
            }
        }
    }
    

    # Build graph
    dot = Digraph(comment="Database", format="png")
    dot.attr(rankdir="TB", fontsize="10")

    # Root node
    root = "root"
    dot.node(root, "AFM and Hyperspectral CL Dataset", shape="box3d", style="filled", color="lightblue")

    # Add recursively
    add_subtree(dot, root, schema)

    # Render to file
    dot.render(output_file, view=True)
    print(f"Schema diagram saved as {output_file}.png")

if __name__ == "__main__":
    make_schema_diagram()
