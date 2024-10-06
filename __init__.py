import easy_nodes
easy_nodes.initialize_easy_nodes(default_category="my_category", auto_register=False)

# This must come after calling initialize_easy_nodes.
from  .Cairn_Nodes  import * # noqa: E402

NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = easy_nodes.get_node_mappings()

# Export so that ComfyUI can pick them up.
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# Optional: export the node list to a file so that e.g. ComfyUI-Manager can pick it up.
easy_nodes.save_node_list(os.path.join(os.path.dirname(__file__), "node_list.json"))