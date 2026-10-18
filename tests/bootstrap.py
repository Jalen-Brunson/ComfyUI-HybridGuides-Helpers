import importlib
import os
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
COMFY = Path(os.environ.get("COMFYUI_PATH", str(ROOT.parents[1])))
MMH3 = Path(os.environ.get("MMH3TOOLS_PATH", str(COMFY/'custom_nodes/ComfyUI-MMH3Tools')))
VHS = Path(os.environ.get("VHS_PATH", str(COMFY/'custom_nodes/comfyui-videohelpersuite')))
sys.path[:0] = [str(COMFY), str(MMH3), str(VHS)]
sys.argv = [sys.argv[0], '--cpu']
import comfy.options
comfy.options.enable_args_parsing()
import torch
import nodes
import server
from aiohttp import web

server.PromptServer.instance = types.SimpleNamespace(prompt_queue=None, routes=web.RouteTableDef())

package = types.ModuleType('hybridguides_helpers')
package.__path__ = [str(ROOT)]
sys.modules[package.__name__] = package
base = importlib.import_module('mmh3tools.nodes_multiprompt')
nodes.NODE_CLASS_MAPPINGS['MMH3ReferenceMultiPrompt'] = base.MMH3ReferenceMultiPrompt
torch.set_num_threads(2)
