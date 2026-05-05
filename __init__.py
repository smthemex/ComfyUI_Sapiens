
from comfy_api.latest import ComfyExtension, io
from typing_extensions import override
from .sapiens_node import SapiensLoader,SapiensSampler,SapiensSplit

class Sapiens_SM_Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            SapiensLoader,
            SapiensSampler,
            SapiensSplit,
        ]
async def comfy_entrypoint() -> Sapiens_SM_Extension:
    return Sapiens_SM_Extension()
