#!/usr/bin/env python
# coding: utf-8

import importlib
import inspect

MODULES = [
    "container_manager_mcp.container_manager",
    "container_manager_mcp.mcp",
    "container_manager_mcp.agent",
]

__all__ = []

for module_name in MODULES:
    module = importlib.import_module(module_name)
    for name, obj in inspect.getmembers(module):
        if (inspect.isclass(obj) or inspect.isfunction(obj)) and not name.startswith(
            "_"
        ):
            globals()[name] = obj
            __all__.append(name)

"""
container-manager

Manage your containers using docker, podman, compose, or docker swarm!
"""
