# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
from .utils import refresh_property_window

import omni.kit.commands
from omni.kit.usd_undo import UsdLayerUndo

from pxr import Sdf
from pxr import OmniScriptingSchema, OmniScriptingSchemaTools

from typing import List


class ApplyScriptingAPICommand(omni.kit.commands.Command):
    def __init__(
        self,
        layer: Sdf.Layer = None,
        paths: List[Sdf.Path] = []
    ):
        self._usd_undo = None
        self._layer = layer
        self._paths = paths

    def do(self):
        stage = omni.usd.get_context().get_stage()
        if self._layer is None:
            self._layer = stage.GetEditTarget().GetLayer()
        self._usd_undo = UsdLayerUndo(self._layer)
        for path in self._paths:
            if not stage.GetPrimAtPath(path).HasAPI(OmniScriptingSchema.OmniScriptingAPI):
                self._usd_undo.reserve(path)
                OmniScriptingSchemaTools.applyOmniScriptingAPI(stage, path)
                prim = stage.GetPrimAtPath(path)
                attr = prim.GetAttribute("omni:scripting:scripts")
                supported_exts = {"*.py": "Python File"}
                attr.SetCustomDataByKey("fileExts", supported_exts)

    def undo(self):
        if self._usd_undo is not None:
            self._usd_undo.undo()


class RemoveScriptingAPICommand(omni.kit.commands.Command):
    def __init__(
        self,
        layer: Sdf.Layer = None,
        paths: List[Sdf.Path] = []
    ):
        self._usd_undo = None
        self._layer = layer
        self._paths = paths

    def do(self):
        stage = omni.usd.get_context().get_stage()
        if self._layer is None:
            self._layer = stage.GetEditTarget().GetLayer()
        self._usd_undo = UsdLayerUndo(self._layer)
        for path in self._paths:
            if stage.GetPrimAtPath(path).HasAPI(OmniScriptingSchema.OmniScriptingAPI):
                self._usd_undo.reserve(path)
                OmniScriptingSchemaTools.removeOmniScriptingAPI(stage, path)

    def undo(self):
        if self._usd_undo is not None:
            self._usd_undo.undo()


class RefreshScriptingPropertyWindowCommand(omni.kit.commands.Command):
    def __init__(self):
        pass

    def do(self):
        refresh_property_window()

    def undo(self):
        pass


omni.kit.commands.register(ApplyScriptingAPICommand)
omni.kit.commands.register(RemoveScriptingAPICommand)
omni.kit.commands.register(RefreshScriptingPropertyWindowCommand)
