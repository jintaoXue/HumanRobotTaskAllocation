# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
from .utils import Prompt, open_script_file

import omni.ui as ui
import omni.usd
from omni.kit.property.usd.usd_property_widget import UsdPropertiesWidget
from omni.kit.property.usd.prim_selection_payload import PrimSelectionPayload
from omni.kit.property.usd.widgets import ICON_PATH

from pxr import Sdf, Usd
from pxr import OmniScriptingSchema
from pathlib import Path


REMOVE_BUTTON_STYLE = style = {"image_url": str(Path(ICON_PATH).joinpath("remove.svg")), "margin": 0, "padding": 0}

SCRIPTS_ATTR = "omni:scripting:scripts"
SCRIPTS_DISPLAY_NAME = "Scripts"


class ScriptingProperties:
    def __init__(self):
        self._registered = False
        if self._registered:
            return
        self._add_menus = []
        import omni.kit.window.property as p
        w = p.get_window()
        if w:
            w.register_widget("prim", "omni_scripting_api", ScriptingAPIPropertiesWidget("Python Scripting"))
            self._registered = True

    def destroy(self):
        if not self._registered:
            return
        self._add_menus = []
        import omni.kit.window.property as p
        w = p.get_window()
        if w:
            w.unregister_widget("prim", "omni_scripting_api")


class ScriptingAPIPropertiesWidget(UsdPropertiesWidget):
    def __init__(self, title: str):
        super().__init__(title, collapsed=False, multi_edit=True)
        self._title = title
        from omni.kit.property.usd import PrimPathWidget
        self._add_button_menus = []
        self._add_button_menus.append(
            PrimPathWidget.add_button_menu_entry(
                "Python Scripting",
                show_fn=self.on_show_scripting,
                onclick_fn=self.on_click_scripting
            )
        )
        self._stage_picker = None
        self._asset_pciker = None

    def destroy(self):
        from omni.kit.property.usd import PrimPathWidget
        for menu in self._add_button_menus:
            PrimPathWidget.remove_button_menu_entry(menu)
        self._add_button_menus = []

    def on_show_scripting(self, objects: dict):
        if "prim_list" not in objects or "stage" not in objects:
            return False
        stage = objects["stage"]
        if not stage:
            return False
        prim_list = objects["prim_list"]
        if len(prim_list) < 1:
            return False
        for item in prim_list:
            if isinstance(item, Sdf.Path):
                prim = stage.GetPrimAtPath(item)
            elif isinstance(item, Usd.Prim):
                prim = item
            if not prim.HasAPI(OmniScriptingSchema.OmniScriptingAPI):
                return True
        return False

    def on_click_scripting(self, payload: PrimSelectionPayload):
        if payload is None:
            return Sdf.Path.emptyPath
        prim_paths = payload.get_paths()
        omni.kit.commands.execute("ApplyScriptingAPICommand", paths=prim_paths)
        omni.kit.commands.execute("RefreshScriptingPropertyWindowCommand")
        return prim_paths

    def on_new_payload(self, payload):
        if not super().on_new_payload(payload):
            return False
        if not self._payload or len(self._payload) == 0:
            return False
        for prim_path in payload:
            prim = self._get_prim(prim_path)
            if not prim:
                return False
            else:
                attrs = prim.GetProperties()
                for attr in attrs:
                    attr_name_str = attr.GetName()
                    if attr_name_str == SCRIPTS_ATTR:
                        return True
                return False
        return False

    def _filter_props_to_build(self, props):
        return [prop for prop in props if prop.GetName().startswith(SCRIPTS_ATTR)]

    def get_additional_kwargs(self, ui_attr):

        def edit_entry(path):
            open_script_file(path.resolvedPath)

        return None, {"on_edit_fn": edit_entry}

    def _customize_props_layout(self, attrs):
        for attr in attrs:
            if attr.attr_name == SCRIPTS_ATTR:
                attr.override_display_name(SCRIPTS_DISPLAY_NAME)
        return attrs

    def build_impl(self):
        if self._collapsable:
            self._collapsable_frame = ui.CollapsableFrame(
                self._title, build_header_fn=self._build_custom_frame_header, collapsed=self._collapsed
            )

            def on_collapsed_changed(collapsed):
                self._collapsed = collapsed

            self._collapsable_frame.set_collapsed_changed_fn(on_collapsed_changed)
        else:
            self._collapsable_frame = ui.Frame(height=10, style={"Frame": {"padding": 5}})
        self._collapsable_frame.set_build_fn(self._build_frame)

    def _on_remove_with_prompt(self):
        def on_remove_scripting_api():
            selected_paths = omni.usd.get_context().get_selection().get_selected_prim_paths()
            omni.kit.commands.execute("RemoveScriptingAPICommand", paths=selected_paths)
            omni.kit.commands.execute("RefreshScriptingPropertyWindowCommand")
        prompt = Prompt(
            "Remove Python Scripting?", "Are you sure you want to remove the 'Python Scripting' component?", "Yes", "No",
            ok_button_fn=lambda: on_remove_scripting_api(), modal=True
        )
        prompt.show()

    def _build_custom_frame_header(self, collapsed, text):
        if collapsed:
            alignment = ui.Alignment.RIGHT_CENTER
            width = 5
            height = 7
        else:
            alignment = ui.Alignment.CENTER_BOTTOM
            width = 7
            height = 5

        with ui.HStack(spacing=8):
            with ui.VStack(width=0):
                ui.Spacer()
                ui.Triangle(
                    style_type_name_override="CollapsableFrame.Header", width=width, height=height, alignment=alignment
                )
                ui.Spacer()
            ui.Label(text, style_type_name_override="CollapsableFrame.Header")
            with ui.HStack(width=0):
                ui.Spacer(width=8)
                with ui.VStack(width=0):
                    ui.Spacer(height=5)
                    ui.Button(identifier="RemovePythonScriptingButton", style=REMOVE_BUTTON_STYLE, height=16, width=16).set_mouse_pressed_fn(lambda *_: self._on_remove_with_prompt())
                ui.Spacer(width=5)
