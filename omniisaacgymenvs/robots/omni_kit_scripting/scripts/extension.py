# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
from .command import *
from .loader.omni_finder_loader import enable_omni_finder_loader, disable_omni_finder_loader
from .properties_widget import ScriptingProperties
from .script_manager import ScriptManager
from .script_editor import ScriptEditor
from .utils import open_script_file

import omni.ext
import os

OPEN_SCRIPT_HANDLER = "Open Python Script Handler"
NEW_PYTHON_SCRIPT_BEHAVIOR = "New Python Script (BehaviorScript)"
NEW_PYTHON_SCRIPT_EMPTY = "New Python Script (Empty)"
EDIT_PYTHON_SCRIPT = "Edit Python Script"


class Extension(omni.ext.IExt):

    _instance = None

    def on_startup(self, ext_id):
        enable_omni_finder_loader()
        self._properties = ScriptingProperties()
        self._script_editor = ScriptEditor()
        self._script_mgr = ScriptManager()
        self._add_content_browser_ui()
        Extension._instance = self

    def on_shutdown(self):
        self._remove_content_browser_ui()
        self._script_editor = None
        if self._script_mgr:
            self._script_mgr.destroy()
            self._script_mgr = None
        if self._properties:
            self._properties.destroy()
            self._properties = None
        disable_omni_finder_loader()
        Extension._instance = None

    @classmethod
    def get_instance(cls):
        return cls._instance

    def get_script_editor(self):
        return self._script_editor

    def _add_content_browser_ui(self):
        omni.kit.window.content_browser.get_instance().add_file_open_handler(
            OPEN_SCRIPT_HANDLER,
            open_fn=lambda script: open_script_file(script),
            file_type=lambda x: str(x).endswith(".py")
        )
        omni.kit.window.content_browser.get_instance().add_context_menu(
            name=EDIT_PYTHON_SCRIPT,
            glyph="pencil.svg",
            click_fn=lambda name, path: open_script_file(path),
            show_fn=lambda path: str(path).endswith(".py"),
            index=3
        )
        ext_path = omni.kit.app.get_app().get_extension_manager().get_extension_path_by_module(__name__)
        template_file_path = os.path.join(ext_path, "data/template_behavior_script.py")
        omni.kit.window.content_browser.get_instance().add_listview_menu(
            name=NEW_PYTHON_SCRIPT_BEHAVIOR,
            glyph="file.svg",
            click_fn=lambda name, path: self._script_editor.create_script_file(path, template_file_path),
            show_fn=None,
            index=3,
        )
        omni.kit.window.content_browser.get_instance().add_listview_menu(
            name=NEW_PYTHON_SCRIPT_EMPTY,
            glyph="file.svg",
            click_fn=lambda name, path: self._script_editor.create_script_file(path, None),
            show_fn=None,
            index=4,
        )

    def _remove_content_browser_ui(self):
        omni.kit.window.content_browser.get_instance().delete_file_open_handler(
            OPEN_SCRIPT_HANDLER)
        omni.kit.window.content_browser.get_instance().delete_context_menu(
            EDIT_PYTHON_SCRIPT)
        omni.kit.window.content_browser.get_instance().delete_listview_menu(
            NEW_PYTHON_SCRIPT_BEHAVIOR)
        omni.kit.window.content_browser.get_instance().delete_listview_menu(
            NEW_PYTHON_SCRIPT_EMPTY)
