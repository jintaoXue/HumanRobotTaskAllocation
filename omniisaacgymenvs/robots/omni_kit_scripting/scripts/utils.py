# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
from .loader.omni_finder_loader import OmniFinder
from .loader import omni_finder_loader
from pxr import Sdf, OmniScriptingSchema

import carb
import omni.client
import omni.usd
import omni.ui as ui
import omni.kit.window.property
import omni.kit.window.filepicker
import omni.kit.scripting

import asyncio
from urllib.parse import urlparse
import re
import sys
import traceback
import os


def is_path_relative(path: str) -> bool:
    """Takes in a local or Omniverse path and determines if it is relative

    Args:
        path: the path to be determined if it is relative
    """
    parse = urlparse(path)
    if len(parse.scheme) > 0:
        return False
    return True


def ensure_absolute_path(path: str) -> str:
    if not is_path_relative(path):
        return path
    else:
        stage_path = omni.usd.get_context().get_stage().GetRootLayer().identifier
        abs_path = omni.client.combine_urls(stage_path, path)
        return abs_path


def refresh_property_window():
    selected_paths = omni.usd.get_context().get_selection().get_selected_prim_paths()
    omni.usd.get_context().get_selection().clear_selected_prim_paths()
    omni.kit.window.property.get_window()._window.frame.rebuild()
    omni.usd.get_context().get_selection().set_selected_prim_paths(selected_paths, True)


class Prompt:
    def __init__(
        self,
        title,
        text,
        ok_button_text="OK",
        cancel_button_text=None,
        middle_button_text=None,
        ok_button_fn=None,
        cancel_button_fn=None,
        middle_button_fn=None,
        modal=False,
    ):
        self._title = title
        self._text = text
        self._cancel_button_text = cancel_button_text
        self._cancel_button_fn = cancel_button_fn
        self._ok_button_fn = ok_button_fn
        self._ok_button_text = ok_button_text
        self._middle_button_text = middle_button_text
        self._middle_button_fn = middle_button_fn
        self._modal = modal
        self._build_ui()

    def __del__(self):
        self._cancel_button_fn = None
        self._ok_button_fn = None

    def __enter__(self):
        self._window.show()
        return self

    def __exit__(self, type, value, trace):
        self._window.hide()

    def show(self):
        self._window.visible = True

    def hide(self):
        self._window.visible = False

    def is_visible(self):
        return self._window.visible

    def set_text(self, text):
        self._text_label.text = text

    def set_confirm_fn(self, on_ok_button_clicked):
        self._ok_button_fn = on_ok_button_clicked

    def set_cancel_fn(self, on_cancel_button_clicked):
        self._cancel_button_fn = on_cancel_button_clicked

    def set_middle_button_fn(self, on_middle_button_clicked):
        self._middle_button_fn = on_middle_button_clicked

    def _on_ok_button_fn(self):
        self.hide()
        if self._ok_button_fn:
            self._ok_button_fn()

    def _on_cancel_button_fn(self):
        self.hide()
        if self._cancel_button_fn:
            self._cancel_button_fn()

    def _on_middle_button_fn(self):
        self.hide()
        if self._middle_button_fn:
            self._middle_button_fn()

    def _build_ui(self):
        self._window = ui.Window(
            self._title, visible=False, height=0, dockPreference=ui.DockPreference.DISABLED
        )
        self._window.flags = (
            ui.WINDOW_FLAGS_NO_COLLAPSE
            | ui.WINDOW_FLAGS_NO_RESIZE
            | ui.WINDOW_FLAGS_NO_SCROLLBAR
            | ui.WINDOW_FLAGS_NO_RESIZE
            | ui.WINDOW_FLAGS_NO_MOVE
        )

        if self._modal:
            self._window.flags = self._window.flags | ui.WINDOW_FLAGS_MODAL

        with self._window.frame:
            with ui.VStack(height=0):
                ui.Spacer(width=0, height=10)
                with ui.HStack(height=0):
                    ui.Spacer()
                    self._text_label = ui.Label(self._text, word_wrap=True, width=self._window.width - 80, height=0)
                    ui.Spacer()
                ui.Spacer(width=0, height=10)
                with ui.HStack(height=0):
                    ui.Spacer(height=0)
                    if self._ok_button_text:
                        ok_button = ui.Button(self._ok_button_text, width=60, height=0)
                        ok_button.set_clicked_fn(self._on_ok_button_fn)
                    if self._middle_button_text:
                        middle_button = ui.Button(self._middle_button_text, width=60, height=0)
                        middle_button.set_clicked_fn(self._on_middle_button_fn)
                    if self._cancel_button_text:
                        cancel_button = ui.Button(self._cancel_button_text, width=60, height=0)
                        cancel_button.set_clicked_fn(self._on_cancel_button_fn)
                    ui.Spacer(height=0)
                ui.Spacer(width=0, height=10)


def show_security_popup(future):
    app = omni.kit.app.get_app()
    manager = app.get_extension_manager()
    if manager.is_extension_enabled("omni.kit.window.popup_dialog"):
        from omni.kit.window.popup_dialog import MessageDialog

        def on_cancel(dialog: MessageDialog):
            future.set_result(False)
            dialog.hide()

        def on_ok(dialog: MessageDialog):
            future.set_result(True)
            dialog.hide()

        message = """
                This USD contains Python Scripting Components assigned with scripts.

                Arbitrary code could be included in the scripts and executed with your system credentials.

                Only proceed with script execution if the author of this USD content is trusted.


                Do you want to still want to enable script execution ?
                """

        dialog = MessageDialog(
            title="Warning",
            width=600,
            message=message,
            cancel_handler=on_cancel,
            ok_handler=on_ok,
            ok_label="Yes",
            cancel_label="No"
        )

        async def show_async(dialog):
            await omni.kit.app.get_app().next_update_async()
            await omni.kit.app.get_app().next_update_async()
            dialog.show()

        asyncio.ensure_future(show_async(dialog))


_traceback_filename_regex = re.compile(r' *File \"(.*)\"')


def traceback_format_exception(callback_fn=None):
    msg = "Python Scripting Error:"
    exc_type, exc_value, exc_traceback = sys.exc_info()
    strings = traceback.format_exception(exc_type, exc_value, exc_traceback)
    carb.log_error(msg)
    # We hide
    in_runtime_namespace = True
    for s in strings:
        display = s.rstrip()
        if s.startswith("Traceback"):
            carb.log_error(display)
        if in_runtime_namespace:
            match = re.match(_traceback_filename_regex, display)
            if match:
                filepath = match.groups()[0]
                in_runtime_namespace = not OmniFinder.file_was_loaded_by_finder(filepath)

        if not in_runtime_namespace and not display.isspace():
            carb.log_error(display)
    if callback_fn:
        callback_fn(f"{msg} {exc_type.__name__}, {exc_value}")


import_dialog = None

def open_script_file(path):

    # If the python file's path begins with an http, it can't be edited. In
    # that case, we need to make a local copy and remap all references in
    # the scene to the local path so we can edit/iterate on it:
    if path.find("http") == 0:

        def on_copy_and_remap(local_path, orig_path):
            omni.client.copy(orig_path, local_path, behavior=omni.client.CopyBehavior.OVERWRITE)

            s = omni.client.stat(local_path)
            if s[0] != omni.client.Result.OK:
                Prompt(
                    title="File Path Remap",
                    text=f"Failed to copy {orig_path} to {local_path}",
                    ok_button_text="Ok",
                    modal=True
                ).show()
                return

            stage = omni.usd.get_context().get_stage()
            script_prims = [x for x in stage.Traverse() if x.HasAPI(OmniScriptingSchema.OmniScriptingAPI)]

            with omni.kit.undo.group():
                for prim in script_prims:
                    attr = prim.GetAttribute("omni:scripting:scripts").Get()
                    if any(a.resolvedPath == orig_path for a in attr):

                        def substitute(a):
                            if a.resolvedPath == orig_path:
                                return Sdf.AssetPath(local_path)
                            return a

                        omni.kit.commands.execute(
                            "ChangeProperty",
                            prop_path=prim.GetPath().AppendProperty("omni:scripting:scripts"),
                            value=[substitute(a) for a in attr],
                            prev=attr
                        )

            open_script_file(local_path)

        def on_get_file_name(filename, path, orig_path):

            global import_dialog
            import_dialog.hide()
            import_dialog = None

            local_path = os.path.join(path, filename)

            if os.path.exists(local_path):
                text = "Overwrite {} and remap all paths in scene?".format(local_path)
            else:
                text = "Copy script to {} and remap all paths in scene?".format(local_path)

            Prompt(
                title="File Path Remap",
                text=text,
                ok_button_text="Yes",
                cancel_button_text="No",
                ok_button_fn=lambda: on_copy_and_remap(local_path, orig_path),
                modal=True
            ).show()

        global import_dialog
        stage_dir = os.path.dirname(omni.usd.get_context().get_stage_url())

        def on_filter_item(item) -> bool:
            if not item or item.is_folder:
                return True
            return item.path.endswith(".py")

        import_dialog = omni.kit.window.filepicker.FilePickerDialog(
            "Save Remote Python Script As",
            apply_button_label="Save",
            current_directory=stage_dir if stage_dir.find("file:") == 0 else None,
            current_filename=os.path.basename(path),
            click_apply_handler=lambda f, p: on_get_file_name(f, p, path),
            item_filter_options=["Python Files (*.py)"],
            item_filter_fn=on_filter_item,
        )

        return

    try:
        omni_finder_loader.import_file(path)
    except:
        traceback_format_exception()

    real_path = omni_finder_loader.get_local_path(path)
    extension = omni.kit.scripting.Extension.get_instance()
    extension.get_script_editor().open_script_file(real_path)
