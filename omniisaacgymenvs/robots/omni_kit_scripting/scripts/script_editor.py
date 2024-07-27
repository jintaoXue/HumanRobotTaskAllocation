# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
from __future__ import annotations

import carb
import carb.settings
import omni.client
import omni.kit
import omni.kit.notification_manager
import omni.kit.window.popup_dialog

import asyncio
import os
import shutil
import sys
import tempfile
from typing import Optional


class ScriptEditor:
    def __init__(self):
        pass

    def open_script_file(self, path):
        """Opens a script file with a python script editor.
        Args:
            path: the url of the file.
        """
        if not os.path.exists(path):  # If the path is local, nothing to do
            return False

        settings = carb.settings.get_settings()
        # find out which editor it's necessary to use check the settings
        editor: Optional[str] = settings.get("/app/editor")
        if not editor:
            # if settings doesn't have it, check the environment variable EDITOR.
            # the standard way to allow the user to set the editor in linux.
            editor = os.environ.get("EDITOR", None)
            if not editor:
                # otherwise use vscode is the default editor
                editor = "code"

        # remove quotes because it's a common way for windows to specify paths
        if editor[0] == '"' and editor[-1] == '"':
            editor = editor[1:-1]

        if not self._is_exe(editor):
            try:
                # check if we can run the editor
                editor = shutil.which(editor)
            except shutil.Error:
                editor = None

        if not editor:
            if os.name == "nt":
                # all Windows have notepad
                editor = "notepad"
            else:
                # most Linux systems have gedit
                editor = "gedit"

        if os.name == "nt":
            # using cmd on the case the editor is bat or cmd file
            call_command = ["cmd", "/c"]
        else:
            call_command = []

        call_command.append(editor)

        if "code" in editor.lower():
            folder_path = os.path.dirname(path)
            call_command.append(folder_path)

            # create a .vscode folder with launch.json, extensions.json and settings.json
            vscode_folder_dst = os.path.join(folder_path, ".vscode")
            if not os.path.exists(vscode_folder_dst):
                ext_path = omni.kit.app.get_app().get_extension_manager().get_extension_path_by_module(__name__)
                vscode_folder_src = os.path.join(ext_path, "data/.vscode")
                shutil.copytree(vscode_folder_src, vscode_folder_dst)

            # force to use a new window.
            call_command.append("-n")
            # goto the file
            call_command.append("-g")
            call_command.append(path)
        else:
            call_command.append(path)

        asyncio.ensure_future(self._run_editor_async(call_command, path))
        return True

    def _is_exe(self, path):
        return os.path.isfile(path) and os.access(path, os.X_OK)

    async def _run_editor_async(self, cmd, path):
        try:
            # pass down the PYTHONPATH so that autocompletion and syntax highlighting works for kit-sdk + any extension loaded modules
            env = os.environ.copy()
            sys_paths_keys = omni.ext._impl.fast_importer._sys_paths.keys()
            pythonpath = os.path.dirname(path) + os.pathsep + os.pathsep.join(sys.path) + os.pathsep +  os.pathsep.join(sys_paths_keys)
            pythonpath = pythonpath + os.pathsep + env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = pythonpath
            process = await asyncio.create_subprocess_exec(*cmd, env=env)
            await process.wait()
        except asyncio.CancelledError:
            pass
        finally:
            pass

    def create_script_file(self, basepath: str, template_file_path: str):
        """Creates a new script file
        Args:
            basepath: folder where the file will be created
        """
        async def validate_python_file_creation(filename: str, abspath: str) -> bool:
            if len(filename) == 0:
                message = "Please provide a name for the script"
                await _warning(message)
                return False
            else:
                return True

        async def _warning(message: str) -> bool:
            try:
                await omni.kit.app.get_app().next_update_async()
                omni.kit.notification_manager.post_notification(
                    text=message,
                    status=omni.kit.notification_manager.NotificationStatus.WARNING,
                    hide_after_timeout=False
                )
            except:
                carb.log_warn(message)

        def _filename_to_classname(filename) -> str:
            filename = filename.replace("-", "_")
            x = filename.split("_")
            classname = []
            for i in x:
                i = i.title()
                classname.append(i)
            classname = "".join(classname)
            return classname

        def _create_python_file(self, location: str, template_file_path: str) -> None:
            async def _create_new_python_file(file_basename: str) -> None:
                abs_filepath = os.path.join(location, f"{file_basename}.py").replace("\\", "/")
                if not await validate_python_file_creation(file_basename, abs_filepath):
                    return

                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_python_file_path = os.path.join(temp_dir, "new_script.py")
                    try:
                        with open(temp_python_file_path, 'w') as f:
                            if template_file_path and len(template_file_path) > 0:
                                with open(template_file_path, 'r') as template_file:
                                    file_contents = template_file.read().replace("__klass", _filename_to_classname(file_basename))
                                    f.write(file_contents)

                    except OSError:
                        _warning(f"""Can't create a new python file at "{temp_python_file_path}" """)
                        return
                    result = await omni.client.copy_async(temp_python_file_path, abs_filepath)
                    if result != omni.client.Result.OK:
                        await _warning(
                            f"Unable to copy new script from \"{temp_python_file_path}\" to \"{abs_filepath}\" (Error code \"{result}\")."
                        )

            def on_ok(dialog: omni.kit.window.popup_dialog.InputDialog):
                asyncio.ensure_future(_create_new_python_file(dialog.get_value()))
                dialog.hide()

            dialog = omni.kit.window.popup_dialog.InputDialog(
                width=300,
                message="New Python Script",
                pre_label="Name: ",
                post_label=".py",
                default_value="new_script",
                ok_handler=lambda dialog: on_ok(dialog),
                ok_label="Create"
            )
            dialog.show()

        _create_python_file(self, basepath, template_file_path)
