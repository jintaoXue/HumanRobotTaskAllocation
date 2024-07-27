# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
from __future__ import annotations

from .loader.omni_cache import OmniCache, EVENT_TYPE_SCRIPT_FILE_UPDATED, EVENT_TYPE_SCRIPT_FILE_DELETED
from .loader.omni_finder_loader import get_dependency_list, get_dependency_module_name, OmniFinder
from .loader.io import isabs
from .utils import show_security_popup, traceback_format_exception
from .loader import omni_finder_loader

import carb
import carb.events
import carb.settings
import omni.stageupdate
import omni.timeline
import omni.client
import omni.usd
import usdrt
from usdrt import Usd
from pxr import Sdf, Tf, Trace, Usd, UsdUtils

import asyncio
import importlib
import inspect
import sys
import os
from typing import Callable
from typing import Dict

SCRIPTS_ATTR = "omni:scripting:scripts"
SETTINGS_IGNORE_WARNING = "/app/scripting/ignoreWarningDialog"
STAGEUPDATE_ORDER = 200


class ScriptManager:
    __instance: ScriptManager = None

    def __init__(self):
        # scripting related to loader and cache
        self._pending_prim_paths_added = set()
        self._pending_prim_paths_removed = set()
        self._pending_prop_paths_changed = set()
        self._pending_sync_task = None
        self._scripting_api_prim_paths = []
        self._script_to_prims = dict()
        self._prim_to_scripts = dict()
        self._script_event_stream = carb.events.acquire_events_interface().create_event_stream()
        self._script_module_names: Dict[str, str] = {}
        self._usd_listener = None
        # settings
        self._settings = carb.settings.get_settings()
        self._settings.set_default_bool(SETTINGS_IGNORE_WARNING, False)
        # timeline events
        self._timeline = omni.timeline.get_timeline_interface()
        self._timeline_playing = False
        timeline_stream = self._timeline.get_timeline_event_stream()
        self._timeline_sub = timeline_stream.create_subscription_to_pop(self._on_timeline_event, order=1000000)
        # stage events
        stage_update = omni.stageupdate.get_stage_update_interface()
        self._stage_sub = stage_update.create_stage_update_node(
            "omni.kit.scripting.SciptManager",
            on_attach_fn=self._on_stage_attach,
            on_detach_fn=self._on_stage_detach,
            on_update_fn=self._on_stage_update,
        )
        stage_update_nodes = stage_update.get_stage_update_nodes()
        stage_update.set_stage_update_node_order(len(stage_update_nodes) - 1, STAGEUPDATE_ORDER)
        self._stage = None
        self._current_time = 0.0
        self._failed_behavior_scripts = set()
        ScriptManager.__instance = self

    def destroy(self):
        self._unload_all_scripts()
        self._on_stage_detach()
        self._timeline_sub = None
        self._timeline = None
        self._timeline_playing = False
        self._failed_behavior_scripts = None
        ScriptManager.__instance = None

    @classmethod
    def get_instance(cls) -> ScriptManager:
        return cls.__instance

    def get_event_stream(self) -> carb.events.IEventStream:
        return self._script_event_stream

    def _on_timeline_event(self, e: carb.events.IEvent):
        if e.type == int(omni.timeline.TimelineEventType.PLAY):
            self._timeline_playing = True
            self._foreach_script_instance(lambda inst: inst.on_play())
        elif e.type == int(omni.timeline.TimelineEventType.PAUSE):
            self._foreach_script_instance(lambda inst: inst.on_pause())
            self._timeline_playing = False
        elif e.type == int(omni.timeline.TimelineEventType.STOP):
            self._foreach_script_instance(lambda inst: inst.on_stop())
            self._timeline_playing = False
            self._current_time = 0.0

    def _on_stage_attach(self, stage_id, meters_per_unit):
        self._usd_context = omni.usd.get_context()
        self._stage = usdrt.Usd.Stage.Attach(stage_id)
        self._open_stage()
        stage = UsdUtils.StageCache.Get().Find(Usd.StageCache.Id.FromLongInt(stage_id))
        self._usd_listener = Tf.Notice.Register(Usd.Notice.ObjectsChanged, self._on_usd_objects_changed, stage)
        self._script_event_sub = OmniCache.get_event_stream().create_subscription_to_pop(self._on_script_event)

    def _on_stage_detach(self):
        if self._usd_listener:
            self._usd_listener.Revoke()
        self._usd_listener = None
        self._stage = None
        self._script_event_sub = None
        OmniFinder.reset()

    def _on_stage_update(self, current_time: float, delta_time: float):
        carb.profiler.begin(3, "omni.kit.scripting.ScriptManager._on_stage_update")
        if self._stage and self._timeline_playing:
            self._current_time += delta_time
            script_errors = set()
            for scripts in self._prim_to_scripts.values():
                for script_path, script_instance in scripts.items():
                    if script_instance:
                        try:
                            if script_path in script_errors:
                                script_instance._has_on_update_error = True
                            if not script_instance._has_on_update_error:
                                script_instance.on_update(self._current_time, delta_time)
                        except:
                            script_instance._has_on_update_error = True
                            script_errors.add(script_path)
                            traceback_format_exception()
        carb.profiler.end(3)

    def _on_script_event(self, e: carb.events.IEvent):
        carb.profiler.begin(4, "omni.kit.scripting.ScriptManager._on_script_event")
        if e.type == int(EVENT_TYPE_SCRIPT_FILE_UPDATED):
            updated_script_path = e.payload["script_path"].replace('\\', '/')
            carb.log_info(f"Script file updated: {updated_script_path}")
            if len(self._failed_behavior_scripts) > 0:
                scripts_to_try_to_reload = [i for i in self._failed_behavior_scripts]
                for i in scripts_to_try_to_reload:
                    reload_success = self._reload_script(i)  # this will restore the dependencies trees
                    if reload_success:
                        self._failed_behavior_scripts.remove(i)

            # if this script update is a behavior script assigned to a prim directly
            if updated_script_path in self._script_to_prims:
                # call on_destroy for previous instances and remove the reference
                prim_paths = self._script_to_prims[updated_script_path]
                for prim_path_str in prim_paths:
                    scripts = self._prim_to_scripts[prim_path_str]
                    for script_path in scripts:
                        if script_path == updated_script_path:
                            script_instance = scripts[script_path]
                            if script_instance:
                                self._destroy_script_instance(prim_path_str, script_path, script_instance)
                                scripts[script_path] = None

                # reload the script
                self._reload_script(updated_script_path)

                # attempt to create the script instance again which calls the on_init
                for prim_path_str in prim_paths:
                    scripts = self._prim_to_scripts[prim_path_str]
                    for script_path in scripts:
                        if script_path == updated_script_path:
                            scripts[script_path] = self._create_script_instance(prim_path_str, script_path)

            else:
                # the script must be a dependency. Therefore lets get only the BehaviorScript dependencies for reloading
                script_dependency_list = get_dependency_list(updated_script_path)
                print(f"Script dependency file updated: {updated_script_path}")
                reloaded_behavior_scripts = []
                depent_nonbehavior_scripts = []
                for dependency in reversed(script_dependency_list):
                    depend_script_path = dependency[0]
                    depend_module_name = dependency[1]
                    if depend_module_name in sys.modules:
                        module = sys.modules[depend_module_name]

                        for m in inspect.getmembers(module):
                            if inspect.isclass(m[1]) and issubclass(m[1], omni.kit.scripting.BehaviorScript) and m[1] != omni.kit.scripting.BehaviorScript:
                                # call on_destroy for previous instances and remove the reference
                                prim_paths = self._script_to_prims[depend_script_path]
                                for prim_path_str in prim_paths:
                                    scripts = self._prim_to_scripts[prim_path_str]
                                    for script_path in scripts:
                                        if script_path == depend_script_path:
                                            script_instance = scripts[script_path]
                                            if script_instance:
                                                self._destroy_script_instance(prim_path_str, script_path, script_instance)
                                                scripts[script_path] = None

                                if depend_script_path not in reloaded_behavior_scripts:
                                    reloaded_behavior_scripts.append(depend_script_path)
                            else:
                                if depend_script_path not in depent_nonbehavior_scripts:
                                    depent_nonbehavior_scripts.append(depend_script_path)

                updated_script_module_name = get_dependency_module_name(updated_script_path)
                sys.modules.pop(updated_script_module_name, None)
                if len(reloaded_behavior_scripts) > 0:
                    # this is not a file disconnected from behavior scripts
                    for i in depent_nonbehavior_scripts:
                        sys.modules.pop(get_dependency_module_name(i), None)
                importlib.invalidate_caches()

                # reload the behavior script
                for behavior_script_path in reloaded_behavior_scripts:
                    self._reload_script(behavior_script_path)

                # re-create script instances for each dependency
                for dependendency in script_dependency_list:
                    depend_script_path = dependendency[0]
                    depend_module_name = dependendency[1]
                    module = sys.modules[depend_module_name]
                    for m in inspect.getmembers(module):
                        if inspect.isclass(m[1]) and issubclass(m[1], omni.kit.scripting.BehaviorScript) and m[1] != omni.kit.scripting.BehaviorScript:
                            # call on_destroy for previous instances and remove the reference
                            prim_paths = self._script_to_prims[depend_script_path]
                            # attempt to create the script instance again which calls the on_init
                            for prim_path_str in prim_paths:
                                scripts = self._prim_to_scripts[prim_path_str]
                                for script_path in scripts:
                                    if script_path == depend_script_path:
                                        try:
                                            # this also calls the `on_init` method in the BehaviorScript contructor/__init__ method
                                            script_instance = m[1](Sdf.Path(prim_path_str))
                                            self._prim_to_scripts[prim_path_str][script_path] = script_instance
                                            self._script_event_stream.push(
                                                omni.kit.scripting.EVENT_TYPE_BEHAVIOR_SCRIPT_LOADED,
                                                payload={
                                                    "prim_path": Sdf.Path(prim_path_str),
                                                    "script_path": script_path,
                                                    "script_instance": script_instance
                                                }
                                            )
                                            self._script_event_stream.pump()
                                        except:
                                            traceback_format_exception()
                                            carb.profiler.end(4)
                                            return

        elif e.type == int(EVENT_TYPE_SCRIPT_FILE_DELETED):
            deleted_script_path = e.payload["script_path"].replace('\\', '/')
            carb.log_info(f"Script file deleted: {deleted_script_path}")
            if deleted_script_path in self._script_to_prims:
                prim_paths = self._script_to_prims[deleted_script_path]
                for prim_path_str in prim_paths:
                    scripts = self._prim_to_scripts[prim_path_str]
                    for script_path in scripts:
                        if script_path == deleted_script_path:
                            script_instance = scripts[script_path]
                            if script_instance:
                                self._destroy_script_instance(prim_path_str, script_path, script_instance)
                                scripts[script_path] = None
        carb.profiler.end(4)

    async def _pending_sync_task_handler(self):
        carb.profiler.begin(5, "omni.kit.scripting.ScriptManager._pending_sync_task_handler")

        await omni.kit.app.get_app().next_update_async()
        await omni.kit.app.get_app().next_update_async()
        self._pending_sync_task = None

        # get the laests scripting api prim paths
        if self._stage:
            scripting_api_prim_paths = self._stage.GetPrimsWithAppliedAPIName("OmniScriptingAPI")

            # destroy scripts where prims are no longer in stage.
            compare_set = set(scripting_api_prim_paths)
            for prim_path in self._scripting_api_prim_paths:
                if prim_path not in compare_set:
                    if prim_path in self._prim_to_scripts:
                        scripts = self._prim_to_scripts[prim_path].copy()
                        for script in scripts:
                            self._destroy_script(prim_path, script)

            # apply script for new prim paths found
            compare_set = set(self._scripting_api_prim_paths)
            for prim_path in scripting_api_prim_paths:
                if prim_path not in compare_set:
                    prim = self._stage.GetPrimAtPath(prim_path)
                    self._apply_scripts(prim)

        carb.profiler.end(5)

    def _on_usd_objects_changed(self, objects_changed, stage):
        carb.profiler.begin(2, "omni.kit.scripting.ScriptManager._on_usd_objects_changed")
        if not stage:
            carb.profiler.end(2)
            return
        dirty_prims_added = set()
        dirty_prims_removed = set()
        dirty_props_changed = set()

        for resync_path in objects_changed.GetResyncedPaths():
            if resync_path.IsPrimPath():
                prim = stage.GetPrimAtPath(resync_path)
                if prim and prim.IsValid() and prim.IsActive():
                    dirty_prims_added.add(resync_path)
                else:
                    dirty_prims_removed.add(resync_path)
            elif resync_path.IsPropertyPath():
                if resync_path.name == SCRIPTS_ATTR:
                    dirty_props_changed.add(resync_path)
        for changed_info_only_path in objects_changed.GetChangedInfoOnlyPaths():
            if changed_info_only_path.IsPropertyPath():
                if changed_info_only_path:
                    path_name = changed_info_only_path.name
                    if path_name == SCRIPTS_ATTR:
                        dirty_props_changed.add(changed_info_only_path)

        self._pending_prim_paths_added.update(dirty_prims_added)
        self._pending_prim_paths_removed.update(dirty_prims_removed)
        self._pending_prop_paths_changed.update(dirty_props_changed)
        if (len(self._pending_prim_paths_added) > 0 or len(self._pending_prim_paths_removed) > 0 or len(self._pending_prop_paths_changed) > 0) and self._pending_sync_task is None:
            self._pending_sync_task = asyncio.ensure_future(self._pending_sync_task_handler())
        carb.profiler.end(2)

    def _foreach_script_instance(self, fn: Callable):
        for scripts in self._prim_to_scripts.values():
            for key, script_instance in scripts.items():
                if script_instance:
                    fn(script_instance)

    def _create_script_instance(self, prim_path_str, script_path):
        script_instance = None
        module_name = self._script_module_names.get(script_path, None)
        if module_name:
            module = sys.modules[module_name]
            for m in inspect.getmembers(module):
                # find the first subclass of `omni.kit.scripting.BehaviorScript`
                if inspect.isclass(m[1]) and issubclass(m[1], omni.kit.scripting.BehaviorScript) and m[1] != omni.kit.scripting.BehaviorScript:
                    try:
                        # this also calls the `on_init` method in the BehaviorScript contructor/__init__ method
                        script_instance = m[1](Sdf.Path(prim_path_str))
                        self._prim_to_scripts[prim_path_str][script_path] = script_instance
                        self._script_event_stream.push(
                            omni.kit.scripting.EVENT_TYPE_BEHAVIOR_SCRIPT_LOADED,
                            payload={
                                "prim_path": Sdf.Path(prim_path_str),
                                "script_path": script_path,
                                "script_instance": script_instance
                            }
                        )
                        self._script_event_stream.pump()
                    except:
                        traceback_format_exception()
                        return None

        return script_instance

    def _destroy_script_instance(self, prim_path_str, script_path, script_instance):
        try:
            script_instance.on_destroy()
            self._script_event_stream.push(
                omni.kit.scripting.EVENT_TYPE_BEHAVIOR_SCRIPT_UNLOADED,
                payload={
                    "prim_path": Sdf.Path(prim_path_str),
                    "script_path": script_path
                }
            )
            self._script_event_stream.pump()
        except:
            traceback_format_exception()

    def _destroy_script(self, prim_path_str, script_path):
        # destroy script instance
        if prim_path_str in self._prim_to_scripts:
            scripts = self._prim_to_scripts[prim_path_str]
            if script_path in scripts:
                script_instance = scripts[script_path]
                if script_instance:
                    self._destroy_script_instance(prim_path_str, script_path, script_instance)
                    scripts[script_path] = None

                self._prim_to_scripts[prim_path_str].pop(script_path)
                if len(self._prim_to_scripts[prim_path_str]) == 0:
                    self._prim_to_scripts.pop(prim_path_str)

        # remove this prim_path as a reference for the script
        if script_path in self._script_to_prims:
            self._script_to_prims[script_path].remove(prim_path_str)
            # if no longer in use, then unload the script
            if len(self._script_to_prims[script_path]) == 0:
                self._unload_script(script_path)
                self._script_to_prims.pop(script_path)

    def _apply_scripts(self, prim: Usd.Prim):
        prim_path_str = str(prim.GetPath())
        script_assets = prim.GetAttribute(SCRIPTS_ATTR).Get()

        # get the previous loaded scripts for this prim path
        prev_scripts = None
        if prim_path_str in self._prim_to_scripts and len(self._prim_to_scripts[prim_path_str]) > 0:
            prev_scripts = self._prim_to_scripts[prim_path_str].copy()
        else:
            self._prim_to_scripts[prim_path_str] = {}

        # go through each exising loaded scripts and load+create as needed and mark for removal
        if script_assets is not None:
            for script_asset in script_assets:
                script_full_path = ''
                if len(script_asset.resolvedPath) == 0:
                    # resolved path is not generated. it could be either because 1) the path is invalid, 2) the usd
                    # couldn't automatically resolve the path if the stage is on an omni server yet the script is local
                    if isabs(script_asset.path) and os.path.exists(script_asset.path):
                        script_full_path = script_asset.path.replace('\\', '/')
                    else:
                        script_full_path = ''
                else:
                    script_full_path = script_asset.resolvedPath.replace('\\', '/')
                if len(script_full_path) > 0:
                    # process script to be loaded
                    script_loaded_success = True
                    if script_full_path not in self._script_to_prims:
                        self._script_to_prims[script_full_path] = set()
                        script_loaded_success = self._load_script(script_full_path)

                    self._script_to_prims[script_full_path].add(prim_path_str)
                    already_loaded = prev_scripts is not None and script_full_path in prev_scripts

                    if not already_loaded:
                        if script_loaded_success:
                            # create the script instance
                            self._create_script_instance(prim_path_str, script_full_path)
                        else:
                            self._prim_to_scripts[prim_path_str][script_full_path] = None

                    # remove scripts that are still used.
                    if prev_scripts and script_full_path in prev_scripts:
                        prev_scripts.pop(script_full_path)

        # destroy unused scripts that were previously loaded
        if prev_scripts:
            for prev_script in prev_scripts:
                self._destroy_script(prim_path_str, prev_script)

    async def _security_check_and_wait(self, future, proceed_fn):
        show_security_popup(future)
        await future
        if future.result():
            proceed_fn()

    def _open_stage(self):
        carb.profiler.begin(1, "omni.kit.scripting.ScriptManager._open_stage")
        self._unload_all_scripts()
        # search for the first python scripting component with scripts assigned
        has_scripts = False
        self._scripting_api_prim_paths = self._stage.GetPrimsWithAppliedAPIName("OmniScriptingAPI")
        for prim_path in self._scripting_api_prim_paths:
            prim = self._stage.GetPrimAtPath(prim_path)
            if prim.HasAttribute(SCRIPTS_ATTR):
                scripts = prim.GetAttribute(SCRIPTS_ATTR).Get()
                if scripts and len(scripts) > 0:
                    has_scripts = True
                    break
        if has_scripts:
            ignoreWarning = self._settings.get_as_bool(SETTINGS_IGNORE_WARNING)
            if not ignoreWarning:
                self._future = asyncio.Future()
                asyncio.ensure_future(self._security_check_and_wait(self._future, self._load_all_scripts))
            else:
                self._load_all_scripts()
        carb.profiler.end(1)

    def _load_all_scripts(self):
        for prim_path in self._scripting_api_prim_paths:
            prim = self._stage.GetPrimAtPath(prim_path)
            self._apply_scripts(prim)

    def _unload_all_scripts(self):
        self._pending_dirty_task = None
        self._pending_prim_paths_added.clear()
        self._pending_prim_paths_removed.clear()
        self._pending_prop_paths_changed.clear()
        for prim_path_str in self._prim_to_scripts:
            scripts = self._prim_to_scripts[prim_path_str]
            for script_path in scripts:
                script_instance = scripts[script_path]
                if script_instance:
                    self._destroy_script_instance(prim_path_str, script_path, script_instance)
                    scripts[script_path] = None
        self._prim_to_scripts.clear()
        self._script_to_prims.clear()
        self._scripting_api_prim_paths.clear()
        unload_list = [loc for loc in self._script_module_names]
        for loc in unload_list:
            self._unload_script(loc)

    def _load_script(self, script_path: str) -> bool:
        if script_path in self._script_module_names:
            return self._reload_script(script_path)

        parse = omni.client.break_url(script_path)
        path = parse.path
        folder, filename = os.path.split(path)
        try:
            module_name = omni_finder_loader.import_file(script_path)
            if module_name is None:
                return False
            self._script_module_names[script_path] = module_name
        except:
            self._failed_behavior_scripts.add(script_path)
            traceback_format_exception()
            return False

        return True

    def _unload_script(self, script_path: str):
        if script_path in self._script_module_names:
            module_name = self._script_module_names[script_path]
            self._script_module_names.pop(script_path, None)
            sys.modules.pop(module_name, None)

    def _reload_script(self, script_path: str) -> bool:
        self._unload_script(script_path)
        importlib.invalidate_caches()
        return self._load_script(script_path)
