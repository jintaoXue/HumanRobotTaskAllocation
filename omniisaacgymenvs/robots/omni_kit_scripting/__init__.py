# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
from .scripts import *
from .scripts import ScriptManager
import carb.events
import carb.settings
import carb.input
import omni.timeline
from omni.timeline._timeline import ITimeline
from omni.kit.app._app import IApp
from omni.appwindow._appwindow import IAppWindow
import omni.usd
from omni.usd._usd import Selection, UsdContext
from pxr import Sdf, Usd, UsdUtils


class BehaviorScript:
    """
    Base class for for developing per USD Prim behaviors.
    """
    def __init__(self, prim_path: Sdf.Path):
        """Initializes a BehaviorScript for the prim_path( Sdf.Path) this script was assigned to."""
        self._has_on_update_error = False
        self._prim_path = prim_path
        self._settings = carb.settings.get_settings()
        self._usd_context = omni.usd.get_context()
        self._selection = self._usd_context.get_selection()
        self._stage = UsdUtils.StageCache.Get().Find(Usd.StageCache.Id.FromLongInt(self._usd_context.get_stage_id()))
        self._prim = self._stage.GetPrimAtPath(self._prim_path)
        self._timeline = omni.timeline.get_timeline_interface()
        self._input = carb.input.acquire_input_interface()
        self._app = omni.kit.app.get_app()
        self._message_bus_event_stream = self._app.get_message_bus_event_stream()
        self.on_init()

    @property
    def prim_path(self) -> Sdf.Path:
        """Returns the prim path that this script is assigned to."""
        return self._prim_path

    @property
    def prim(self) -> Usd.Prim:
        """Returns the prim that this script is assigned to."""
        return self._prim

    @property
    def stage(self) -> Usd.Stage:
        """Returns the current USD stage that is opened/loaded."""
        return self._stage

    @property
    def usd_context(self) -> UsdContext:
        """Returns the current USD context."""
        return self._usd_context

    @property
    def selection(self) -> Selection:
        """Returns the current USD context selection interface in the application."""
        return self._selection

    @property
    def settings(self) -> carb.settings.ISettings:
        """Returns the current settings."""
        return self._settings

    @property
    def timeline(self) -> ITimeline:
        """Returns the application timeline interface."""
        return self._timeline

    @property
    def input(self) -> carb.input.IInput:
        """Returns the application input interface."""
        return self._input

    @property
    def default_app_window(self) -> IAppWindow:
        return omni.appwindow.get_default_app_window()

    @property
    def app(self) -> IApp:
        """Returns the kit application interface."""
        return self._app

    @property
    def message_bus_event_stream(self) -> carb.events.IEvents:
        """Returns the application message bus event stream."""
        return self._message_bus_event_stream

    def on_init(self):
        """Override this method to handle when the BehaviorScript is initialize from being assigned to a USD prim."""
        pass

    def on_destroy(self):
        """Override this method to handle when the BehaviorScript is destroyed from being unassigned from a USD prim."""
        pass

    def on_play(self):
        """Override this method to handle when `play` is pressed.
        """
        pass

    def on_pause(self):
        """Override this method to handle when `pause` is pressed.
        """
        pass

    def on_stop(self):
        """Override this method to handle when the `stop` is pressed.
        """
        pass

    def on_update(self, current_time: float, delta_time: float):
        """Override this method to handle per frame update events that occur when `playing`.
        Args:
            current_time: The current time. (in seconds).
            delta_time: The delta time (in seconds).
        """
        pass


EVENT_TYPE_BEHAVIOR_SCRIPT_LOADED = carb.events.type_from_string("omni.kit.scripting.EVENT_TYPE_BEHAVIOR_SCRIPT_LOADED")
EVENT_TYPE_BEHAVIOR_SCRIPT_UNLOADED = carb.events.type_from_string("omni.kit.scripting.EVENT_TYPE_BEHAVIOR_SCRIPT_UNLOADED")


def get_event_stream() -> carb.events.IEventStream:
    """
    Returns the scripting events stream to receive events when script are loaded and unloaded.

    EVENT_TYPE_BEHAVIOR_SCRIPT_LOADED
    Args:
        prim_path: The Sdf.Path the BehaviorScript is associated to:
        script_path: The script path.
        script_instance: The python BehaviorScript instance.


    EVENT_TYPE_BEHAVIOR_SCRIPT_UNLOADED
    Args:
        prim_path: The Sdf.Path the BehaviorScript is associated to:
        script_path: The script path.
    """
    return ScriptManager.get_instance().get_event_stream()