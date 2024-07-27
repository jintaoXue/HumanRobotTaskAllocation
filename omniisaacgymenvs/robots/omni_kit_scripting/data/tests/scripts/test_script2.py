import carb
import carb.events
from omni.kit.scripting import BehaviorScript


EVENT_TYPE_ON_INIT = carb.events.type_from_string("omni.kit.scripting.tests.EVENT_TYPE_ON_INIT")
EVENT_TYPE_ON_DESTROY = carb.events.type_from_string("omni.kit.scripting.tests.EVENT_TYPE_ON_DESTROY")
EVENT_TYPE_ON_PLAY = carb.events.type_from_string("omni.kit.scripting.tests.EVENT_TYPE_ON_PLAY")
EVENT_TYPE_ON_PAUSE = carb.events.type_from_string("omni.kit.scripting.tests.EVENT_TYPE_ON_PAUSE")
EVENT_TYPE_ON_STOP = carb.events.type_from_string("omni.kit.scripting.tests.EVENT_TYPE_ON_STOP")
EVENT_TYPE_ON_UPDATE = carb.events.type_from_string("omni.kit.scripting.tests.EVENT_TYPE_ON_UPDATE")


class TestScript2(BehaviorScript):
    def on_init(self):
        carb.log_info(f"{type(self).__name__}.on_init()->{self.prim_path}")
        self.message_bus_event_stream.push(EVENT_TYPE_ON_INIT, payload={"prim_path": self.prim_path})

    def on_destroy(self):
        carb.log_info(f"{type(self).__name__}.on_destroy()->{self.prim_path}")
        self.message_bus_event_stream.push(EVENT_TYPE_ON_DESTROY, payload={"prim_path": self.prim_path})

    def on_play(self):
        carb.log_info(f"{type(self).__name__}.on_play()->{self.prim_path}")
        self.message_bus_event_stream.push(EVENT_TYPE_ON_PLAY, payload={"prim_path": self.prim_path})

    def on_pause(self):
        carb.log_info(f"{type(self).__name__}.on_pause()->{self.prim_path}")
        self.message_bus_event_stream.push(EVENT_TYPE_ON_PAUSE, payload={"prim_path": self.prim_path})

    def on_stop(self):
        carb.log_info(f"{type(self).__name__}.on_stop()->{self.prim_path}")
        self.message_bus_event_stream.push(EVENT_TYPE_ON_STOP, payload={"prim_path": self.prim_path})

    def on_update(self, current_time: float, delta_time: float):
        carb.log_info(f"{type(self).__name__}.on_update({current_time}, {delta_time})->{self.prim_path}")
        self.message_bus_event_stream.push(EVENT_TYPE_ON_UPDATE, payload={"prim_path": self.prim_path})
