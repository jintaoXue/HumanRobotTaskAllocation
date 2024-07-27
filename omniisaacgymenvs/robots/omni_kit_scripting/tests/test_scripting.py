
# pragma: no cover

from .base_test import BaseTest

from pxr import OmniScriptingSchema
import carb.events
import omni.kit.app
import omni.kit.commands
import omni.kit.scripting
import omni.kit.undo
import omni.usd
import omni.timeline

import omni.kit.ui_test as ui_test

import asyncio
import os

TEST_PRIM_PATH = "/World/defaultLight"
SCRIPTS_ATTR = "omni:scripting:scripts"
TEST_SCRIPT = "test_script.py"
TEST_SCRIPT2 = "test_script2.py"

EVENT_TYPE_ON_INIT = carb.events.type_from_string("omni.kit.scripting.tests.EVENT_TYPE_ON_INIT")
EVENT_TYPE_ON_DESTROY = carb.events.type_from_string("omni.kit.scripting.tests.EVENT_TYPE_ON_DESTROY")
EVENT_TYPE_ON_UPDATE = carb.events.type_from_string("omni.kit.scripting.tests.EVENT_TYPE_ON_UPDATE")


class TestScripting(BaseTest):

    async def setUp(self):
        await super().setUp()
        self._future_test = None

    async def tearDown(self):
        await super().tearDown()

    async def click_menu_add_python_scripting(self, prim):
        self._selection.set_selected_prim_paths([prim.GetPath().pathString], True)
        await ui_test.find("Stage").right_click()
        await ui_test.human_delay()
        await ui_test.select_context_menu("Add/Python Scripting")
        await ui_test.human_delay()

    async def click_menu_add_python_scripting2(self, prim, prim2):
        self._selection.set_selected_prim_paths([prim.GetPath().pathString, prim2.GetPath().pathString], True)
        await ui_test.find("Stage").right_click()
        await ui_test.human_delay()
        await ui_test.select_context_menu("Add/Python Scripting")
        await ui_test.human_delay()

    async def click_menu_delete_prim(self, prim):
        self._selection.set_selected_prim_paths([prim.GetPath().pathString], True)
        await ui_test.find("Stage").right_click()
        await ui_test.human_delay()
        await ui_test.select_context_menu("Delete")
        await ui_test.human_delay()

    async def click_menu_duplicate_prim(self, prim):
        self._selection.set_selected_prim_paths([prim.GetPath().pathString], True)
        await ui_test.find("Stage").right_click()
        await ui_test.human_delay()
        await ui_test.select_context_menu("Duplicate")
        await ui_test.human_delay()

    def on_script_event(self, event: carb.events.IEvent):
        if self._future_test and int(self._required_event) == int(event.type) and not self._future_test.done():
            self.assertTrue(True)
            self._future_test.set_result(event.type)

    async def wait_for_script_event(self, required_event):
        self._required_event = required_event
        self._future_test = asyncio.Future()

        async def wait_for_event():
            await self._future_test

        try:
            await asyncio.wait_for(wait_for_event(), timeout=30.0)
        except asyncio.TimeoutError:
            self.assertTrue(False)
        self._future_test = None
        self._required_script_event = -1

    async def test_apply_api_and_remove_api_command(self):
        await self.load_stage(self.usd_data_dir, "NewStage.usda")
        await self.wait_for_stage_loading()
        self._stage = self._context.get_stage()
        prim = self._stage.GetPrimAtPath(TEST_PRIM_PATH)
        self.assertFalse(prim.HasAPI(OmniScriptingSchema.OmniScriptingAPI))
        omni.kit.commands.execute("ApplyScriptingAPICommand", paths=[prim.GetPath()])
        self.assertTrue(prim.HasAPI(OmniScriptingSchema.OmniScriptingAPI))
        omni.kit.undo.undo()
        self.assertFalse(prim.HasAPI(OmniScriptingSchema.OmniScriptingAPI))
        omni.kit.undo.redo()
        self.assertTrue(prim.HasAPI(OmniScriptingSchema.OmniScriptingAPI))
        omni.kit.commands.execute("RemoveScriptingAPICommand", paths=[prim.GetPath()])
        self.assertFalse(prim.HasAPI(OmniScriptingSchema.OmniScriptingAPI))
        omni.kit.undo.undo()
        self.assertTrue(prim.HasAPI(OmniScriptingSchema.OmniScriptingAPI))
        omni.kit.undo.redo()
        self.assertFalse(prim.HasAPI(OmniScriptingSchema.OmniScriptingAPI))

    async def test_click_apply_api_remove_api(self):
        await self.load_stage(self.usd_data_dir, "NewStage.usda")
        await self.wait_for_stage_loading()
        self._stage = self._context.get_stage()
        prim = self._stage.GetPrimAtPath(TEST_PRIM_PATH)
        self.assertFalse(prim.HasAPI(OmniScriptingSchema.OmniScriptingAPI))
        await self.click_menu_add_python_scripting(prim)
        self.assertTrue(prim.HasAPI(OmniScriptingSchema.OmniScriptingAPI))
        await ui_test.find("Property//Frame/**/Button[*].identifier=='RemovePythonScriptingButton'").click()
        await ui_test.human_delay()
        await ui_test.find("Remove Python Scripting?//Frame/**/Button[1]").click()
        await ui_test.human_delay()
        self.assertTrue(prim.HasAPI(OmniScriptingSchema.OmniScriptingAPI))
        await ui_test.find("Property//Frame/**/Button[*].identifier=='RemovePythonScriptingButton'").click()
        await ui_test.human_delay()
        await ui_test.find("Remove Python Scripting?//Frame/**/Button[0]").click()
        await ui_test.human_delay()
        self.assertFalse(prim.HasAPI(OmniScriptingSchema.OmniScriptingAPI))

    async def test_click_apply_api_multi_select(self):
        await self.load_stage(self.usd_data_dir, "NewStage.usda")
        await self.wait_for_stage_loading()
        self._stage = self._context.get_stage()
        omni.kit.commands.execute("CopyPrim", path_from=TEST_PRIM_PATH)
        prim = self._stage.GetPrimAtPath(TEST_PRIM_PATH)
        prim2 = self._stage.GetPrimAtPath(f"{TEST_PRIM_PATH}_01")
        self.assertFalse(prim.HasAPI(OmniScriptingSchema.OmniScriptingAPI))
        await self.click_menu_add_python_scripting2(prim, prim2)
        self.assertTrue(prim.HasAPI(OmniScriptingSchema.OmniScriptingAPI))
        self.assertTrue(prim2.HasAPI(OmniScriptingSchema.OmniScriptingAPI))

    async def test_set_scripts(self):
        await self.load_stage(self.usd_data_dir, "NewStage.usda")
        await self.wait_for_stage_loading()
        self._stage = self._context.get_stage()
        prim = self._stage.GetPrimAtPath(TEST_PRIM_PATH)
        await self.click_menu_add_python_scripting(prim)
        self.assertTrue(prim.HasAPI(OmniScriptingSchema.OmniScriptingAPI))
        event_stream = omni.kit.app.get_app().get_message_bus_event_stream()
        self._script_sub = event_stream.create_subscription_to_pop_by_type(EVENT_TYPE_ON_INIT, self.on_script_event)
        asyncio.ensure_future(self.wait_for_script_event(EVENT_TYPE_ON_INIT))
        script_path = os.path.join(self.scripts_data_dir, TEST_SCRIPT)
        prim.GetAttribute(SCRIPTS_ATTR).Set([script_path])
        self._script_sub = None

    async def test_set_scripts_and_remove_api(self):
        await self.load_stage(self.usd_data_dir, "NewStage.usda")
        await self.wait_for_stage_loading()
        self._stage = self._context.get_stage()
        prim = self._stage.GetPrimAtPath(TEST_PRIM_PATH)
        await self.click_menu_add_python_scripting(prim)
        self.assertTrue(prim.HasAPI(OmniScriptingSchema.OmniScriptingAPI))
        event_stream = omni.kit.app.get_app().get_message_bus_event_stream()
        self._script_sub = event_stream.create_subscription_to_pop_by_type(EVENT_TYPE_ON_DESTROY, self.on_script_event)
        script_path = os.path.join(self.scripts_data_dir, TEST_SCRIPT)
        prim.GetAttribute(SCRIPTS_ATTR).Set([script_path])
        asyncio.ensure_future(self.wait_for_script_event(EVENT_TYPE_ON_DESTROY))
        omni.kit.commands.execute("RemoveScriptingAPICommand", paths=[prim.GetPath()])
        self._script_sub = None

    async def test_set_scripts_and_play(self):
        await self.load_stage(self.usd_data_dir, "NewStage.usda")
        await self.wait_for_stage_loading()
        self._stage = self._context.get_stage()
        prim = self._stage.GetPrimAtPath(TEST_PRIM_PATH)
        await self.click_menu_add_python_scripting(prim)
        self.assertTrue(prim.HasAPI(OmniScriptingSchema.OmniScriptingAPI))
        event_stream = omni.kit.app.get_app().get_message_bus_event_stream()
        self._script_sub = event_stream.create_subscription_to_pop_by_type(EVENT_TYPE_ON_UPDATE, self.on_script_event)
        script_path = os.path.join(self.scripts_data_dir, TEST_SCRIPT)
        prim.GetAttribute(SCRIPTS_ATTR).Set([script_path])
        asyncio.ensure_future(self.wait_for_script_event(EVENT_TYPE_ON_UPDATE))
        await self.play()
        self._script_sub = None

    async def test_set_scripts_multiple(self):
        await self.load_stage(self.usd_data_dir, "NewStage.usda")
        await self.wait_for_stage_loading()
        self._stage = self._context.get_stage()
        prim = self._stage.GetPrimAtPath(TEST_PRIM_PATH)
        await self.click_menu_add_python_scripting(prim)
        self.assertTrue(prim.HasAPI(OmniScriptingSchema.OmniScriptingAPI))
        event_stream = omni.kit.app.get_app().get_message_bus_event_stream()
        self._script_sub = event_stream.create_subscription_to_pop_by_type(EVENT_TYPE_ON_INIT, self.on_script_event)
        script_path = os.path.join(self.scripts_data_dir, TEST_SCRIPT)
        script_path2 = os.path.join(self.scripts_data_dir, TEST_SCRIPT2)
        asyncio.ensure_future(self.wait_for_script_event(EVENT_TYPE_ON_INIT))
        prim.GetAttribute(SCRIPTS_ATTR).Set([script_path])
        asyncio.ensure_future(self.wait_for_script_event(EVENT_TYPE_ON_INIT))
        prim.GetAttribute(SCRIPTS_ATTR).Set([script_path, script_path2])
        self._script_sub = None

    async def test_set_scripts_clear(self):
        await self.load_stage(self.usd_data_dir, "NewStage.usda")
        await self.wait_for_stage_loading()
        self._stage = self._context.get_stage()
        prim = self._stage.GetPrimAtPath(TEST_PRIM_PATH)
        await self.click_menu_add_python_scripting(prim)
        self.assertTrue(prim.HasAPI(OmniScriptingSchema.OmniScriptingAPI))
        event_stream = omni.kit.app.get_app().get_message_bus_event_stream()
        self._script_sub = event_stream.create_subscription_to_pop_by_type(EVENT_TYPE_ON_DESTROY, self.on_script_event)
        script_path = os.path.join(self.scripts_data_dir, TEST_SCRIPT)
        prim.GetAttribute(SCRIPTS_ATTR).Set([script_path])
        asyncio.ensure_future(self.wait_for_script_event(EVENT_TYPE_ON_DESTROY))
        prim.GetAttribute(SCRIPTS_ATTR).Set([])
        self._script_sub = None

    async def test_delete_prim_command(self):
        await self.load_stage(self.usd_data_dir, "NewStage.usda")
        await self.wait_for_stage_loading()
        self._stage = self._context.get_stage()
        prim = self._stage.GetPrimAtPath(TEST_PRIM_PATH)
        await self.click_menu_add_python_scripting(prim)
        self.assertTrue(prim.HasAPI(OmniScriptingSchema.OmniScriptingAPI))
        script_path = os.path.join(self.scripts_data_dir, TEST_SCRIPT)
        prim.GetAttribute(SCRIPTS_ATTR).Set([script_path])
        event_stream = omni.kit.app.get_app().get_message_bus_event_stream()
        self._script_sub = event_stream.create_subscription_to_pop_by_type(EVENT_TYPE_ON_DESTROY, self.on_script_event)
        asyncio.ensure_future(self.wait_for_script_event(EVENT_TYPE_ON_DESTROY))
        omni.kit.commands.execute("DeletePrims", paths=[TEST_PRIM_PATH])
        self._script_sub = None

    async def test_delete_prim_ui(self):
        await self.load_stage(self.usd_data_dir, "NewStage.usda")
        await self.wait_for_stage_loading()
        self._stage = self._context.get_stage()
        prim = self._stage.GetPrimAtPath(TEST_PRIM_PATH)
        await self.click_menu_add_python_scripting(prim)
        self.assertTrue(prim.HasAPI(OmniScriptingSchema.OmniScriptingAPI))
        script_path = os.path.join(self.scripts_data_dir, TEST_SCRIPT)
        prim.GetAttribute(SCRIPTS_ATTR).Set([script_path])
        event_stream = omni.kit.app.get_app().get_message_bus_event_stream()
        self._script_sub = event_stream.create_subscription_to_pop_by_type(EVENT_TYPE_ON_DESTROY, self.on_script_event)
        asyncio.ensure_future(self.wait_for_script_event(EVENT_TYPE_ON_DESTROY))
        await self.click_menu_delete_prim(prim)
        self._script_sub = None

    async def test_duplicate_prim_command(self):
        await self.load_stage(self.usd_data_dir, "NewStage.usda")
        await self.wait_for_stage_loading()
        self._stage = self._context.get_stage()
        prim = self._stage.GetPrimAtPath(TEST_PRIM_PATH)
        await self.click_menu_add_python_scripting(prim)
        self.assertTrue(prim.HasAPI(OmniScriptingSchema.OmniScriptingAPI))
        script_path = os.path.join(self.scripts_data_dir, TEST_SCRIPT)
        prim.GetAttribute(SCRIPTS_ATTR).Set([script_path])
        event_stream = omni.kit.app.get_app().get_message_bus_event_stream()
        self._script_sub = event_stream.create_subscription_to_pop_by_type(EVENT_TYPE_ON_INIT, self.on_script_event)
        asyncio.ensure_future(self.wait_for_script_event(EVENT_TYPE_ON_INIT))
        omni.kit.commands.execute("CopyPrim", path_from=TEST_PRIM_PATH)
        self._script_sub = None

    async def test_duplicate_prim_ui(self):
        await self.load_stage(self.usd_data_dir, "NewStage.usda")
        await self.wait_for_stage_loading()
        self._stage = self._context.get_stage()
        prim = self._stage.GetPrimAtPath(TEST_PRIM_PATH)
        await self.click_menu_add_python_scripting(prim)
        self.assertTrue(prim.HasAPI(OmniScriptingSchema.OmniScriptingAPI))
        script_path = os.path.join(self.scripts_data_dir, TEST_SCRIPT)
        prim.GetAttribute(SCRIPTS_ATTR).Set([script_path])
        event_stream = omni.kit.app.get_app().get_message_bus_event_stream()
        self._script_sub = event_stream.create_subscription_to_pop_by_type(EVENT_TYPE_ON_INIT, self.on_script_event)
        asyncio.ensure_future(self.wait_for_script_event(EVENT_TYPE_ON_INIT))
        await self.click_menu_duplicate_prim(prim)
        self._script_sub = None

    async def test_reload_script(self):
        await self.load_stage(self.usd_data_dir, "NewStage.usda")
        await self.wait_for_stage_loading()
        self._stage = self._context.get_stage()
        prim = self._stage.GetPrimAtPath(TEST_PRIM_PATH)
        await self.click_menu_add_python_scripting(prim)
        self.assertTrue(prim.HasAPI(OmniScriptingSchema.OmniScriptingAPI))
        event_stream = omni.kit.app.get_app().get_message_bus_event_stream()
        self._script_sub = event_stream.create_subscription_to_pop_by_type(EVENT_TYPE_ON_INIT, self.on_script_event)
        asyncio.ensure_future(self.wait_for_script_event(EVENT_TYPE_ON_INIT))
        script_path = os.path.join(self.scripts_data_dir, TEST_SCRIPT)
        prim.GetAttribute(SCRIPTS_ATTR).Set([script_path])
        event_stream = omni.kit.app.get_app().get_message_bus_event_stream()
        self._script_sub = event_stream.create_subscription_to_pop_by_type(EVENT_TYPE_ON_DESTROY, self.on_script_event)
        asyncio.ensure_future(self.wait_for_script_event(EVENT_TYPE_ON_DESTROY))
        script_file_object = open(script_path, 'a')
        pos = script_file_object.tell()
        script_file_object.write(' ')
        script_file_object.truncate(pos)
        script_file_object.close()
        self.assertTrue(True)
        self._script_sub = None
