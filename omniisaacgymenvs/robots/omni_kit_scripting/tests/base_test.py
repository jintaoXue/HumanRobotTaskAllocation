# pragma: no cover

import carb
import carb.settings
import omni.kit.commands
import omni.kit.ui_test as ui_test
import omni.kit.undo
import omni.timeline
import omni.usd
import omni.ui as ui
from omni.kit.test_suite.helpers import wait_stage_loading
from omni.ui.tests.test_base import OmniUiTest
from pathlib import Path
import inspect
import os


class BaseTest(OmniUiTest):
    async def setUp(self):
        """ Runs before each test. Use omni.kit.window.inpsector for widget paths"""
        await super().setUp()
        self._context = omni.usd.get_context()
        self._selection = self._context.get_selection()
        self._settings = carb.settings.get_settings()
        self._settings_cache = {}
        self._timeline = omni.timeline.get_timeline_interface()
        self._ext_dir = Path(omni.kit.app.get_app().get_extension_manager().get_extension_path_by_module(__name__))
        self._scripts_data_dir = self._ext_dir.joinpath("data/tests/scripts")
        self._golden_data_dir = self._ext_dir.joinpath("data/tests/golden")
        self._usd_data_dir = self._ext_dir.joinpath("data/tests/usd")
        self._timeline.stop()
        self._timeline.set_current_time(0.0)
        self._timeline.set_auto_update(False)

    async def tearDown(self):
        """ Runs after each test."""
        self._golden_data_dir = None
        self._usd_data_dir = None
        self._timeline.set_auto_update(True)
        self._timeline.stop()
        await omni.kit.app.get_app().next_update_async()
        await super().tearDown()

    @property
    def ext_dir(self):
        return self._ext_dir

    @property
    def golden_data_dir(self):
        return self._golden_data_dir

    @property
    def usd_data_dir(self):
        return self._usd_data_dir

    @property
    def scripts_data_dir(self):
        return self._scripts_data_dir

    async def setup_window(self, width, height):
        app_window = omni.appwindow.get_default_app_window()
        dpi_scale = ui.Workspace.get_dpi_scale()
        width_with_dpi = int(width * dpi_scale)
        height_with_dpi = int(height * dpi_scale)
        current_width = app_window.get_width()
        current_height = app_window.get_height()
        if width_with_dpi == current_width and height_with_dpi == current_height:
            self._saved_width = None
            self._saved_height = None
        else:
            self._saved_width = current_width
            self._saved_height = current_height
            app_window.resize(width_with_dpi, height_with_dpi)
            await omni.appwindow.get_default_app_window().get_window_resize_event_stream().next_event()

        # move the cursor away to avoid hovering on element and trigger tooltips that break the tests
        windowing = carb.windowing.acquire_windowing_interface()
        os_window = app_window.get_window()
        windowing.set_cursor_position(os_window, (0, 0))

        self._restore_window = None
        self._restore_position = None
        self._restore_dock_window = None

    async def restore_window(self):
        if self._saved_width is not None and self._saved_height is not None:
            app_window = omni.appwindow.get_default_app_window()
            app_window.resize(self._saved_width, self._saved_height)
            await omni.appwindow.get_default_app_window().get_window_resize_event_stream().next_event()

        if self._restore_dock_window and self._restore_window:
            self._restore_dock_window.dock_in(self._restore_window, self._restore_position)
            self._restore_window = None
            self._restore_position = None
            self._restore_dock_window = None

    async def wait_for_stage_loading(self):
        while True:
            _, files_loaded, total_files = omni.usd.get_context().get_stage_loading_status()
            if files_loaded or total_files:
                await ui_test.human_delay()
                continue
            break

        await ui_test.human_delay()

    async def load_stage(self, base_url, stage_name):
        stage_name = os.path.join(base_url, stage_name)
        result = None
        (result, err) = await self._context.open_stage_async(stage_name, omni.usd.UsdContextInitialLoadSet.LOAD_ALL)
        await wait_stage_loading()
        self.assertTrue(result)
        await omni.kit.app.get_app().next_update_async()
        await omni.kit.app.get_app().next_update_async()
        return result

    async def setup_settings(self, settings_list):
        for name, value, _ in settings_list:
            self._settings_cache[name] = self._settings.get(name)
            self._settings.set(name, value)
        for _ in range(5):
            await omni.kit.app.get_app().next_update_async()

    def restore_settings(self, settings_list):
        if len(self._settings_cache) == 0:
            return

        def restore(settings_list_to_restore):
            for name, _, default in settings_list_to_restore:
                try:
                    cache = self._settings_cache[name]
                    self._settings.set(name, cache if cache is not None else default)
                except KeyError:
                    pass

        restore(settings_list)

    async def play(self):
        self._timeline.set_auto_update(False)
        self._timeline.play()
        await omni.kit.app.get_app().next_update_async()
        await omni.kit.app.get_app().next_update_async()

    async def step_frame(self, nun_frame: int):
        self._timeline.set_auto_update(False)
        for i in range(nun_frame):
            self._timeline.forward_one_frame()
            await omni.kit.app.get_app().next_update_async()
            await omni.kit.app.get_app().next_update_async()

    def set_time(self, time_in_secs: float):
        self._timeline.set_auto_update(False)
        self._timeline.set_current_time(time_in_secs)

    async def snapshot_compare(self, golden_img_name: str, stack_depth=1):
        await ui_test.human_delay(10)
        test_name = f"{self.__module__}.{self.__class__.__name__}.{inspect.stack()[stack_depth][3]}"
        await self.finalize_test(
            golden_img_dir=self.golden_data_dir,
            golden_img_name=f"{test_name}.{golden_img_name}.png",
            threshold=25,
        )
