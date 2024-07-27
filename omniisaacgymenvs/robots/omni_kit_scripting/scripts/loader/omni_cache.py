# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# pragma: no cover

from . import io

import carb.events
import omni.client as oc

import asyncio
from pathlib import Path
import shutil
import os


EVENT_TYPE_SCRIPT_FILE_UPDATED = carb.events.type_from_string("omni.kit.scripting.EVENT_TYPE_SCRIPT_FILE_UPDATED")
EVENT_TYPE_SCRIPT_FILE_DELETED = carb.events.type_from_string("omni.kit.scripting.EVENT_TYPE_SCRIPT_FILE_DELETED")


class OmniCache:
    _path = None
    _file_watches = {}
    _file_watches_info = {}
    _path_cache = {}
    _loop = None
    _event_stream = None

    @classmethod
    def initialize(cls) -> None:
        cls._path = Path.home() / ".nvidia-omniverse" / "pycache"
        try:
            if cls._path.exists():
                shutil.rmtree(str(cls._path))
            cls._path.mkdir(parents=True)
        except:
            carb.log_info("Python Scripting: skipping OmniCache cleanup.")

        if not cls._path.exists():
            carb.log_error("Python Scripting: Failed to access OmniCache directory.")

        cls._file_watches = {}
        cls._file_watches_info = {}
        cls._file_watches_remote = {}
        cls._file_watches_remote_ignored = set()
        cls._path_cache = {}
        cls._loop = asyncio.get_event_loop()
        cls._event_stream = carb.events.acquire_events_interface().create_event_stream()
        carb.log_info("OmniCache initialized.")

    @classmethod
    def shutdown(cls) -> None:
        cls._event_stream = None
        cls._loop = None
        cls._file_watches.clear()
        cls._file_watches_info.clear()
        cls._file_watches_remote.clear()
        cls._file_watches_remote_ignored.clear()
        cls._path_cache.clear()
        try:
            if cls._path.exists():
                shutil.rmtree(str(cls._path))
        except:
            carb.log_info("Python Scripting: skipping OmniCache cleanup.")
        cls._path = None
        carb.log_info("OmniCache shutdown.")

    @classmethod
    def get_event_stream(cls):
        return cls._event_stream

    @classmethod
    def _sanitize_name(cls, name: str) -> str:
        replace = [
            "#", "%", "&", "{", "}", "\\", "<", ">", "*", "?", "/", " ", "$", "!", "'", "\"", ":", "+", "`", "|", "="
        ]
        result = name
        for c in replace:
            result = result.replace(c, f"_{ord(c)}_")
        return result

    @classmethod
    def convert_path(cls, path: str) -> str:
        if not io.is_omni_path(path):
            return os.path.realpath(path)
        if path in cls._path_cache:
            return cls._path_cache[path]

        url = oc.break_url(str(path))

        result = cls._path / cls._sanitize_name(url.scheme)
        if url.user:
            result = result / cls._sanitize_name(url.user)
        if url.host:
            result = result / cls._sanitize_name(url.host)
        if url.port:
            result = result / cls._sanitize_name(url.port)
        if url.path:
            path_parts = url.path.rsplit("/")
            for p in path_parts:
                result = result / cls._sanitize_name(p)

        real_path = os.path.realpath(str(result))
        cls._path_cache[str(path)] = real_path
        return real_path

    @classmethod
    def source_path(cls, cache_path: str) -> str:
        for key, value in cls._path_cache.items():
            if value and value == cache_path:
                return key

        if cache_path.startswith(str(cls._path)):
            # The cache_path is a path that *should* have a source path but doesn't because we haven't seen it yet
            #   The only time this should happen is for a file/folder that's a child of a path we've already seen
            cache_path_obj = Path(cache_path)
            parent_path = str(cache_path_obj.parent)
            for key, value in cls._path_cache.items():
                if value and value == parent_path:
                    return io.join(key, str(cache_path_obj.name))

        return cache_path

    @classmethod
    def _on_omni_copy(cls, result: oc.Result, src_path, dst_path):
        # print(f"_on_omni_copy: src_path {src_path}, dst_path: {dst_path}")
        if result != oc.Result.OK:
            return
        if dst_path in cls._file_watches_remote_ignored:
            cls._file_watches_remote_ignored.remove(dst_path)
            return

        cls._event_stream.push(EVENT_TYPE_SCRIPT_FILE_UPDATED, payload={"script_path": dst_path})

        async def pump_later():
            cls._event_stream.pump()

        asyncio.run_coroutine_threadsafe(pump_later(), cls._loop)

    @classmethod
    def _on_omni_sub_local(cls, result: oc.Result, entry: oc.ListEntry, path: str) -> None:
        if result != oc.Result.OK:
            return
        # print(f"_on_omni_sub_local UPDATED: path:{path}, size:{entry.size}, modified_time: {entry.modified_time}")
        cls._file_watches_info[path] = (entry.size, entry.modified_time)

    @classmethod
    def _on_omni_event_local(cls, result: oc.Result, event: oc.ListEvent, entry: oc.ListEntry, src: str, dst: str) -> None:
        if result != oc.Result.OK:
            return
        if event == oc.ListEvent.UPDATED:
            if entry.size == 0:
                return
            if src in cls._file_watches_info:
                info = cls._file_watches_info[src]
                if entry.size == info[0] and entry.modified_time == info[1]:
                    return
            # print(f"_on_omni_event_local UPDATED: src:{src}, dst:{dst}, modified_time: {entry.modified_time}, entry {entry}")
            if dst:
                cls._file_watches_remote_ignored.add(dst)
                oc.copy_with_callback(
                    src,
                    dst,
                    lambda r, src_path=src, dst_path=dst: cls._on_omni_copy(r, src_path, dst_path),
                    oc.CopyBehavior.OVERWRITE
                )
            else:
                cls._event_stream.push(EVENT_TYPE_SCRIPT_FILE_UPDATED, payload={"script_path": src})

                async def pump_later():
                    cls._event_stream.pump()

                asyncio.run_coroutine_threadsafe(pump_later(), cls._loop)

            cls._file_watches_info[src] = (entry.size, entry.modified_time)

        elif event == oc.ListEvent.DELETED:
            # print(f"_on_omni_event_local DELETED: src:{src}, entry {entry}")
            if src in cls._file_watches:
                del cls._file_watches_info[src]
            if src in cls._file_watches_info:
                del cls._file_watches_info[src]

            cls._event_stream.push(EVENT_TYPE_SCRIPT_FILE_DELETED, payload={"script_path": src})

            async def pump_later():
                cls._event_stream.pump()

            asyncio.run_coroutine_threadsafe(pump_later(), cls._loop)

    @classmethod
    def _on_omni_event_remote(cls, result: oc.Result, event: oc.ListEvent, entry: oc.ListEntry, src: str, dst: str) -> None:
        if result != oc.Result.OK:
            return

        if event == oc.ListEvent.DELETED:
            # print(f"_on_omni_event_remote DELETED: src:{src}, dst:{dst}, entry {entry}")
            if dst in cls._file_watches_remote:
                if os.path.exists(src):
                    os.remove(src)
                del cls._path_cache[dst]
                del cls._file_watches_remote[dst]

            if src in cls._file_watches:
                del cls._file_watches[src]
            if src in cls._file_watches_info:
                del cls._file_watches_info[src]

            cls._event_stream.push(EVENT_TYPE_SCRIPT_FILE_DELETED, payload={"script_path": dst})

            async def pump_later():
                cls._event_stream.pump()

            asyncio.run_coroutine_threadsafe(pump_later(), cls._loop)

        elif event == oc.ListEvent.CREATED or event == oc.ListEvent.UPDATED:
            # print(f"_on_omni_event_remote CREATED: src:{src}, dst:{dst}, entry {entry}")
            if dst in cls._file_watches_remote_ignored:
                cls._file_watches_remote_ignored.remove(dst)
                return

            if src in cls._file_watches:
                del cls._file_watches[src]
            if src in cls._file_watches_info:
                del cls._file_watches_info[src]

            cls._event_stream.push(EVENT_TYPE_SCRIPT_FILE_UPDATED, payload={"script_path": dst})

            async def pump_later():
                cls._event_stream.pump()

            asyncio.run_coroutine_threadsafe(pump_later(), cls._loop)

            cls._file_watches_info[src] = (entry.size, entry.modified_time)

    @classmethod
    def add_script(cls, path: str) -> str:
        # setup stat subscriptions
        cached_path = cls.convert_path(path)

        if cached_path in cls._file_watches:
            return cached_path

        if io.is_omni_path(path):
            # copy remote file into local cache
            Path(cached_path).parent.mkdir(parents=True, exist_ok=True)
            oc.copy(path, cached_path, behavior=oc.CopyBehavior.OVERWRITE)
            # track local file changes
            stat_sub_local = oc.stat_subscribe_with_callback(
                cached_path,
                lambda r, en, src=cached_path: cls._on_omni_sub_local(r, en, src),
                lambda r, ev, en, src=cached_path, dst=path: cls._on_omni_event_local(r, ev, en, src, dst)
            )
            cls._file_watches[cached_path] = stat_sub_local
            # track remote file changes
            stat_sub_remote = oc.stat_subscribe_with_callback(
                path,
                None,
                lambda r, ev, en, src=cached_path, dst=path: cls._on_omni_event_remote(r, ev, en, src, dst)
            )
            cls._file_watches_remote[path] = stat_sub_remote

        else:
            stat_sub_local = oc.stat_subscribe_with_callback(
                cached_path,
                lambda r, en, src=cached_path: cls._on_omni_sub_local(r, en, src),
                lambda r, ev, en, src=cached_path, dst=None: cls._on_omni_event_local(r, ev, en, src, dst)
            )
            cls._file_watches[cached_path] = stat_sub_local

        return cached_path
