# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# pragma: no cover
from .omni_cache import OmniCache
from . import io
# import carb
from importlib.machinery import ModuleSpec, SourceFileLoader
from importlib.abc import MetaPathFinder
import importlib.util
import re
import sys
from typing import List, Tuple
# import carb
# import types
import importlib.machinery
import copy
import ast
import os


class OmniLoader(SourceFileLoader):
    """
    Implements a source loader for files that can come from any data source implemented by data.io.
    Fully supports the Python import system, including namespaces.

    The primary use of this class is to import modules from Omniverse or the local file system
    """
    def __init__(self, fullname, path):
        super().__init__(fullname, path)
        self._fullname = fullname

    def get_data(self, path):
        real_path = OmniCache.convert_path(path)
        if path.endswith(".py"):
            OmniCache.add_script(path)
        return io.read(real_path)

    # returns the time stamp of a file (int)
    def path_mtime(self, path):
        real_path = OmniCache.add_script(path)
        return io.get_mtime(real_path)

    def path_stats(self, path):
        real_path = OmniCache.add_script(path)
        # The Python source says that size is optional, but it's not.
        return {"mtime": self.path_mtime(real_path), "size": io.get_size(real_path)}

    # Writes compiled bytecode for cache, Optional
    def set_data(self, path, data, *, _mode=0o666):
        raise NotImplementedError()

    def create_module(self, spec):
        return None


class OmniFinder(MetaPathFinder):
    _module_paths = {}
    _dependencies = {}
    _dependency_module_names = {}
    _collect_dependencies = None

    @classmethod
    def reset(cls):
        for local_path, path, module_name in cls._module_paths.values():
            sys.modules.pop(module_name, None)
        cls._module_paths.clear()
        for _, sys_module_name in cls._dependency_module_names.items():
            sys.modules.pop(sys_module_name, None)
        cls._dependencies.clear()
        cls._dependency_module_names.clear()
        cls._collect_dependencies = None

    @classmethod
    def add_folder(cls, path, module_name):
        cls._module_paths[module_name] = (OmniCache.convert_path(path), path, module_name)

    @classmethod
    def remove_folder(cls, module_name):
        del cls._module_paths[module_name]

    @classmethod
    def spec_from_module(cls, fullname, path):
        spec = ModuleSpec(fullname, OmniLoader(fullname, path), origin=path)
        spec.has_location = True
        return spec

    @classmethod
    def spec_from_package(cls, fullname, path):
        spec = ModuleSpec(fullname, OmniLoader(fullname, path), origin=path)
        spec.has_location = True
        return spec

    @classmethod
    def spec_from_namespace(cls, fullname, path, name):
        spec = ModuleSpec(fullname, None, origin=path)
        spec.submodule_search_locations = []
        if name:
            spec.submodule_search_locations.append(io.join(path, name))
        else:
            spec.submodule_search_locations.append(path)
        spec.has_location = False
        return spec

    @classmethod
    def add_fix_missing_dependency_path(cls):

        # make sure all namespace dependencies are inside cls._dependencies
        for i_dep in copy.copy(cls._dependencies):
            if io.is_file(i_dep):
                if io.containing_folder(i_dep) not in cls._dependencies:
                    cls._dependencies.setdefault(io.containing_folder(i_dep), [])

        # remove dependencies not present in dependency_module_names
        for i_dep in copy.copy(cls._dependencies):
            if i_dep not in cls._dependency_module_names:
                cls._dependencies.pop(i_dep)

        # make sure all the paths are using the / instead of \\
        for module_name in copy.copy(cls._dependency_module_names):
            formal_module_name = module_name.replace('\\', '/')
            if formal_module_name == module_name:
                continue  # it is formal name already
            else:
                cls._dependency_module_names[formal_module_name] = \
                    cls._dependency_module_names.pop(module_name)
                if module_name in cls._dependencies:
                    cls._dependencies.pop(module_name)
                    cls._dependencies.setdefault(formal_module_name, [])

    @classmethod
    def find_spec(cls, fullname, path, target=None):
        """
        Try to find a spec for the specified module.

        Returns the matching spec, or None if not found.
        """
        for n in cls._module_paths.keys():
            if fullname.startswith(n):
                break
        else:  # no break
            return None

        if not path:
            return None
        # print(f"find_spec({fullname}, {path})")
        spec = None
        name = fullname.rsplit(".", 1)[-1]
        filename = name + ".py"
        dependency_filename = None
        # print(f"    filename = {filename}")
        for search_path in path:
            if not isinstance(search_path, str):
                continue
            # print(f"    search_path = {search_path}")
            real_search_path = OmniCache.source_path(search_path)
            if io.is_file(real_search_path):
                p = io.containing_folder(real_search_path)
            else:
                p = real_search_path
            # print(f"    real_search_path = {real_search_path}")
            full_file_path = io.join(p, filename)
            full_directory_path = io.join(p, name)
            dir_list = io.list_files(p)
            if filename in dir_list and io.is_file(full_file_path):
                # print(f"    file full_file_path = {full_file_path}")
                OmniCache.add_script(full_file_path)
                spec = cls.spec_from_module(fullname, OmniCache.convert_path(full_file_path))
                dependency_filename = full_file_path
                cls._dependency_module_names[dependency_filename] = spec.name
                break
            elif name in dir_list and io.is_folder(full_directory_path):
                # sub directory
                if io.is_file(io.join(full_directory_path, "__init__.py")):
                    full_file_path = io.join(full_directory_path, "__init__.py")
                    # print(f"    spec_from_package full_file_path = {full_file_path}")
                    OmniCache.add_script(full_file_path)
                    spec = cls.spec_from_package(fullname, OmniCache.convert_path(full_file_path))
                    # TODO: check this part. full_directory_path or full_file_path???
                    # dependency_filename = full_directory_path ???
                    dependency_filename = full_file_path
                    cls._dependency_module_names[dependency_filename] = spec.name
                    break
                else:
                    full_file_path = path
                    # print(f"    spec_from_namespace full_file_path = {full_file_path}")
                    spec = cls.spec_from_namespace(fullname, OmniCache.convert_path(search_path), name)
                    # add \\ or / to end of the directory path
                    dependency_filename = os.path.join(full_directory_path, '').replace('\\', '/')
                    cls._dependency_module_names[dependency_filename] = spec.name
                    break
            elif name == fullname:
                full_file_path = search_path
                # print(f"    spec_from_namespace full_file_path = {full_file_path}")
                spec = cls.spec_from_namespace(fullname, OmniCache.convert_path(search_path), None)
                dependency_filename = full_file_path
                cls._dependency_module_names[dependency_filename] = spec.name
                break
        if dependency_filename:
            cls._dependencies.setdefault(dependency_filename, [])

        return spec

    @classmethod
    def file_was_loaded_by_finder(cls, filename: str):
        for local_path, path, module_name in cls._module_paths.values():
            if filename.startswith(local_path):
                return True
        return False


_email_regex = re.compile(r'.*(([A-Za-z0-9]+[.-_])*[A-Za-z0-9]+@[A-Za-z0-9-]+(\.[A-Z|a-z]{2,})+).*')


def _make_import_friendly(path):
    # Check for email addresses that may appear in a URI
    match = re.fullmatch(_email_regex, path)
    if match:
        addr = match.groups()[0]
        addr_new = addr.replace('@', '__AT__').replace('.', '__DOT__')
        path.replace(addr, addr_new)

    # Replace all other naughty characters
    return path.replace(
        '://', '__ov__').replace(
        ':\\', ':/').replace(
        ':/', '__path__').replace(
        '/', '_').replace(
        '\\', '_').replace(
        '-', '__dash__').replace(
        '@', '__at__').replace(
        '.', '__dot__')


def _import_module(path, module_name, is_dir=False):
    spec = OmniFinder.find_spec(module_name, [path])

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    OmniFinder._dependencies[path] = []
    OmniFinder._collect_dependencies = (path, module_name)
    try:
        spec.loader.exec_module(module)
    except:  # NOQA
        raise
    finally:
        OmniFinder._collect_dependencies = None

    OmniFinder.add_fix_missing_dependency_path()  # add namespace deps
    _resolve_dependencies([i for i in OmniFinder._dependencies])


def _find_imported_dependencies(file: str):
    modules = []
    """ info: {"file": the file path,
               "func_import": the last name is assumed to be a function,
               "lineno": where the import code is,
               "import level": 2 for 'from .. import', 3 for 'from ... import',
               "string": how the original import code looks like}
    """
    info = []

    def visit_Import(node):
        for name in node.names:
            # this is not needed though we still add them
            modules.append(name.name)
            info.append({'file': file, 'func_import': False,
                         'lineno': node.lineno, 'str': f"import {name.name}",
                         'import_level': -1})  # import has no import level

    def visit_ImportFrom(node):

        if node.module is not None:
            # case 1: from .b import test_b      # test_b()
            # case 2: from .f.g.h import test_h  # test_h()
            # case 3: from .f.g import h         # h.test_h()

            # case 5: from ..i import test_i     # test_i()
            # case 4: from ..j import k          # k.test_k()
            for i_node in node.names:
                # assume node.names are py files
                modules.append(node.module + '.' + i_node.name)
                info.append(
                    {'file': file, 'func_import': False, 'lineno': node.lineno,
                     'str': f"from {node.module} import {i_node.name}",
                     'import_level': node.level}
                )
            modules.append(node.module)  # assume node.names are functions
            info.append(
                {'file': file, 'func_import': True, 'lineno': node.lineno,
                 'str': f"from {node.module} import *",
                 'import_level': node.level}
            )
        else:
            # case 1: from . import b            # b.test_b()
            # case 2: from . import b, c         # b.test_b(), c.test_c()

            # case 3: from .. import j, l         # j.test_j(), l.test_l()
            for i_node in node.names:
                modules.append(i_node.name)
                info.append(
                    {'file': file, 'func_import': False, 'lineno': node.lineno,
                     'str': f"from . import {i_node.name}",
                     'import_level': node.level}
                )

    node_iter = ast.NodeVisitor()
    node_iter.visit_Import = visit_Import
    node_iter.visit_ImportFrom = visit_ImportFrom

    with open(file) as f:
        node_iter.visit(ast.parse(f.read()))

    # process the modules into path like string
    potential_modules = []
    for module_id, i_module in enumerate(modules):
        path = os.path.join(*(i_module.split('.')))
        if info[module_id]['import_level'] >= 2:
            parent_path = ['..'] * (int(info[module_id]['import_level']) - 1)
            path = os.path.join(*parent_path, path)
        potential_modules.append(path)

    return info, potential_modules


def argsort(seq):
    return sorted(range(len(seq)), key=seq.__getitem__)


def _remove_redundant_path_seperator(path: str):
    """ remove path seperators. Assume the path ends with .py file
        convert 'omniverse://ov-content/Users/tingwuw@nvidia.com/fix_and_play/../../basics.py'
        into 'omniverse://ov-content/Users/tingwuw@nvidia.com/basics.py'
    """

    path = path.replace('\\', '/')  # all using / instead of \\
    while "/../" in path:
        # there is one that needs to be removed
        dotdot_pos = path.find("/../")
        left = path[:dotdot_pos]
        right = path[dotdot_pos + 3:]  # len("/../") - 1 = 3
        left = left[:left.rfind('/')]  # remove one dir for /../
        path = left + right
    return path

def _resolve_dependencies(modules_path: str):
    """ For every modules from files, register and check for
        dependencies between them
    """
    # step 1: get all the names and references
    module_names_in_sys_module, module_py_files, module_dirs, is_namespace = \
        {}, {}, {}, {}
    for i_module_path in modules_path:
        if io.is_file(i_module_path):
            # it's an actual module
            module_py_files[i_module_path] = \
                io.filename_from_path(i_module_path)
            module_dirs[i_module_path] = io.containing_folder(i_module_path)
            is_namespace[i_module_path] = False
            module_names_in_sys_module[i_module_path] = \
                OmniFinder._dependency_module_names[i_module_path]

        else:
            # it's a namespace module
            module_py_files[i_module_path] = i_module_path
            module_dirs[i_module_path] = i_module_path
            is_namespace[i_module_path] = True
            module_names_in_sys_module[i_module_path] = \
                OmniFinder._dependency_module_names[i_module_path]

    # step 2: find all the direct dependencies for each module
    dependencies = {i_module_path: [] for i_module_path in modules_path}
    reverse_dependencies = {i_module_path: [] for i_module_path in modules_path}
    for i_module_path in modules_path:

        if module_names_in_sys_module[i_module_path] not in sys.modules:
            # print(f"Warning: {i_module_path} not found!!")
            continue
        if is_namespace[i_module_path]:
            continue  # a namcespace module or not found. skip this module

        # get the dependencies information
        i_dep_info, i_module_dependencies = \
            _find_imported_dependencies(OmniCache.convert_path(i_module_path))
        for i_dep_id, i_dependencies in enumerate(i_module_dependencies):
            i_expected_fullpath = io.join(module_dirs[i_module_path],
                                          i_dependencies.replace('\\', '/') + '.py')
            i_expected_fullpath = _remove_redundant_path_seperator(i_expected_fullpath)

            if i_expected_fullpath in modules_path:
                if i_expected_fullpath not in dependencies[i_module_path]:
                    # if not in already, add this
                    dependencies[i_module_path].append(i_expected_fullpath)
                    reverse_dependencies[i_expected_fullpath].append(i_module_path)

        # get the namespace dependency if needed
        if not is_namespace[i_module_path]:
            namespace_module_path = module_dirs[i_module_path]
            for potential_deps in dependencies:
                # this is happening because of the mixture of \\ and /
                if namespace_module_path.replace('\\', '/') == potential_deps.replace('\\', '/'):
                    dependencies[potential_deps].append(i_module_path)
                    reverse_dependencies[i_module_path].append(potential_deps)

    # step 3: the entire ordered dependency tree construction
    global_reverse_dependencies = {}

    def get_global_dependencies(target_module_path: str):
        if target_module_path in global_reverse_dependencies:
            return global_reverse_dependencies[target_module_path]
        else:
            global_reverse_dependencies[target_module_path] = []
            for j_module_path in reverse_dependencies[target_module_path]:
                global_reverse_dependencies[target_module_path].extend(
                    [j_module_path] + get_global_dependencies(j_module_path)
                )
            # remove duplicate
            global_reverse_dependencies[target_module_path] = \
                list(set(global_reverse_dependencies[target_module_path]))

            # make sure the import order is preserved. if a depends on b,
            # and b depends on c, the order for c should be [a, b]
            ordered_list = []
            curr_deps = \
                copy.copy(global_reverse_dependencies[target_module_path])
            while len(curr_deps) > 0:
                # find the highest order module one by one
                curr_module = curr_deps[0]
                curr_module_deps = get_global_dependencies(curr_module)
                available_modules = \
                    [i for i in curr_module_deps if i in curr_deps]
                if len(available_modules) == 0:
                    # this is a highest node. no other candidates to consider
                    pass
                else:
                    # get the highest module in this module's dependencies
                    curr_module = available_modules[0]
                ordered_list.append(curr_module)
                curr_deps.remove(curr_module)
            global_reverse_dependencies[target_module_path] = ordered_list

            return global_reverse_dependencies[target_module_path]

    for i_module_path in modules_path:
        OmniFinder._dependencies[i_module_path] = []
        i_reverse_dependencies = get_global_dependencies(i_module_path)
        for ii_reverse_dependent_module in i_reverse_dependencies:
            ii_collect_dependencies = \
                (ii_reverse_dependent_module,
                 module_names_in_sys_module[ii_reverse_dependent_module])
            OmniFinder._dependencies[i_module_path].append(ii_collect_dependencies)


def import_file(path: str) -> str:
    """
    Imports a single file and sets up the loader so it can find relative files. Returns the fullname of the module.
    """
    if io.is_file(path):
        module_path = io.containing_folder(path)
        module_name = _make_import_friendly(module_path)
        if module_name not in sys.modules:
            OmniFinder.add_folder(module_path, module_name)  # the folder is not here
            _import_module(module_path, module_name, is_dir=True)

        name = f"{module_name}.{io.split_filename(io.filename_from_path(path))[0]}"
    elif io.is_folder(path):
        # name = _make_import_friendly(io.containing_folder(path))
        name = _make_import_friendly(path)
    else:
        raise FileNotFoundError(path)

    _import_module(path, name, is_dir=False)

    return name


def remove_file(path: str) -> str:
    for file_list in OmniFinder._dependencies:
        if path in file_list:
            file_list.remove(path)


def get_dependency_list(path: str) -> List[Tuple[str, str]]:
    if path in OmniFinder._dependencies:
        return OmniFinder._dependencies[path]
    return []


def get_dependency_module_name(path: str) -> str:
    if path in OmniFinder._dependency_module_names:
        return OmniFinder._dependency_module_names[path]
    return None


def get_local_path(path: str) -> str:
    return OmniCache.convert_path(path)


def enable_omni_finder_loader():
    OmniCache.initialize()
    sys.meta_path.insert(0, OmniFinder())


def disable_omni_finder_loader():
    for item in sys.meta_path:
        if isinstance(item, OmniFinder):
            item.reset()
            sys.meta_path.remove(item)
            break
    OmniCache.shutdown()
